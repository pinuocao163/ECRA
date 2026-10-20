import argparse
import os
import json
from pathlib import Path

from .cognitive import calibrate_records, parse_response, read_jsonl
from .data import LABELS, load_config, make_splits, read_features, validate_features
from .io import digest, feature_path, write_records
from .retrieval import DualCueRetrieval, make_messages


def prepare(args):
    config = load_config(args.config)
    features = read_features(args.features)
    validate_features(features, config)
    splits = make_splits(features, config, args.partitions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        dataset=config["dataset"],
        splits=splits,
        features=os.path.relpath(Path(args.features).resolve(), output.resolve()),
        features_sha256=digest(args.features),
        config=config,
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    if args.command == "splits":
        return
    pair_path = getattr(args, "pairs", None)
    pairs = (
        [[0, 4], [3, 5], [2, 1]]
        if config["dataset"] == "IEMOCAP"
        else [[2, 1], [6, 5], [0, 3]]
    )
    if pair_path:
        pair_metadata = json.loads(Path(pair_path).read_text())
        if pair_metadata["splits"] != splits:
            raise ValueError("Prompt confusion pairs use different dialogue partitions")
        pairs = pair_metadata["pairs"]
    metadata["prompt_pairs"] = dict(
        source="training_oof" if pair_path else "paper_reported", pairs=pairs
    )
    retrieval = DualCueRetrieval(features, splits["train"], config)
    with open(output / "prompts.jsonl", "w") as handle:
        for split, vids in splits.items():
            for vid in vids:
                for turn in range(len(features["sentences"][vid])):
                    query, stats, examples = retrieval.retrieve(vid, turn)
                    record = dict(
                        vid=vid,
                        turn_id=turn,
                        split=split,
                        messages=make_messages(
                            query, stats, examples, config["dataset"], pairs
                        ),
                        retrieved_examples=examples,
                        multimodal_stats=stats,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata["retrieval"] = retrieval.metadata()
    metadata["prompts_sha256"] = digest(output / "prompts.jsonl")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                split: dict(
                    dialogues=len(vids),
                    utterances=sum(len(features["sentences"][vid]) for vid in vids),
                )
                for split, vids in splits.items()
            },
            indent=2,
        )
    )


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(args.work)
    metadata = json.loads((work / "manifest.json").read_text())
    config = metadata["config"]
    output = work / "raw_cache.jsonl"
    if metadata.get("prompts_sha256") != digest(work / "prompts.jsonl"):
        raise ValueError("Prompt hash differs from prepared manifest")
    if args.resume and output.exists():
        previous = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
        if (
            any(
                previous.get(key) != metadata.get(key)
                for key in ("features_sha256", "prompts_sha256", "splits", "config")
            )
            or previous.get("generator") != args.model_path
        ):
            raise ValueError(
                "Cannot resume a cache from different prompts, features, configuration, or generator"
            )
    existing = read_jsonl(output) if args.resume and output.exists() else []
    done = {(record["vid"], int(record["turn_id"])) for record in existing}
    prompts = [
        record
        for record in read_jsonl(work / "prompts.jsonl")
        if (record["vid"], record["turn_id"]) not in done
    ]
    if not prompts:
        print("All prompts already generated")
        return
    if args.limit is not None:
        prompts = prompts[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(
            "Cache generation requires an available CUDA device with bfloat16 support"
        )
    with torch.cuda.device(args.device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support bfloat16")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, torch_dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    batch_size = config["generation_batch_size"]
    metadata.update(
        stage="raw",
        generator=args.model_path,
        decoding=dict(
            do_sample=False,
            dtype="bfloat16",
            input_tokens=config["max_input_tokens"],
            output_tokens=config["max_new_tokens"],
            batch_size=batch_size,
        ),
    )
    output.with_suffix(".jsonl.meta.json").write_text(json.dumps(metadata, indent=2))
    with open(output, "a" if args.resume else "w") as handle:
        for start in range(0, len(prompts), batch_size):
            items = prompts[start : start + batch_size]
            chats = [
                tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=True
                )
                for item in items
            ]
            inputs = tokenizer(
                chats,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config["max_input_tokens"],
            ).to(args.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=config["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            responses = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for item, response in zip(items, responses):
                record = parse_response(
                    response, LABELS[config["dataset"]], config["default_confidence"]
                )
                record.update(
                    {
                        key: item[key]
                        for key in (
                            "vid",
                            "turn_id",
                            "split",
                            "retrieved_examples",
                            "multimodal_stats",
                        )
                    }
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"Generated {min(start + batch_size, len(prompts))}/{len(prompts)}",
                flush=True,
            )


def calibrate(args):
    work = Path(args.work)
    manifest = json.loads((work / "manifest.json").read_text())
    raw_path = work / "raw_cache.jsonl"
    metadata = json.loads(raw_path.with_suffix(".jsonl.meta.json").read_text())
    if metadata.get("stage") != "raw" or any(
        metadata.get(key) != manifest.get(key)
        for key in ("splits", "features_sha256", "prompts_sha256", "config")
    ):
        raise ValueError("Raw cache provenance does not match manifest")
    features = read_features(feature_path(work, manifest))
    if digest(feature_path(work, manifest)) != manifest["features_sha256"]:
        raise ValueError("Feature file changed since prompt preparation")
    records = read_jsonl(raw_path)
    expected = {
        (vid, turn)
        for vids in manifest["splits"].values()
        for vid in vids
        for turn in range(len(features["sentences"][vid]))
    }
    actual = {(record["vid"], int(record["turn_id"])) for record in records}
    if actual != expected:
        raise ValueError(
            f"Incomplete raw cache: missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )
    records, calibration = calibrate_records(
        records, features, manifest["splits"], manifest["config"]
    )
    output = work / "cache.jsonl"
    write_records(output, records)
    metadata.update(stage="calibrated", calibration=calibration)
    output.with_suffix(".jsonl.meta.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            dict(records=len(records), temperatures=calibration["temperatures"]),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("splits", "prompts"):
        p = commands.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--features", required=True)
        p.add_argument("--partitions")
        p.add_argument("--output", required=True)
        if command == "prompts":
            p.add_argument("--pairs")
        p.set_defaults(function=prepare)
    p = commands.add_parser("generate")
    p.add_argument("--work", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int)
    p.set_defaults(function=generate)
    p = commands.add_parser("calibrate")
    p.add_argument("--work", required=True)
    p.set_defaults(function=calibrate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
