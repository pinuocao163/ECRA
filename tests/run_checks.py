import argparse
import gc
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ecra.cognitive import APPRAISALS, HINTS, guidance, prior
from ecra.data import (
    LABELS,
    DialogueDataset,
    collate_dialogues,
    load_config,
    read_features,
)
from ecra.model import ECRA, compute_loss

ROOT = Path(__file__).resolve().parents[1]


def full_model_check(feature_root, output):
    results = []
    for dataset in ("IEMOCAP", "MELD"):
        config = load_config(ROOT / "configs" / (dataset.lower() + ".json"))
        features = read_features(
            Path(feature_root) / (dataset.lower() + "_multimodal_features.pkl")
        )
        vid = next(
            vid for vid in features["development"] if len(features["labels"][vid]) >= 4
        )
        classes = len(LABELS[dataset])
        cache = {}
        for turn, label_id in enumerate(features["labels"][vid]):
            label = LABELS[dataset][int(label_id)]
            cache[(vid, turn)] = np.concatenate(
                [
                    prior(int(label_id), 0.95, classes),
                    APPRAISALS[label],
                    HINTS[label],
                    [0.95],
                    [1.0, 0.9, 0.85, 1.0],
                    guidance(APPRAISALS[label]),
                ]
            ).astype(np.float32)
        item = DialogueDataset(features, [vid], config, cache)[0]
        item = {
            key: value[:4] if isinstance(value, torch.Tensor) else value
            for key, value in item.items()
        }
        batch = collate_dialogues([item])
        model = ECRA(config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=tuple(config["betas"]),
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
        output_model = model(batch)
        losses = compute_loss(output_model, batch, config, [[0, 1], [2, 3]])
        losses["total"].backward()
        finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite or not all(torch.isfinite(value) for value in losses.values()):
            raise AssertionError("Nonfinite full-size model loss or gradients")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            batch["available"].zero_()
            rejected = model(batch)
            if not torch.equal(rejected["base_logits"], rejected["logits"]):
                raise AssertionError(
                    "Rejected full model predictions differ from reference"
                )
        results.append(
            dict(
                dataset=dataset,
                hidden_dim=config["hidden_dim"],
                heads=config["heads"],
                parameters=sum(p.numel() for p in model.parameters()),
                real_feature_dialogue=vid,
                real_feature_utterances=4,
                cognitive_inputs="synthetic diagnostic fields, not LLM outputs",
                losses={key: float(value.detach()) for key, value in losses.items()},
                finite_gradients=True,
                adam_step=True,
                rejection_exact=True,
            )
        )
        del model, optimizer, output_model, losses, rejected, features, cache, batch
        gc.collect()
    (output / "full_model_checks.json").write_text(json.dumps(results, indent=2))
    return results


def synthetic_pipeline(output):
    rng = np.random.RandomState(2026)
    work = output / "synthetic"
    work.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "iemocap.json")
    config.update(
        hidden_dim=16,
        ff_dim=16,
        heads=4,
        text_dim=8,
        audio_dim=6,
        visual_dim=4,
        dropout=0.0,
        epochs=1,
        batch_size=3,
        seeds=[2026, 2027],
    )
    config_path = work / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    dictionaries = [{} for _ in range(10)]
    for i in range(13):
        vid = f"dialogue_{i:02d}"
        values = [
            list(range(4)),
            ["M", "F", "M", "F"],
            [(i + j) % 6 for j in range(4)],
            rng.randn(4, 8),
            rng.randn(4, 8),
            rng.randn(4, 8),
            rng.randn(4, 8),
            rng.randn(4, 6),
            rng.randn(4, 4),
            [f"Emotion statement {i} turn {j}, things are changing" for j in range(4)],
        ]
        for target, value in zip(dictionaries, values):
            target[vid] = value
    feature_path = work / "features.pkl"
    with open(feature_path, "wb") as handle:
        pickle.dump(
            dictionaries + [[f"dialogue_{i:02d}" for i in range(12)], ["dialogue_12"]],
            handle,
        )

    def run(script, *arguments):
        subprocess.run(
            [sys.executable, "-m", "ecra." + Path(script).stem, *map(str, arguments)],
            cwd=ROOT,
            check=True,
        )

    prepared = work / "prepared"
    run(
        "prepare.py",
        "splits",
        "--config",
        config_path,
        "--features",
        feature_path,
        "--output",
        prepared,
    )
    run(
        "train.py",
        "--threads",
        2,
        "pairs",
        "--work",
        prepared,
        "--device",
        "cpu",
        "--epochs",
        1,
    )
    run(
        "prepare.py",
        "prompts",
        "--config",
        config_path,
        "--features",
        feature_path,
        "--pairs",
        prepared / "pairs.json",
        "--output",
        prepared,
    )
    from ecra.cognitive import parse_response, read_jsonl

    labels = LABELS["IEMOCAP"]
    raw = []
    for item in read_jsonl(prepared / "prompts.jsonl"):
        vid, turn = item["vid"], item["turn_id"]
        i = int(vid.rsplit("_", 1)[1])
        label = labels[(i + turn + (1 if (i + turn) % 5 == 0 else 0)) % 6]
        record = parse_response(
            json.dumps(dict(label=label, confidence=0.90 + 0.01 * ((i + turn) % 5))),
            labels,
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
        raw.append(record)
    (prepared / "raw_cache.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in raw)
    )
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest.update(stage="raw", generator="SYNTHETIC_VALIDATION_ONLY")
    (prepared / "raw_cache.jsonl.meta.json").write_text(json.dumps(manifest))
    run("prepare.py", "calibrate", "--work", prepared)
    run(
        "train.py",
        "--threads",
        2,
        "fit",
        "--work",
        prepared,
        "--device",
        "cpu",
        "--epochs",
        1,
    )
    checkpoint = prepared / "runs" / "2026" / "best.pt"
    run(
        "train.py",
        "--threads",
        2,
        "evaluate",
        "--work",
        prepared,
        "--checkpoint",
        checkpoint,
        "--device",
        "cpu",
    )
    for seed in config["seeds"]:
        metrics = json.loads(
            (prepared / "runs" / str(seed) / "test_metrics.json").read_text()
        )
        if not np.isclose(metrics["accuracy_change"], metrics["flip_accuracy_change"]):
            raise AssertionError("Prediction flip counts disagree with accuracy change")
        if (
            metrics["conditional_harm"]["numerator"]
            > metrics["conditional_harm"]["denominator"]
        ):
            raise AssertionError("Invalid conditional harmful-flip count")
    return dict(
        kind="synthetic integration test only",
        prompt_preparation=True,
        calibration=True,
        five_fold_base_pair_derivation=True,
        training_seeds=[2026, 2027],
        validation_selection=True,
        checkpoint_reload=True,
        test_evaluation=True,
    )


def audit_prepared(work):
    from ecra.cognitive import read_jsonl
    from ecra.io import digest, feature_path
    from ecra.retrieval import context

    manifest = json.loads((work / "manifest.json").read_text())
    if manifest["prompts_sha256"] != digest(work / "prompts.jsonl"):
        raise AssertionError("Prepared prompt hash mismatch")
    features = read_features(feature_path(work, manifest))
    train_vids = set(manifest["splits"]["train"])
    records = read_jsonl(work / "prompts.jsonl")
    expected = {
        (vid, turn)
        for vids in manifest["splits"].values()
        for vid in vids
        for turn in range(len(features["sentences"][vid]))
    }
    if {(r["vid"], int(r["turn_id"])) for r in records} != expected:
        raise AssertionError("Prompt targets do not match dialogue partitions")
    evidence_count = 0
    for record in records:
        if len(record["retrieved_examples"]) != manifest["config"]["rag_k"]:
            raise AssertionError("Unexpected evidence count")
        for example in record["retrieved_examples"]:
            evidence_count += 1
            if example["vid"] not in train_vids or example["vid"] == record["vid"]:
                raise AssertionError("Evidence source leakage")
            expected_context = context(
                features,
                example["vid"],
                example["turn_id"],
                manifest["config"]["context_window"],
            )
            if expected_context != example["context"]:
                raise AssertionError("Evidence context contains unexpected turns")
            label = LABELS[manifest["dataset"]][
                int(features["labels"][example["vid"]][example["turn_id"]])
            ]
            if label != example["label"]:
                raise AssertionError("Evidence label mismatch")
    return dict(
        dataset=manifest["dataset"],
        prompt_records=len(records),
        evidence_records=evidence_count,
        all_targets_covered=True,
        training_only_evidence=True,
        target_dialogue_excluded=True,
        causal_evidence_contexts=True,
        evidence_labels_match=True,
        prompts_sha256=manifest["prompts_sha256"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--prepared", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
        cwd=ROOT,
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix=".verification-", dir=ROOT) as directory:
        output = Path(directory)
        full = full_model_check(args.feature_root, output) if args.feature_root else []
        integration = synthetic_pipeline(output)
        audits = [audit_prepared(work) for work in args.prepared]
        report = dict(
            method="ECRA",
            paper_sha256=load_config(ROOT / "configs" / "iemocap.json")["paper_sha256"],
            confidence_policy="split_specific_oof",
            cognitive_targets_detached=True,
            prepared_data_audits=audits,
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            full_models=full,
            integration=integration,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
