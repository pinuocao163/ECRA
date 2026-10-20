import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .cognitive import APPRAISALS, load_cache
from .data import LABELS, DialogueDataset, collate_dialogues
from .model import ECRA, compute_loss
from .io import digest, feature_path, load_checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(features, vids, config, cache, shuffle):
    return DataLoader(
        DialogueDataset(features, vids, config, cache),
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_dialogues,
        num_workers=0,
    )


def move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, config, pairs, device, optimizer=None):
    model.train(optimizer is not None)
    targets, predictions, baseline, rows = [], [], [], []
    sums = {}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_loss(output, batch, config, pairs)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite objective")
            if optimizer is not None:
                losses["total"].backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
            n = int(batch["mask"].sum())
            count += n
            for name, loss in losses.items():
                sums[name] = sums.get(name, 0.0) + float(loss.detach()) * n
            predicted = output["logits"].detach().argmax(-1)
            base_predicted = output["base_logits"].detach().argmax(-1)
            valid = batch["mask"]
            targets.extend(batch["labels"][valid].cpu().tolist())
            predictions.extend(predicted[valid].cpu().tolist())
            baseline.extend(base_predicted[valid].cpu().tolist())
            if optimizer is None:
                probabilities = output["logits"].detach().softmax(-1).cpu()
                for i, vid in enumerate(raw_batch["vids"]):
                    for turn in range(int(valid[i].sum())):
                        record = dict(
                            vid=vid,
                            turn_id=turn,
                            label=int(batch["labels"][i, turn]),
                            prediction=int(predicted[i, turn]),
                            base_prediction=int(base_predicted[i, turn]),
                            probabilities=probabilities[i, turn].tolist(),
                            admitted=bool(output["admission"][i, turn]),
                        )
                        if "eta" in output:
                            record.update(
                                support=float(output["support"][i, turn]),
                                trust=float(output["trust"][i, turn]),
                                eta=float(output["eta"][i, turn]),
                                beta=output["beta"][i, turn].detach().cpu().tolist(),
                                cache_available=bool(batch["available"][i, turn]),
                                cache_confidence=float(
                                    batch["cache"][
                                        i, turn, output["logits"].size(-1) + 7
                                    ]
                                ),
                                cache_prediction=int(
                                    batch["cache"][
                                        i, turn, : output["logits"].size(-1)
                                    ].argmax()
                                ),
                            )
                        rows.append(record)
    if not count:
        raise ValueError("Empty dialogue loader")
    labels = list(range(6 if config["dataset"] == "IEMOCAP" else 7))
    y, pred, base = np.asarray(targets), np.asarray(predictions), np.asarray(baseline)
    metrics = dict(
        loss={key: value / count for key, value in sums.items()},
        accuracy=float(100 * accuracy_score(y, pred)),
        weighted_f1=float(
            100 * f1_score(y, pred, labels=labels, average="weighted", zero_division=0)
        ),
        macro_f1=float(
            100 * f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        class_f1=(
            100 * f1_score(y, pred, labels=labels, average=None, zero_division=0)
        ).tolist(),
        confusion=confusion_matrix(y, pred, labels=labels).tolist(),
        utterances=count,
        beneficial_flips=int(((base != y) & (pred == y)).sum()),
        harmful_flips=int(((base == y) & (pred != y)).sum()),
    )
    metrics.update(
        base_correct=int((base == y).sum()),
        final_correct=int((pred == y).sum()),
        accuracy_change=100 * float(((pred == y).sum() - (base == y).sum()) / count),
        flip_accuracy_change=100
        * (metrics["beneficial_flips"] - metrics["harmful_flips"])
        / count,
    )
    if rows:
        metrics["admission_rate"] = sum(row["admitted"] for row in rows) / len(rows)
        error_subset = [
            row
            for row in rows
            if row.get("cache_available")
            and row["cache_confidence"] >= 0.85
            and row["cache_prediction"] != row["label"]
        ]
        correct_base = [
            row for row in error_subset if row["base_prediction"] == row["label"]
        ]
        harmful = sum(row["prediction"] != row["label"] for row in correct_base)
        metrics["conditional_harm"] = dict(
            numerator=harmful,
            denominator=len(correct_base),
            rate=harmful / len(correct_base) if correct_base else None,
        )
    return metrics, rows


def fit(
    features,
    train_vids,
    val_vids,
    config,
    cache,
    pairs,
    seed,
    destination,
    device,
    base_only=False,
    epochs=None,
):
    seed_everything(seed)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(features, train_vids, config, cache, True)
    val_loader = make_loader(features, val_vids, config, cache, False)
    model = ECRA(config, base_only=base_only).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    best_score = -1.0
    checkpoint = destination / "best.pt"
    with open(destination / "history.jsonl", "w") as history:
        for epoch in range(epochs if epochs is not None else config["epochs"]):
            train_metrics, _ = run_epoch(
                model, train_loader, config, pairs, device, optimizer
            )
            val_metrics, _ = run_epoch(model, val_loader, config, pairs, device)
            record = dict(epoch=epoch + 1, train=train_metrics, validation=val_metrics)
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        epoch=epoch + 1,
                        train_wf1=train_metrics["weighted_f1"],
                        val_wf1=val_metrics["weighted_f1"],
                    )
                ),
                flush=True,
            )
            if val_metrics["weighted_f1"] > best_score:
                best_score = val_metrics["weighted_f1"]
                torch.save(
                    dict(
                        model=model.state_dict(),
                        config=config,
                        seed=seed,
                        epoch=epoch + 1,
                        validation=val_metrics,
                        train_vids=train_vids,
                        val_vids=val_vids,
                        pairs=pairs,
                        base_only=base_only,
                    ),
                    checkpoint,
                )
    state = load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model"])
    return model, checkpoint


def read_work(work):
    from .data import read_features, validate_features

    work = Path(work)
    manifest = json.loads((work / "manifest.json").read_text())
    if digest(feature_path(work, manifest)) != manifest["features_sha256"]:
        raise ValueError("Feature hash differs from prepared manifest")
    config = manifest["config"]
    if config.get("method") != "ECRA" or config.get("paper_sha256") != digest(
        Path(__file__).resolve().parents[1] / "main.pdf"
    ):
        raise ValueError(
            "Prepared work uses a different paper revision; prepare with the current configuration"
        )
    features = read_features(feature_path(work, manifest))
    validate_features(features, config)
    return work, manifest, config, features


def derive_pairs(args):
    from sklearn.model_selection import KFold

    work, manifest, config, features = read_work(args.work)
    train_vids = np.asarray(manifest["splits"]["train"])
    oof = []
    seed = config["seeds"][0]
    for fold_index, (fit_indices, holdout_indices) in enumerate(
        KFold(5, shuffle=True, random_state=config["fold_seed"]).split(train_vids)
    ):
        candidate = train_vids[fit_indices].tolist()
        rng = np.random.RandomState(config["split_seed"])
        shuffled = rng.permutation(candidate).tolist()
        n_val = max(1, int(0.1 * len(shuffled)))
        inner_val, inner_train = shuffled[:n_val], shuffled[n_val:]
        destination = work / "base_oof" / f"fold_{fold_index}"
        model, _ = fit(
            features,
            inner_train,
            inner_val,
            config,
            {},
            [],
            seed,
            destination,
            args.device,
            base_only=True,
            epochs=args.epochs,
        )
        _, records = run_epoch(
            model,
            make_loader(
                features, train_vids[holdout_indices].tolist(), config, {}, False
            ),
            config,
            [],
            args.device,
        )
        oof.extend(records)
        del model
    classes = len(LABELS[config["dataset"]])
    matrix = confusion_matrix(
        [r["label"] for r in oof],
        [r["prediction"] for r in oof],
        labels=list(range(classes)),
    )
    counts = matrix.sum(-1)
    prototypes = np.array(
        [APPRAISALS[label][:3] for label in LABELS[config["dataset"]]]
    )
    rankings = []
    for a in range(classes):
        for b in range(a + 1, classes):
            score = (matrix[a, b] + matrix[b, a]) / max(
                int(counts[a] + counts[b]), 1
            ) + np.exp(-np.linalg.norm(prototypes[a] - prototypes[b]))
            rankings.append((float(score), a, b))
    rankings.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    result = dict(
        pairs=[[a, b] for _, a, b in rankings[:3]],
        ranking=[dict(score=score, pair=[a, b]) for score, a, b in rankings],
        confusion=matrix.tolist(),
        seed=seed,
        fold_seed=config["fold_seed"],
        splits=manifest["splits"],
        epoch_override=args.epochs,
    )
    (work / "pairs.json").write_text(json.dumps(result, indent=2))
    (work / "base_oof" / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oof)
    )
    print(json.dumps(result, indent=2))


def train(args):
    work, manifest, config, features = read_work(args.work)
    config = copy.deepcopy(config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    splits = manifest["splits"]
    cache, pairs = {}, []
    if not args.base_only:
        cache = load_cache(
            work / "cache.jsonl",
            config,
            splits,
            manifest["features_sha256"],
            manifest["prompts_sha256"],
        )
        pair_metadata = json.loads((work / "pairs.json").read_text())
        if pair_metadata["splits"] != splits:
            raise ValueError("Confusable pair partition mismatch")
        pairs = pair_metadata["pairs"]
        prompt_pairs = manifest.get("prompt_pairs", {}).get("pairs")
        if prompt_pairs is not None and sorted(
            sorted(pair) for pair in prompt_pairs
        ) != sorted(sorted(pair) for pair in pairs):
            raise ValueError(
                "Prompt confusion pairs differ from training OOF pairs; regenerate prompts and caches"
            )
        if pair_metadata.get("epoch_override") is not None and args.epochs is None:
            raise ValueError("Reduced-epoch pairs cannot be used for a full paper run")
    all_metrics = []
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    for seed in seeds:
        destination = work / ("base_runs" if args.base_only else "runs") / str(seed)
        model, checkpoint = fit(
            features,
            splits["train"],
            splits["val"],
            config,
            cache,
            pairs,
            seed,
            destination,
            args.device,
            base_only=args.base_only,
        )
        metrics, rows = run_epoch(
            model,
            make_loader(features, splits["test"], config, cache, False),
            config,
            pairs,
            args.device,
        )
        metrics.update(
            seed=seed, checkpoint=str(checkpoint), partial_run=args.epochs is not None
        )
        (destination / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        (destination / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        all_metrics.append(metrics)
        del model
    summary = {
        metric: dict(
            mean=float(np.mean([result[metric] for result in all_metrics])),
            sample_sd=(
                float(np.std([result[metric] for result in all_metrics], ddof=1))
                if len(all_metrics) > 1
                else None
            ),
        )
        for metric in ("accuracy", "weighted_f1", "macro_f1")
    }
    summary.update(seeds=seeds, partial_run=args.epochs is not None)
    (work / ("base_summary.json" if args.base_only else "summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


def evaluate(args):
    work, manifest, _, features = read_work(args.work)
    state = load_checkpoint(args.checkpoint, args.device)
    config = state["config"]
    if config.get("paper_sha256") != manifest["config"].get("paper_sha256"):
        raise ValueError("Checkpoint belongs to another paper revision")
    splits = manifest["splits"]
    if state["train_vids"] != splits["train"] or state["val_vids"] != splits["val"]:
        raise ValueError("Checkpoint was trained on different dialogue partitions")
    model = ECRA(config, base_only=state["base_only"]).to(args.device)
    model.load_state_dict(state["model"])
    cache = (
        {}
        if state["base_only"]
        else load_cache(
            work / "cache.jsonl",
            config,
            splits,
            manifest["features_sha256"],
            manifest["prompts_sha256"],
        )
    )
    metrics, rows = run_epoch(
        model,
        make_loader(features, splits[args.split], config, cache, False),
        config,
        state["pairs"],
        args.device,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / f"{args.split}_evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, function in [
        ("pairs", derive_pairs),
        ("fit", train),
        ("evaluate", evaluate),
    ]:
        p = commands.add_parser(command)
        p.add_argument("--work", required=True)
        p.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        p.set_defaults(function=function)
        if command in ("pairs", "fit"):
            p.add_argument("--epochs", type=int)
        if command == "fit":
            p.add_argument("--seeds", type=int, nargs="+")
            p.add_argument("--base-only", action="store_true")
        if command == "evaluate":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--split", choices=["val", "test"], default="test")
            p.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.function(args)


if __name__ == "__main__":
    main()
