import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


LABELS = {
    "iemocap": {"happy", "sad", "neutral", "angry", "excited", "frustrated"},
    "meld": {"neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"},
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_reports(path):
    parts = re.split(r"(?m)^seed(\d+):", path.read_text(encoding="utf-8"))
    reports = []
    signatures = collections.defaultdict(list)
    metric_pattern = r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    for index in range(1, len(parts), 2):
        seed, block = int(parts[index]), parts[index + 1]
        header = re.search(r"=====\s+\w+_([\d.]+)_([\d.]+)\s+=====", block)
        accuracy = re.search(r"(?m)^\s*accuracy\s+([\d.]+)\s+([\d.]+)\s*$", block)
        macro = re.search(r"(?m)^\s*macro avg\s+" + metric_pattern + r"\s*$", block)
        weighted = re.search(
            r"(?m)^\s*weighted avg\s+" + metric_pattern + r"\s*$", block
        )
        if any(match is None for match in (header, accuracy, macro, weighted)):
            raise ValueError(
                f"Incomplete classification report: {path.name}, seed {seed}"
            )
        rows = [
            tuple(map(float, row))
            for row in re.findall(r"(?m)^\s*(\d+)\s+" + metric_pattern + r"\s*$", block)
        ]
        if not rows:
            raise ValueError(f"Missing class rows: {path.name}, seed {seed}")
        support = sum(row[4] for row in rows)
        if support <= 0:
            raise ValueError(f"Empty support: {path.name}, seed {seed}")
        body_accuracy = float(accuracy.group(1))
        body_weighted = float(weighted.group(3))
        body_macro = float(macro.group(3))
        expected = dict(
            accuracy=sum(row[2] * row[4] for row in rows) / support,
            weighted_f1=sum(row[3] * row[4] for row in rows) / support,
            macro_f1=statistics.mean(row[3] for row in rows),
        )
        differences = dict(
            accuracy=expected["accuracy"] - body_accuracy,
            weighted_f1=expected["weighted_f1"] - body_weighted,
            macro_f1=expected["macro_f1"] - body_macro,
        )
        consistent = all(abs(value) <= 1.01e-4 for value in differences.values())
        consistent = consistent and all(
            abs(float(match.group(group)) - support) < 1e-8
            for match, group in ((accuracy, 2), (macro, 4), (weighted, 4))
        )
        for _, precision, recall, f1, count in rows:
            harmonic = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            consistent = (
                consistent
                and all(0 <= value <= 1 for value in (precision, recall, f1))
                and count >= 0
                and abs(harmonic - f1) <= 1.51e-4
            )
        header_accuracy, header_weighted = map(float, header.groups())
        header_matches = (
            abs(header_accuracy - body_accuracy * 100) < 0.0051
            and abs(header_weighted - body_weighted * 100) < 0.0051
        )
        reports.append(
            dict(
                supplied_seed_label=seed,
                accuracy_percent=100 * body_accuracy,
                weighted_f1_percent=100 * body_weighted,
                macro_f1_percent=100 * body_macro,
                support=support,
                header_accuracy_percent=header_accuracy,
                header_weighted_f1_percent=header_weighted,
                header_matches_body=header_matches,
                class_summary_consistent_with_rounding=bool(consistent),
                class_summary_differences=differences,
            )
        )
        signatures[" ".join(block[header.end() :].split())].append(seed)
    if len(reports) < 2:
        raise ValueError(f"Expected multiple reports in {path.name}")
    seed_labels = [report["supplied_seed_label"] for report in reports]
    if len(seed_labels) != len(set(seed_labels)):
        raise ValueError(f"Duplicate seed labels in {path.name}")
    summary = {}
    for metric in ("accuracy_percent", "weighted_f1_percent", "macro_f1_percent"):
        values = [report[metric] for report in reports]
        summary[metric] = dict(
            mean=statistics.mean(values), sample_sd=statistics.stdev(values)
        )
    return dict(
        source="rounded classification-report body metrics; headers audited separately",
        reports=reports,
        summary=summary,
        header_mismatch_seeds=[
            report["supplied_seed_label"]
            for report in reports
            if not report["header_matches_body"]
        ],
        all_class_summaries_consistent=all(
            report["class_summary_consistent_with_rounding"] for report in reports
        ),
        identical_report_bodies=[
            seeds for seeds in signatures.values() if len(seeds) > 1
        ],
    )


def find_report(directory, dataset):
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".txt"
        and path.stem.lower() == dataset.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one report for {dataset}, found {len(matches)}")
    return matches[0]


def audit(root):
    inventory = json.loads((root / "resources" / "inventory.json").read_text())
    for item in inventory["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Resource changed since organization: {item['path']}")
    metadata, reports = {}, {}
    for dataset in LABELS:
        metadata[dataset] = {}
        for path in sorted(
            (root / "resources" / "dataset_metadata" / dataset).iterdir()
        ):
            with path.open(encoding="utf-8", newline="") as handle:
                if path.suffix == ".csv":
                    rows = list(csv.DictReader(handle))
                    value = dict(
                        parsed_rows=len(rows),
                        dialogue_ids=len({row["Dialogue_ID"] for row in rows}),
                    )
                else:
                    value = dict(text_lines=sum(1 for _ in handle))
            metadata[dataset][path.name] = value
        reports[dataset] = audit_reports(
            find_report(root / "resources" / "results", dataset)
        )
    return dict(
        inventory_files_verified=len(inventory["files"]),
        resource_bytes=sum(item["bytes"] for item in inventory["files"]),
        dataset_metadata=metadata,
        supplied_reports=reports,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    (root / "resources" / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                files_verified=result["inventory_files_verified"],
                resource_bytes=result["resource_bytes"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
