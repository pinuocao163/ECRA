import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.model_selection import KFold

from .data import LABELS

APPRAISALS = {
    "happy": [0.70, 0.45, 0.35, 0.70],
    "joy": [0.85, 0.65, 0.45, 0.75],
    "excited": [0.85, 0.90, 0.50, 0.75],
    "sad": [-0.80, -0.45, -0.55, 0.65],
    "sadness": [-0.80, -0.45, -0.55, 0.65],
    "neutral": [0.0, 0.0, 0.0, 0.60],
    "angry": [-0.75, 0.85, 0.65, 0.75],
    "anger": [-0.75, 0.85, 0.65, 0.75],
    "frustrated": [-0.65, 0.70, -0.35, 0.70],
    "fear": [-0.70, 0.80, -0.70, 0.65],
    "surprise": [0.30, 0.85, 0.10, 0.55],
    "disgust": [-0.70, 0.55, 0.20, 0.70],
}
HINTS = {
    "happy": [0.45, 0.25, 0.30],
    "joy": [0.40, 0.30, 0.30],
    "excited": [0.25, 0.55, 0.20],
    "sad": [0.45, 0.35, 0.20],
    "sadness": [0.45, 0.35, 0.20],
    "neutral": [0.55, 0.25, 0.20],
    "angry": [0.25, 0.55, 0.20],
    "anger": [0.25, 0.55, 0.20],
    "frustrated": [0.35, 0.45, 0.20],
    "fear": [0.35, 0.45, 0.20],
    "surprise": [0.35, 0.25, 0.40],
    "disgust": [0.35, 0.25, 0.40],
}


def finite_float(value, default):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def normalize_label(value, labels):
    value = str(value).strip().lower() if value is not None else ""
    if value in labels:
        return value
    aliases = {
        "sadness": "sad",
        "sad": "sadness",
        "anger": "angry",
        "angry": "anger",
        "joyful": "joy",
        "frustration": "frustrated",
    }
    mapped = aliases.get(value)
    return mapped if mapped in labels else None


def parse_response(response, labels, default_confidence=0.85):
    parsed = None
    try:
        parsed = json.loads(response)
    except ValueError:
        start, end = response.find("{"), response.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(response[start : end + 1])
            except ValueError:
                pass
    if not isinstance(parsed, dict):
        parsed = {}
    for key in ("label", "confidence", "rationale"):
        if key not in parsed:
            match = re.search(
                r"[\"\']?" + key + r"[\"\']?\s*[:=]\s*[\"\']?([^\n,;}\"\']+)",
                response,
                re.I,
            )
            if match:
                parsed[key] = match.group(1).strip()
    label = normalize_label(parsed.get("label"), labels)
    confidence = float(
        np.clip(finite_float(parsed.get("confidence"), default_confidence), 0, 1)
    )
    defaults = APPRAISALS.get(label, [0.0, 0.0, 0.0, confidence])
    hints_default = HINTS.get(label, [1 / 3] * 3)
    a = parsed.get("appraisal", {})
    a = a if isinstance(a, dict) else {}
    appraisal = [
        float(np.clip(finite_float(a.get(key), default), -1, 1))
        for key, default in zip(
            ("valence", "arousal", "dominance", "certainty"), defaults
        )
    ]
    q = parsed.get("modality_hint", {})
    q = q if isinstance(q, dict) else {}
    hints = np.array(
        [
            max(0.0, finite_float(q.get(key), default))
            for key, default in zip(("text", "audio", "visual"), hints_default)
        ]
    )
    if hints.sum() <= 0:
        hints = np.array(hints_default)
    hints /= hints.sum()
    return dict(
        label=label,
        raw_confidence=confidence,
        appraisal=appraisal,
        modality_hint=hints.tolist(),
        rationale=str(parsed.get("rationale") or ""),
        raw_response=response,
    )


def guidance(appraisal):
    v, a, d, _ = appraisal
    return [max(v, 0) * a, max(-v, 0) * max(-a, 0), max(-v, 0) * max(a, 0) * d]


def support_fields(examples, label):
    if not examples:
        return [0.0, 0.0, 0.0, 0.0]
    scores = [entry["score"] for entry in examples]
    mm = [entry["multimodal_score"] for entry in examples]
    return [
        float(np.clip((max(scores) + 1) / 2, 0, 1)),
        float(np.clip((np.mean(scores) + 1) / 2, 0, 1)),
        float(np.clip((np.mean(mm) + 1) / 2, 0, 1)),
        float(np.mean([entry["label"] == label for entry in examples])),
    ]


def prior(label_index, confidence, classes):
    peak = max(confidence, 1 / classes + 1e-6)
    values = np.full(classes, (1 - peak) / (classes - 1), dtype=np.float32)
    values[label_index] = peak
    return values


def read_jsonl(path):
    records = []
    seen = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["vid"]), int(record["turn_id"])
            if key in seen:
                raise ValueError(f"Duplicate cache key: {key}")
            seen.add(key)
            record["vid"] = key[0]
            records.append(record)
    return records


def fit_temperature(confidences, correct):
    logits = logit(np.clip(confidences, 1e-6, 1 - 1e-6))
    correct = np.asarray(correct, dtype=np.float64)

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return np.logaddexp(0.0, scaled).sum() - (correct * scaled).sum()

    result = minimize_scalar(
        objective, bounds=(-12, 12), method="bounded", options={"xatol": 1e-10}
    )
    if not result.success:
        raise RuntimeError("Confidence temperature optimization failed")
    return float(np.exp(result.x))


def calibrate_records(records, features, splits, config):
    labels = LABELS[config["dataset"]]
    allowed = {vid: key for key, vids in splits.items() for vid in vids}
    for record in records:
        vid, turn = record["vid"], int(record["turn_id"])
        if vid not in allowed or turn not in range(len(features["sentences"][vid])):
            raise ValueError(f"Unknown target {vid}/{turn}")
        examples = record["retrieved_examples"]
        for example in examples:
            if example["vid"] not in splits["train"] or example["vid"] == vid:
                raise ValueError(
                    "Retrieval leakage: evidence must come from another training dialogue"
                )
        if len(examples) != config["rag_k"]:
            raise ValueError("Unexpected retrieval evidence count")
    train_vids = np.asarray(splits["train"])
    folds = list(
        KFold(5, shuffle=True, random_state=config["fold_seed"]).split(train_vids)
    )
    temperatures = []
    heldout_fold = {}
    for fold_index, (fit_ids, heldout_ids) in enumerate(folds):
        fit_vids = set(train_vids[fit_ids])
        fitting = [r for r in records if r["vid"] in fit_vids and r["label"] in labels]
        if not fitting:
            raise ValueError("No valid training cache records for calibration fold")
        confidence = [r["raw_confidence"] for r in fitting]
        correct = [
            labels.index(r["label"])
            == int(features["labels"][r["vid"]][int(r["turn_id"])])
            for r in fitting
        ]
        temperatures.append(fit_temperature(confidence, correct))
        heldout_fold.update({str(vid): fold_index for vid in train_vids[heldout_ids]})
    for record in records:
        value = logit(np.clip(record["raw_confidence"], 1e-6, 1 - 1e-6))
        confidence_mean = float(np.mean(expit(value / np.asarray(temperatures))))
        confidence_oof = (
            float(expit(value / temperatures[heldout_fold[record["vid"]]]))
            if allowed[record["vid"]] == "train"
            else None
        )
        policy = config.get("confidence_policy", "split_specific_oof")
        if policy not in ("ensemble_mean", "split_specific_oof"):
            raise ValueError(f"Unknown confidence policy: {policy}")
        record["confidence_mean"] = confidence_mean
        record["confidence_oof"] = confidence_oof
        record["confidence"] = (
            confidence_oof
            if policy == "split_specific_oof" and confidence_oof is not None
            else confidence_mean
        )
        record["split"] = allowed[record["vid"]]
        record["support_fields"] = support_fields(
            record["retrieved_examples"], record["label"]
        )
    return records, dict(
        temperatures=temperatures,
        heldout_fold=heldout_fold,
        fold_seed=config["fold_seed"],
        numerical_log_temperature_bounds=[-12, 12],
        confidence_policy=config.get("confidence_policy", "split_specific_oof"),
    )


def load_cache(path, config, splits, features_sha256=None, prompts_sha256=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    if (
        metadata.get("stage") != "calibrated"
        or metadata["splits"] != splits
        or metadata["dataset"] != config["dataset"]
    ):
        raise ValueError("Cache provenance or dialogue partition mismatch")
    if metadata.get("config", {}).get("confidence_policy") != config.get(
        "confidence_policy"
    ):
        raise ValueError("Cache confidence policy mismatch; recalibrate the raw cache")
    if metadata.get("config", {}).get("paper_sha256") != config.get("paper_sha256"):
        raise ValueError(
            "Cache belongs to another paper revision; regenerate or migrate it explicitly"
        )
    if (
        features_sha256 is not None
        and metadata.get("features_sha256") != features_sha256
    ):
        raise ValueError("Cache feature hash does not match current input features")
    if prompts_sha256 is not None and metadata.get("prompts_sha256") != prompts_sha256:
        raise ValueError("Cache was generated from different prompts")
    labels = LABELS[config["dataset"]]
    vectors = {}
    for record in read_jsonl(path):
        if record["label"] not in labels:
            continue
        confidence = record["confidence"]
        vector = np.concatenate(
            [
                prior(labels.index(record["label"]), confidence, len(labels)),
                record["appraisal"],
                record["modality_hint"],
                [confidence],
                record["support_fields"],
                guidance(record["appraisal"]),
            ]
        ).astype(np.float32)
        if vector.shape != (len(labels) + 15,) or not np.isfinite(vector).all():
            raise ValueError("Invalid cognitive vector")
        vectors[(record["vid"], int(record["turn_id"]))] = vector
    return vectors
