import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

LABELS = {
    "IEMOCAP": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "MELD": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
}


def load_config(path):
    return json.loads(Path(path).read_text())


def read_features(path):
    with open(path, "rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    if len(raw) not in (12, 13):
        raise ValueError(
            "Expected the documented 12/13-field multimodal feature format"
        )
    fields = dict(
        zip(
            [
                "utterance_ids",
                "speakers",
                "labels",
                "text",
                "roberta2",
                "roberta3",
                "roberta4",
                "audio",
                "visual",
                "sentences",
            ],
            raw[:10],
        )
    )
    fields = {
        name: {str(key): value for key, value in values.items()}
        for name, values in fields.items()
    }
    fields["development"] = sorted(map(str, raw[10]))
    fields["test"] = sorted(map(str, raw[11]))
    return fields


def make_splits(features, config, partition_path=None):
    development = features["development"]
    if config["dataset"] == "MELD":
        if partition_path is None:
            raise ValueError(
                "MELD requires --partitions pointing to supplied meld_features_roberta.pkl or a train/val/test JSON manifest"
            )
        if str(partition_path).endswith(".json"):
            splits = json.loads(Path(partition_path).read_text())
        else:
            with open(partition_path, "rb") as handle:
                raw = pickle.load(handle, encoding="latin1")
            splits = dict(train=raw[8], val=raw[10], test=raw[9])
        splits = {key: list(map(str, splits[key])) for key in ("train", "val", "test")}
    else:
        shuffled = (
            np.random.RandomState(config["split_seed"])
            .permutation(development)
            .tolist()
        )
        n_val = max(1, int(0.1 * len(shuffled)))
        splits = dict(
            train=shuffled[n_val:], val=shuffled[:n_val], test=features["test"]
        )
    groups = [set(splits[key]) for key in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Dialogue partitions overlap")
    if groups[0] | groups[1] != set(development) or groups[2] != set(features["test"]):
        raise ValueError(
            "Partitions must cover the supplied development and test dialogues exactly"
        )
    return splits


def validate_features(features, config):
    for vid in features["development"] + features["test"]:
        length = len(features["sentences"][vid])
        for modality in ("text", "audio", "visual"):
            values = np.asarray(features[modality][vid])
            if (
                values.shape != (length, config[modality + "_dim"])
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"Invalid {modality} features for dialogue {vid}: {values.shape}"
                )
        if (
            len(features["labels"][vid]) != length
            or len(features["speakers"][vid]) != length
        ):
            raise ValueError(f"Inconsistent utterance alignment for {vid}")
        if any(
            int(label) not in range(len(LABELS[config["dataset"]]))
            for label in features["labels"][vid]
        ):
            raise ValueError(f"Invalid labels in {vid}")


class DialogueDataset(Dataset):
    def __init__(self, features, vids, config, cache=None):
        self.features = features
        self.vids = list(vids)
        self.config = config
        self.cache = cache or {}

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        f = self.features
        speakers = np.asarray(f["speakers"][vid])
        if speakers.ndim == 2:
            speakers = speakers.argmax(-1)
        elif self.config["dataset"] == "IEMOCAP":
            speakers = np.asarray([0 if value == "M" else 1 for value in speakers])
        else:
            speakers = speakers.astype(np.int64)
        length = len(speakers)
        c = len(LABELS[self.config["dataset"]])
        vectors = np.zeros((length, c + 15), dtype=np.float32)
        available = np.zeros(length, dtype=np.bool_)
        for turn in range(length):
            record = self.cache.get((vid, turn))
            if record is not None:
                vectors[turn] = record
                available[turn] = True
        return dict(
            vid=vid,
            text=torch.tensor(np.asarray(f["text"][vid]), dtype=torch.float32),
            audio=torch.tensor(np.asarray(f["audio"][vid]), dtype=torch.float32),
            visual=torch.tensor(np.asarray(f["visual"][vid]), dtype=torch.float32),
            speakers=torch.tensor(speakers, dtype=torch.long),
            labels=torch.tensor(f["labels"][vid], dtype=torch.long),
            cache=torch.from_numpy(vectors),
            available=torch.from_numpy(available),
        )


def collate_dialogues(items):
    result = {"vids": [item["vid"] for item in items]}
    for field in (
        "text",
        "audio",
        "visual",
        "speakers",
        "labels",
        "cache",
        "available",
    ):
        result[field] = pad_sequence([item[field] for item in items], batch_first=True)
    lengths = torch.tensor([len(item["labels"]) for item in items])
    result["mask"] = torch.arange(result["labels"].shape[1])[None, :] < lengths[:, None]
    return result
