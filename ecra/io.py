import hashlib
import inspect
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_records(path, records):
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def feature_path(work, manifest):
    path = Path(manifest["features"])
    return path if path.is_absolute() else (Path(work) / path).resolve()


def load_checkpoint(path, device):
    import torch

    options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)
