import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".black-cache",
    "dist",
    "outputs",
    "models",
    "data",
}


def submission_files(root, output):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in EXCLUDED or part.startswith(".verification-")
            for part in relative.parts
        ):
            continue
        if path == output or path.suffix in (".pyc", ".pyo", ".log"):
            continue
        if path.is_symlink():
            raise ValueError(f"Submission files must not be symlinks: {relative}")
        if path.is_file():
            yield path, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/ecra_submission.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    files = list(submission_files(root, output))
    if not any(relative.as_posix() == "main.pdf" for _, relative in files):
        raise ValueError("The submission must include main.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path, relative in files:
            info = zipfile.ZipInfo(
                "ecra_submission/" + relative.as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    with zipfile.ZipFile(output) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Archive checksum failed: {damaged}")
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    print(f"Created {output.name}: {len(files)} files, {output.stat().st_size} bytes")
    print(f"SHA-256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
