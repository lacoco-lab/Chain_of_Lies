#!/usr/bin/env python3
"""Create the complete Zenodo archives on a volume with sufficient free space."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

PACKAGE_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "code_package" else PACKAGE_ROOT
DATA = ROOT / "models_data_and_evaluation_outputs"
LIGHTWEIGHT = ROOT / "release_packages" / "lightweight_supplement.zip"
CHECKPOINT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}


def is_checkpoint(path: Path) -> bool:
    return path.suffix.lower() in CHECKPOINT_SUFFIXES


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def plan() -> dict[str, object]:
    data_files = [path for path in DATA.rglob("*") if path.is_file()]
    checkpoints = [path for path in data_files if is_checkpoint(path)]
    other = [path for path in data_files if not is_checkpoint(path)]
    lightweight_code = (
        ROOT / "release_packages" / "lightweight_supplement" / "code_package"
    )
    return {
        "schema_version": 1,
        "lightweight_archive": {
            "name": LIGHTWEIGHT.name,
            "bytes": LIGHTWEIGHT.stat().st_size if LIGHTWEIGHT.is_file() else None,
            "sha256": file_digest(LIGHTWEIGHT) if LIGHTWEIGHT.is_file() else None,
        },
        "code_identity": {
            "main_code_sha256": tree_digest(ROOT / "code_package"),
            "lightweight_code_sha256": (
                tree_digest(lightweight_code) if lightweight_code.is_dir() else None
            ),
        },
        "full_data_without_checkpoints": {
            "source": "models_data_and_evaluation_outputs/ excluding checkpoint suffixes",
            "files": len(other),
            "source_bytes": sum(path.stat().st_size for path in other),
        },
        "model_checkpoints": {
            "source": "models_data_and_evaluation_outputs/ checkpoint suffixes",
            "suffixes": sorted(CHECKPOINT_SUFFIXES),
            "files": len(checkpoints),
            "source_bytes": sum(path.stat().st_size for path in checkpoints),
        },
        "complete_data_partition": len(other) + len(checkpoints) == len(data_files),
        "total_data_files": len(data_files),
        "total_data_bytes": sum(path.stat().st_size for path in data_files),
    }


def build(name: str, destination: Path, predicate) -> dict[str, int | str]:
    output = destination / name
    count = total = 0
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for path in sorted(p for p in DATA.rglob("*") if p.is_file()):
            if not predicate(path):
                continue
            archive.add(
                path, arcname=path.relative_to(ROOT).as_posix(), recursive=False
            )
            count += 1
            total += path.stat().st_size
    return {
        "archive": output.name,
        "files": count,
        "source_bytes": total,
        "archive_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    upload_plan = plan()
    plan_path = destination / "ZENODO_UPLOAD_PLAN.json"
    plan_path.write_text(json.dumps(upload_plan, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(upload_plan, indent=2))
        return
    if shutil.disk_usage(destination).free < 25 * 1024**3:
        raise SystemExit("Destination needs at least 25 GiB free.")
    if not LIGHTWEIGHT.is_file():
        raise SystemExit("Build the lightweight supplement first.")
    copied = destination / LIGHTWEIGHT.name
    if copied.resolve() != LIGHTWEIGHT.resolve():
        shutil.copy2(LIGHTWEIGHT, copied)
    results = [
        {
            "archive": copied.name,
            "files": None,
            "source_bytes": LIGHTWEIGHT.stat().st_size,
            "archive_bytes": copied.stat().st_size,
        },
        build(
            "full_data_without_checkpoints.tar.gz",
            destination,
            lambda p: not is_checkpoint(p),
        ),
        build("model_checkpoints.tar.gz", destination, is_checkpoint),
    ]
    all_files = sum(1 for p in DATA.rglob("*") if p.is_file())
    partitioned = sum(int(item["files"] or 0) for item in results)
    if partitioned != all_files:
        raise RuntimeError(f"Data partition mismatch: {partitioned} != {all_files}")
    manifest = {"schema_version": 1, "data_files": all_files, "archives": results}
    (destination / "ZENODO_UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
