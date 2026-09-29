#!/usr/bin/env python3
"""Archive compact Qwen Goldreich reports and metrics without model checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

LEGACY_ROOTS = (
    "ce_parity_goldreich_bit_attention",
    "ce_parity_goldreich_composition_rank",
    "ce_parity_goldreich_curriculum",
    "ce_parity_goldreich_encrypted_bridge",
    "ce_parity_goldreich_final",
    "ce_parity_goldreich_trajectory",
    "ce_parity_goldreich_update_ablation",
)
KEEP_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}


def archive(artifacts_root: Path, output_root: Path) -> dict:
    copied = []
    for name in LEGACY_ROOTS:
        source_root = artifacts_root / name
        if not source_root.exists():
            continue
        for source in source_root.rglob("*"):
            if not source.is_file() or source.suffix.lower() not in KEEP_SUFFIXES:
                continue
            relative = Path(name) / source.relative_to(source_root)
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append({"path": str(relative), "bytes": source.stat().st_size})
    manifest = {
        "schema_version": 1,
        "description": "Compact metrics/report archive; model adapters and checkpoints excluded",
        "source_roots": list(LEGACY_ROOTS),
        "file_count": len(copied),
        "total_bytes": sum(item["bytes"] for item in copied),
        "files": copied,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/parity_goldreich_legacy_summary"),
    )
    args = parser.parse_args()
    result = archive(args.artifacts_root, args.output_root)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "file_count": result["file_count"],
                "total_bytes": result["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
