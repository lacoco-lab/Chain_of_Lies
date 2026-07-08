#!/usr/bin/env python3
"""Remove redundant CE ablation artifact files while keeping eval-safe checkpoints.

The training code saves both:

- ``seed_X/<mode>/<variant>/`` as the latest adapter, and
- ``seed_X/<mode>/<variant>/ckpt_task/`` as the selected checkpoint used by eval.

For the current CE ablations, downstream evaluation and sharing only need the selected
``ckpt_task`` checkpoint plus compact top-level training history/metadata. This script removes the
duplicate top-level adapter/tokenizer files from each variant directory and leaves summaries,
evaluation JSONs, and ``ckpt_task`` untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path


KEEP_TOP_LEVEL_FILES = {"train_history.json", "training_metadata.json"}


def _variant_dirs(root: Path) -> list[Path]:
    return sorted(
        variant_dir
        for seed_dir in root.glob("seed_*")
        if seed_dir.is_dir()
        for mode_dir in seed_dir.iterdir()
        if mode_dir.is_dir()
        for variant_dir in mode_dir.iterdir()
        if variant_dir.is_dir() and (variant_dir / "ckpt_task").is_dir()
    )


def prune_root(root: Path, *, execute: bool) -> list[Path]:
    removable: list[Path] = []
    for variant_dir in _variant_dirs(root):
        for path in sorted(variant_dir.iterdir()):
            if path.is_dir() or path.name in KEEP_TOP_LEVEL_FILES:
                continue
            removable.append(path)

    if execute:
        for path in removable:
            path.unlink()
    return removable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/ce_cot_ablation"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, only prints what would be removed.",
    )
    args = parser.parse_args()

    removable = prune_root(args.root, execute=args.execute)
    action = "removed" if args.execute else "would remove"
    for path in removable:
        print(f"[prune] {action} {path}")
    print(f"[prune] {action} {len(removable)} files under {args.root}")


if __name__ == "__main__":
    main()
