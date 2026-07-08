#!/usr/bin/env python3
"""Remove generated data and artifacts for the CE CoT ablation experiment."""

from __future__ import annotations

import shutil
from pathlib import Path


def _remove_path(path: Path) -> None:
    if not path.exists():
        print(f"[reset] skip missing: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"[reset] removed directory: {path}")
    else:
        path.unlink()
        print(f"[reset] removed file: {path}")


def main() -> None:
    targets = [
        Path("generated_data/prompt_splits/arith_piggyback"),
        Path("generated_data/prompt_splits/arith_piggyback_control"),
        Path("generated_data/prompt_splits/seed_0"),
        Path("generated_data/prompt_splits/seed_1"),
        Path("generated_data/prompt_splits/seed_2"),
        Path("artifacts/ce_cot_ablation"),
        Path("generated_data/eval_responses/ce_cot_ablation"),
    ]
    for target in targets:
        _remove_path(target)


if __name__ == "__main__":
    main()
