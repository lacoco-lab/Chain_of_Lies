#!/usr/bin/env python3
"""Remove data and artifacts for the CE CoT ablation experiment."""

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
        Path("data/RL_splits/arith_piggyback"),
        Path("data/RL_splits/arith_piggyback_control"),
        Path("data/RL_splits/seed_0"),
        Path("data/RL_splits/seed_1"),
        Path("data/RL_splits/seed_2"),
        Path("artifacts/ce_cot_ablation"),
        Path("data/ce_cot_ablation_eval_responses"),
    ]
    for target in targets:
        _remove_path(target)


if __name__ == "__main__":
    main()
