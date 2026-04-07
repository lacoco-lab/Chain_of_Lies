#!/usr/bin/env python3
"""
Remove RL-generated artifacts, prompt splits, and evaluation outputs so a fresh
cluster rerun starts from a clean project state.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_TARGETS = [
    Path("artifacts/rule_based_rl"),
    Path("data/RL_splits"),
    Path("data/rl_eval_responses"),
    Path("artifacts/rule_based_rl/per_variant_eval"),
    Path("artifacts/rule_based_rl/evaluation_summary_best.json"),
    Path("artifacts/rule_based_rl/evaluation_summary_best_task.json"),
    Path("chain_of_lies/rl/evaluation_summary.json"),
]


def _remove_path(path: Path) -> None:
    if not path.exists():
        print(f"[reset] skip missing: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"[reset] removed directory: {path}")
        return
    path.unlink()
    print(f"[reset] removed file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete RL artifacts/results/splits for a clean rerun.")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Optional additional path to remove. Can be passed multiple times.",
    )
    args = parser.parse_args()

    targets = DEFAULT_TARGETS + [Path(item) for item in args.target]
    for target in targets:
        _remove_path(target)


if __name__ == "__main__":
    main()
