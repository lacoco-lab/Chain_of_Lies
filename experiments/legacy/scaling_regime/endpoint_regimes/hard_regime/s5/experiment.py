#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.shared_task_experiment import (
    generate,
    run_cell,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "cell"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--model", choices=("qwen", "llama"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    (
        generate("s5", args.config)
        if args.action == "generate"
        else run_cell("s5", args.config, str(args.model), int(args.seed))
    )
