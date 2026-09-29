#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        )
    ),
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.shared_task_summarize import (
    summarize,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    p.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/hard_regime/knowledge")
    )
    a = p.parse_args()
    print(
        json.dumps(
            {"experiment": summarize(a.config, a.artifacts_root)["experiment"]},
            indent=2,
        )
    )
