#!/usr/bin/env python3
"""Small deterministic smoke test for the balanced Multiplication pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.scaling_regime.tasks.multiplication.generate_splits import generate
from experiments.scaling_regime.tasks.multiplication.validate_splits import validate


def main() -> None:
    source = Path(__file__).with_name("config.json")
    config = json.loads(source.read_text(encoding="utf-8"))
    config["train_examples_per_length"] = 8
    config["eval_examples_per_length"] = 4
    with tempfile.TemporaryDirectory(
        prefix="multiplication-length-smoke-"
    ) as temporary:
        root = Path(temporary)
        config_path = root / "config.json"
        data_root = root / "data"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        first = generate(config_path, data_root)
        report = validate(config_path, data_root)
        assert report["valid"]
        assert (
            first["piggyback_relation"]
            == "same_multiplier_and_small_wrapped_factor_shift"
        )
        try:
            generate(config_path, data_root)
        except FileExistsError:
            pass
        else:
            raise AssertionError(
                "Generator did not fail closed on an existing output directory."
            )
    print("multiplication length smoke test: PASS")


if __name__ == "__main__":
    main()
