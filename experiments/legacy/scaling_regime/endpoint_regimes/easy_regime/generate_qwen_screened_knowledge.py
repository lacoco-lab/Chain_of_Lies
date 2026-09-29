#!/usr/bin/env python3
"""Generate and validate the isolated Qwen screened-knowledge Easy rerun.

Run this directly in the cluster virtual environment; it does not require a GPU
or a Condor submission. The configuration selects only the knowledge task and
writes to dedicated v1 roots, leaving completed Easy v2 data untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.experiment import (
    _roots,
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.generate_splits import (
    generate,
    load_config,
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.validate_splits import (
    validate,
)

CONFIG_PATH = Path(__file__).with_name("config_qwen_screened_knowledge.json")


def main() -> None:
    config = load_config(CONFIG_PATH)
    data_root, _, _ = _roots(config)
    manifest = generate(CONFIG_PATH, data_root)
    report = validate(CONFIG_PATH, data_root)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "experiment": manifest["experiment"],
                "tasks": manifest["tasks"],
                "data_root": str(data_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
