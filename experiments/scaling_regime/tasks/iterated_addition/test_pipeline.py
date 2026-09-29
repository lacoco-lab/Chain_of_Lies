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

from experiments.scaling_regime.tasks.iterated_addition.generate_splits import generate
from experiments.scaling_regime.tasks.iterated_addition.validate_splits import validate


def test_small_balanced_pipeline(tmp_path: Path) -> None:
    base = json.loads(
        Path(
            "experiments/scaling_regime/tasks/iterated_addition/config.json"
        ).read_text(encoding="utf-8")
    )
    base["train_examples_per_length"] = 8
    base["eval_examples_per_length"] = 3
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(base), encoding="utf-8")
    data_root = tmp_path / "data"
    manifest = generate(config_path, data_root)
    report = validate(config_path, data_root)
    assert report["valid"] is True
    assert manifest["piggyback_shared_rule"] == "all_but_one"
    assert manifest["piggyback_shared_count_by_length"] == {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }
    assert (
        len(list((data_root / "seed_0" / "knowledge_length_control").glob("*.json")))
        == 40
    )
    assert (
        len(
            list(
                (
                    data_root / "shared_eval" / "length_2" / "knowledge_length_control"
                ).glob("*.json")
            )
        )
        == 3
    )
