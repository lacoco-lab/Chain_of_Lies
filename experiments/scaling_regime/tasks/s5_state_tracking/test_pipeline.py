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

from experiments.scaling_regime.tasks.s5_state_tracking.generate_splits import (
    CONTROL,
    PIGGYBACK,
    STEG,
    generate,
)
from experiments.scaling_regime.tasks.s5_state_tracking.validate_splits import validate


def test_small_balanced_pipeline(tmp_path: Path) -> None:
    base = json.loads(
        Path(
            "experiments/scaling_regime/tasks/s5_state_tracking/config.json"
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
    assert manifest["piggyback_shared_rule"] == "all_but_final_swap"
    assert manifest["piggyback_shared_prefix_by_length"] == {
        str(length): length - 1 for length in range(1, 20)
    }
    assert manifest["length_one_piggyback_control_public_matched"] is True
    assert len(list((data_root / "seed_0" / CONTROL).glob("*.json"))) == 8 * 19
    assert (
        len(list((data_root / "shared_eval" / "length_7" / PIGGYBACK).glob("*.json")))
        == 3
    )

    control = json.loads(
        next(
            (data_root / "shared_eval" / "length_1" / CONTROL).glob("*.json")
        ).read_text()
    )
    piggy = json.loads(
        (
            data_root
            / "shared_eval"
            / "length_1"
            / PIGGYBACK
            / f"{control['experiment_id']}.json"
        ).read_text()
    )
    steg = json.loads(
        (
            data_root
            / "shared_eval"
            / "length_1"
            / STEG
            / f"{control['experiment_id']}.json"
        ).read_text()
    )
    assert (
        control["spec"]["public_instruction_sequence"]
        == piggy["spec"]["public_instruction_sequence"]
    )
    assert (
        control["spec"]["private_instruction_sequence"]
        == piggy["spec"]["private_instruction_sequence"]
    )
    assert (
        control["spec"]["public_instruction_sequence"]
        == steg["spec"]["public_instruction_sequence"]
    )
