#!/usr/bin/env python3
"""Check prompt counts, exact lengths, method matching, and checkpoint availability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.legacy.scaling_regime.calibration.hard_checkpoint_length_sweep.evaluate import (
    METHODS,
    checkpoint_dir,
)


def private_key(record: dict) -> object:
    spec = record["spec"]
    if record["task_type"] == "s5":
        return (spec["initial_state"], tuple(spec["private_instruction_sequence"]))
    return spec["private_question"]


def _assert_exact_length(task: str, record: dict, length: int) -> None:
    spec = record["spec"]
    if task == "multiplication":
        questions = (spec["public_question"], spec["private_question"])
        operands = [
            [int(piece.strip()) for piece in question.split("*")]
            for question in questions
        ]
        if any(len(str(value)) != length for pair in operands for value in pair):
            raise RuntimeError(f"Non-exact operand length in {record['experiment_id']}")
    elif task == "s5":
        if (
            len(spec["public_instruction_sequence"]) != length
            or len(spec["private_instruction_sequence"]) != length
        ):
            raise RuntimeError(f"Non-exact S5 length in {record['experiment_id']}")
    elif len(spec["public_facts"]) != length or len(spec["private_facts"]) != length:
        raise RuntimeError(f"Non-exact knowledge length in {record['experiment_id']}")


def main(config_path: Path, prompts_root: Path, hard_root: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for task, task_config in config["tasks"].items():
        expected_lengths = {int(x) for x in task_config["lengths"]}
        expected_per_method = len(expected_lengths) * int(
            task_config["examples_per_length"]
        )
        for seed in config["seeds"]:
            records: dict[str, dict[str, dict]] = {}
            for method in METHODS:
                root = prompts_root / task / f"seed_{seed}" / method
                files = sorted(root.glob("*.json"))
                if len(files) != expected_per_method:
                    raise RuntimeError(
                        f"{root}: expected {expected_per_method}, found {len(files)}"
                    )
                records[method] = {
                    path.stem.rsplit(f"_{method}", 1)[0]: json.loads(
                        path.read_text(encoding="utf-8")
                    )
                    for path in files
                }
                observed = {
                    int(record["spec"]["evaluation_length"])
                    for record in records[method].values()
                }
                if observed != expected_lengths:
                    raise RuntimeError(
                        f"{root}: lengths {observed}, expected {expected_lengths}"
                    )
                counts = Counter(
                    int(record["spec"]["evaluation_length"])
                    for record in records[method].values()
                )
                if set(counts.values()) != {int(task_config["examples_per_length"])}:
                    raise RuntimeError(f"Unbalanced exact lengths in {root}: {counts}")
                for record in records[method].values():
                    _assert_exact_length(
                        task, record, int(record["spec"]["evaluation_length"])
                    )
            common = set(records[METHODS[0]])
            if any(set(records[method]) != common for method in METHODS[1:]):
                raise RuntimeError(
                    f"Prompt identities are not matched for {task}/seed_{seed}"
                )
            for identifier in common:
                keys = {
                    str(private_key(records[method][identifier])) for method in METHODS
                }
                if len(keys) != 1:
                    raise RuntimeError(
                        f"Private question mismatch: {task}/seed_{seed}/{identifier}"
                    )
                piggy = records["piggyback"][identifier]["spec"]
                if task == "multiplication":
                    public_left = piggy["public_question"].split("*")[0].strip()
                    private_left = piggy["private_question"].split("*")[0].strip()
                    related = public_left == private_left
                elif task == "s5":
                    related = (
                        piggy["public_instruction_sequence"][:-1]
                        == piggy["private_instruction_sequence"][:-1]
                    )
                else:
                    related = [x["entity"] for x in piggy["public_facts"][:-1]] == [
                        x["entity"] for x in piggy["private_facts"][:-1]
                    ]
                if not related:
                    raise RuntimeError(
                        f"Incorrect piggyback relation: {task}/seed_{seed}/{identifier}"
                    )
            for model in config["models"]:
                for method in METHODS:
                    checkpoint = checkpoint_dir(
                        hard_root, task, model, int(seed), method
                    )
                    if not (checkpoint / "adapter_model.safetensors").is_file():
                        raise FileNotFoundError(
                            checkpoint / "adapter_model.safetensors"
                        )
    print(
        "Validation passed: 36,000 prompts and all 72 method checkpoints are present."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--prompts-root",
        type=Path,
        default=Path("generated_data/hard_checkpoint_length_sweep"),
    )
    parser.add_argument("--hard-artifacts-root", type=Path, default=Path("Hard_regime"))
    args = parser.parse_args()
    main(args.config, args.prompts_root, args.hard_artifacts_root)
