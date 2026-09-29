#!/usr/bin/env python3
"""Validate Proposition-4 parity splits, targets, and complementary-twin symmetry."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from generate_splits import (  # noqa: E402
    complement,
    final_parity,
    pair_cumulative_trace,
    single_cumulative_trace,
    supervised_suffix,
)
from chain_of_lies.training.shared.trainer_utils import (  # noqa: E402
    _canonical_public_cot_suffix,
)


def _load(directory: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    if not records:
        raise ValueError(f"No prompt records in {directory}")
    return records


def validate(config_path: Path, split_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed_root = split_root / f"seed_{config['seed']}"
    condition_map = {
        str(item["name"]): str(item["trace_protocol"]) for item in config["conditions"]
    }
    inputs_by_condition_split: dict[tuple[str, str], list[str]] = {}
    twin_groups_checked = 0

    for condition, protocol in condition_map.items():
        for split, expected_n in (
            ("train", int(config["train_n"])),
            ("val", int(config["validation_n"])),
        ):
            records = _load(seed_root / condition / f"{split}_prompts")
            if len(records) != expected_n:
                raise RuntimeError(
                    f"Expected {expected_n} {condition}/{split} records, found {len(records)}"
                )
            inputs: list[str] = []
            twins: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                spec = record["spec"]
                bits = str(spec["input_bits"])
                inputs.append(bits)
                twins[str(spec["twin_id"])].append(record)
                if (
                    record.get("task_type") != "parity"
                    or record.get("variant_name") != condition
                ):
                    raise RuntimeError(
                        f"Bad type/condition in {record['experiment_id']}"
                    )
                if len(bits) != int(config["num_bits"]) or set(bits) - {"0", "1"}:
                    raise RuntimeError(
                        f"Invalid bitstring in {record['experiment_id']}"
                    )
                if int(spec["final_parity"]) != final_parity(bits):
                    raise RuntimeError(f"Bad final parity in {record['experiment_id']}")
                if spec["trace_protocol"] != protocol:
                    raise RuntimeError(f"Wrong protocol in {record['experiment_id']}")
                expected_trace = (
                    []
                    if protocol == "answer_only"
                    else (
                        pair_cumulative_trace(bits)
                        if protocol == "pair_cumulative"
                        else single_cumulative_trace(bits)
                    )
                )
                if spec["gold_trace"] != expected_trace:
                    raise RuntimeError(f"Bad trace in {record['experiment_id']}")
                expected_suffix = supervised_suffix(bits, protocol)
                if record.get("supervised_suffix") != expected_suffix:
                    raise RuntimeError(
                        f"Bad supervised suffix in {record['experiment_id']}"
                    )
                if (
                    _canonical_public_cot_suffix(
                        record, supervision_mode="record_target"
                    )
                    != expected_suffix
                ):
                    raise RuntimeError(
                        f"Shared trainer cannot load target in {record['experiment_id']}"
                    )
                if (
                    "private" in record["prompt_text"].lower()
                    or "private" in expected_suffix.lower()
                ):
                    raise RuntimeError(
                        "Parity sanity check must not introduce a private task."
                    )

            if len(inputs) != len(set(inputs)):
                raise RuntimeError(f"Duplicate inputs in {condition}/{split}")
            parity_counts = Counter(final_parity(bits) for bits in inputs)
            if parity_counts != Counter({0: expected_n // 2, 1: expected_n // 2}):
                raise RuntimeError(
                    f"Unbalanced parity in {condition}/{split}: {parity_counts}"
                )
            for twin_id, pair in twins.items():
                if len(pair) != 2:
                    raise RuntimeError(f"Twin group {twin_id} has {len(pair)} records")
                first, second = pair
                first_bits = str(first["spec"]["input_bits"])
                second_bits = str(second["spec"]["input_bits"])
                if complement(first_bits) != second_bits:
                    raise RuntimeError(f"Twin group {twin_id} is not complementary")
                if pair_cumulative_trace(first_bits) != pair_cumulative_trace(
                    second_bits
                ):
                    raise RuntimeError(
                        f"Twin group {twin_id} does not share the Proposition-4 trace"
                    )
                if final_parity(first_bits) != final_parity(second_bits):
                    raise RuntimeError(
                        f"Twin group {twin_id} does not share final parity"
                    )
                twin_groups_checked += 1
            inputs_by_condition_split[(condition, split)] = inputs

    reference = next(iter(condition_map))
    for split in ("train", "val"):
        expected_inputs = inputs_by_condition_split[(reference, split)]
        for condition in condition_map:
            if inputs_by_condition_split[(condition, split)] != expected_inputs:
                raise RuntimeError(f"Conditions do not use identical {split} inputs")
    train_inputs = set(inputs_by_condition_split[(reference, "train")])
    val_inputs = set(inputs_by_condition_split[(reference, "val")])
    if train_inputs & val_inputs:
        raise RuntimeError("Train and validation inputs overlap")

    return {
        "train_n": len(train_inputs),
        "validation_n": len(val_inputs),
        "train_validation_overlap": 0,
        "conditions": condition_map,
        "conditions_share_identical_inputs": True,
        "balanced_final_parity": True,
        "complementary_twin_groups_checked": twin_groups_checked,
        "no_private_task": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--split-root", type=Path, default=Path("generated_data/parity_prop4")
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.split_root), indent=2))


if __name__ == "__main__":
    main()
