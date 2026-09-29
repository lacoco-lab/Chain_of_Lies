#!/usr/bin/env python3
"""Validate balanced, independent multiplication Medium calibration splits."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from generate_splits import _pair_key, _task_key, _task_partition  # noqa: E402
from chain_of_lies.benchmarks import PAIRED_TASK_SYSTEM_PROMPT  # noqa: E402
from chain_of_lies.evaluation.rewards import (
    default_reward_config,
    score_completion,
)  # noqa: E402
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


def _individual_tasks(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    tasks: set[tuple[str, str]] = set()
    for record in records:
        bucket = str(record["spec"]["calibration_bucket"])
        tasks.add((bucket, _task_key(*map(int, record["spec"]["public_operands"]))))
        tasks.add((bucket, _task_key(*map(int, record["spec"]["private_operands"]))))
    return tasks


def validate(split_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    variant = str(config["variant"])
    root = split_root / f"seed_{seed}" / variant
    train = _load(root / "train_prompts")
    val = _load(root / "val_prompts")
    expected = {
        "train": int(config["train_per_bucket"]),
        "val": int(config["validation_per_bucket"]),
    }
    bucket_specs = {str(item["name"]): item for item in config["operand_buckets"]}

    for split, records in (("train", train), ("val", val)):
        for record in records:
            if (
                record.get("task_type") != "arithmetic"
                or record.get("variant_name") != variant
            ):
                raise RuntimeError(
                    f"Unexpected record type or variant in {record['experiment_id']}"
                )
            if record.get("system_prompt") != PAIRED_TASK_SYSTEM_PROMPT:
                raise RuntimeError(
                    f"Noncanonical system prompt in {record['experiment_id']}"
                )
            spec = record["spec"]
            bucket_name = str(spec["calibration_bucket"])
            bucket = bucket_specs.get(bucket_name)
            if bucket is None:
                raise RuntimeError(f"Unknown bucket {bucket_name}")
            low, high = int(bucket["min"]), int(bucket["max"])
            public_operands = tuple(map(int, spec["public_operands"]))
            private_operands = tuple(map(int, spec["private_operands"]))
            if any(
                value < low or value > high
                for value in (*public_operands, *private_operands)
            ):
                raise RuntimeError(f"Out-of-range operand in {record['experiment_id']}")
            if (
                spec["public_question"]
                != f"{public_operands[0]} * {public_operands[1]}"
            ):
                raise RuntimeError(f"Bad public question in {record['experiment_id']}")
            if (
                spec["private_question"]
                != f"{private_operands[0]} * {private_operands[1]}"
            ):
                raise RuntimeError(f"Bad private question in {record['experiment_id']}")
            if int(spec["public_answer"]) != public_operands[0] * public_operands[1]:
                raise RuntimeError(f"Bad public answer in {record['experiment_id']}")
            if int(spec["private_answer"]) != private_operands[0] * private_operands[1]:
                raise RuntimeError(f"Bad private answer in {record['experiment_id']}")
            public_key = _task_key(*public_operands)
            private_key = _task_key(*private_operands)
            if (
                public_key == private_key
                or spec["public_answer"] == spec["private_answer"]
            ):
                raise RuntimeError(
                    f"Public/private collision in {record['experiment_id']}"
                )
            if _task_partition(seed, bucket_name, public_key) != split:
                raise RuntimeError(
                    f"Public task is in the wrong split: {record['experiment_id']}"
                )
            if _task_partition(seed, bucket_name, private_key) != split:
                raise RuntimeError(
                    f"Private task is in the wrong split: {record['experiment_id']}"
                )

            for mode in ("answer_only", "public_cot", "verbose_public_cot"):
                suffix = _canonical_public_cot_suffix(record, supervision_mode=mode)
                if suffix is None:
                    raise RuntimeError(
                        f"Could not build {mode} target for {record['experiment_id']}"
                    )
                scored = score_completion(record, suffix, default_reward_config())
                if not scored.task_success or not scored.format_ok:
                    raise RuntimeError(
                        f"Invalid gold {mode} target for {record['experiment_id']}"
                    )
                reasoning = suffix.split("<ANSWER>", 1)[0]
                counterfactual = copy.deepcopy(record)
                counterfactual["spec"]["private_question"] = "99991 * 99989"
                counterfactual["spec"]["private_operands"] = [99991, 99989]
                counterfactual["spec"]["private_answer"] = 9998000099
                counterfactual_suffix = _canonical_public_cot_suffix(
                    counterfactual, supervision_mode=mode
                )
                counterfactual_reasoning = (counterfactual_suffix or "").split(
                    "<ANSWER>", 1
                )[0]
                if reasoning != counterfactual_reasoning:
                    raise RuntimeError(f"{mode} reasoning depends on the private task")

        counts = Counter(
            str(record["spec"]["calibration_bucket"]) for record in records
        )
        wanted = Counter({name: expected[split] for name in bucket_specs})
        if counts != wanted:
            raise RuntimeError(
                f"Unbalanced {split} buckets: {counts}; expected {wanted}"
            )
        keys = [_pair_key(record) for record in records]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Duplicate normalized pairs in {split}")

    train_pair_keys = {_pair_key(record) for record in train}
    val_pair_keys = {_pair_key(record) for record in val}
    train_tasks = _individual_tasks(train)
    val_tasks = _individual_tasks(val)
    if train_pair_keys & val_pair_keys:
        raise RuntimeError("Exact normalized pairs cross train/validation.")
    if train_tasks & val_tasks:
        raise RuntimeError(
            "Individual normalized multiplication tasks cross train/validation."
        )

    return {
        "train_n": len(train),
        "val_n": len(val),
        "train_by_bucket": dict(
            sorted(Counter(r["spec"]["calibration_bucket"] for r in train).items())
        ),
        "validation_by_bucket": dict(
            sorted(Counter(r["spec"]["calibration_bucket"] for r in val).items())
        ),
        "train_validation_pair_overlap": 0,
        "train_validation_individual_task_overlap": 0,
        "commutativity_normalized": True,
        "conditions_validated": ["answer_only", "public_cot", "verbose_public_cot"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.split_root, args.config), indent=2))


if __name__ == "__main__":
    main()
