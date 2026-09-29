#!/usr/bin/env python3
"""Validate the independent, balanced, mixed-length S5 calibration data."""

from __future__ import annotations

import argparse
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

from chain_of_lies.training.shared.trainer_utils import (  # noqa: E402
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)
from chain_of_lies.variants.s5.data_generation.generate import (  # noqa: E402
    S5_SYMBOLS,
    S5_VARIANT_CONTROL,
    replay_sequence,
    spec_sequences,
)


def _load(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("*.json"))
    ]
    if not records:
        raise ValueError(f"No prompt records in {path}")
    return records


def _key(record: dict[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    spec = record["spec"]
    return (
        spec["initial_state"],
        tuple(spec["public_instruction_sequence"]),
        tuple(spec["private_instruction_sequence"]),
    )


def validate(split_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lengths = [int(value) for value in config["lengths"]]
    root = split_root / f"seed_{config['seed']}" / S5_VARIANT_CONTROL
    train = _load(root / "train_prompts")
    val = _load(root / "val_prompts")
    expected_train = int(config["train_per_length"])
    expected_val = int(config["validation_per_length"])

    for record in (*train, *val):
        if record.get("difficulty_variant") != S5_VARIANT_CONTROL:
            raise RuntimeError("Non-control record found in Medium calibration.")
        spec = record["spec"]
        initial = spec["initial_state"]
        if sorted(initial) != sorted(S5_SYMBOLS):
            raise RuntimeError(f"Invalid initial state: {initial}")
        if (
            spec["public_initial_state"] != initial
            or spec["private_initial_state"] != initial
        ):
            raise RuntimeError(
                "Public/private tasks do not share their random initial state."
            )
        public, private = spec_sequences(spec)
        length = int(spec["calibration_length"])
        if len(public) != length or len(private) != length or public == private:
            raise RuntimeError(
                f"Bad independent length-{length} pair in {record['experiment_id']}"
            )
        public_final, _ = replay_sequence(public, initial_state=initial)
        private_final, _ = replay_sequence(private, initial_state=initial)
        if (
            public_final != spec["public_answer"]
            or private_final != spec["private_answer"]
        ):
            raise RuntimeError(f"Bad answer in {record['experiment_id']}")
        cot = _canonical_public_cot_suffix(record, supervision_mode="public_cot")
        verbose = _canonical_public_cot_suffix(
            record, supervision_mode="verbose_public_cot"
        )
        filler_record = json.loads(json.dumps(record))
        filler_record["spec"]["filler_token_count"] = 64
        filler = _canonical_public_cot_suffix(
            filler_record, supervision_mode="filler_public_cot"
        )
        if (
            cot is None
            or verbose is None
            or filler is None
            or filler.count(FILLER_TOKEN_MARKER) != 1
        ):
            raise RuntimeError(
                f"Could not construct all targets for {record['experiment_id']}"
            )

    train_counts = Counter(
        int(record["spec"]["calibration_length"]) for record in train
    )
    val_counts = Counter(int(record["spec"]["calibration_length"]) for record in val)
    if train_counts != Counter({length: expected_train for length in lengths}):
        raise RuntimeError(f"Unbalanced training buckets: {train_counts}")
    if val_counts != Counter({length: expected_val for length in lengths}):
        raise RuntimeError(f"Unbalanced validation buckets: {val_counts}")
    train_keys = {_key(record) for record in train}
    val_keys = {_key(record) for record in val}
    if (
        len(train_keys) != len(train)
        or len(val_keys) != len(val)
        or train_keys & val_keys
    ):
        raise RuntimeError("Duplicate task pairs or train/validation overlap found.")
    return {
        "train_n": len(train),
        "val_n": len(val),
        "train_by_length": dict(sorted(train_counts.items())),
        "validation_by_length": dict(sorted(val_counts.items())),
        "train_val_overlap": 0,
        "protocol": "shared_random_initial_independent_sequences",
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
