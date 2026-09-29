#!/usr/bin/env python3
"""Validate every invariant specific to the random-initial S5 replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.training.shared.trainer_utils import (
    _canonical_public_cot_suffix,
)  # noqa: E402
from chain_of_lies.variants.s5.data_generation.generate import (  # noqa: E402
    S5_SYMBOLS,
    S5_VARIANT_CONTROL,
    S5_VARIANT_PIGGYBACK,
    replay_sequence,
    sequence_hamming_distance,
    spec_sequences,
)


def _load(prompt_dir: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(prompt_dir.glob("*.json"))
    ]
    if not records:
        raise ValueError(f"No prompts found in {prompt_dir}")
    return records


def _key(record: dict[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    spec = record["spec"]
    return (
        spec["initial_state"],
        tuple(spec["public_instruction_sequence"]),
        tuple(spec["private_instruction_sequence"]),
    )


def _validate_record(record: dict[str, Any], variant: str) -> None:
    spec = record["spec"]
    initial = spec["initial_state"]
    if sorted(initial) != sorted(S5_SYMBOLS):
        raise RuntimeError(
            f"Invalid initial permutation in {record['experiment_id']}: {initial}"
        )
    if (
        spec.get("public_initial_state") != initial
        or spec.get("private_initial_state") != initial
    ):
        raise RuntimeError(
            f"Public/private initial states are not shared in {record['experiment_id']}"
        )
    if f"Each arrangement starts as {initial}." not in record["prompt_text"]:
        raise RuntimeError(
            f"Prompt does not expose the sampled initial state in {record['experiment_id']}"
        )

    public_sequence, private_sequence = spec_sequences(spec)
    if not 10 <= len(public_sequence) <= 19 or len(public_sequence) != len(
        private_sequence
    ):
        raise RuntimeError(f"Bad sequence lengths in {record['experiment_id']}")
    public_final, _ = replay_sequence(public_sequence, initial_state=initial)
    private_final, _ = replay_sequence(private_sequence, initial_state=initial)
    if public_final != spec["public_answer"] or private_final != spec["private_answer"]:
        raise RuntimeError(
            f"Final state was not computed from {initial} in {record['experiment_id']}"
        )

    differences = sequence_hamming_distance(public_sequence, private_sequence)
    if variant == S5_VARIANT_PIGGYBACK:
        expected_position = len(public_sequence) - 1
        if differences != 1 or spec.get("changed_positions") != [expected_position]:
            raise RuntimeError(
                f"Piggyback does not differ only at the final swap in {record['experiment_id']}"
            )
    elif variant == S5_VARIANT_CONTROL:
        if spec.get("sampling_strategy") != "independent_public_private_same_length":
            raise RuntimeError(
                f"Control is not independently sampled in {record['experiment_id']}"
            )

    canonical = _canonical_public_cot_suffix(record, supervision_mode="public_cot")
    if canonical is None or f"Initial state: {initial}" not in canonical:
        raise RuntimeError(
            f"Training target ignores the random initial state in {record['experiment_id']}"
        )
    private_sequence_text = "; ".join(spec["private_instruction_sequence"])
    reasoning = canonical.split("<ANSWER>", 1)[0]
    if private_sequence_text in reasoning:
        raise RuntimeError(
            f"Public CoT exposes the private sequence in {record['experiment_id']}"
        )


def validate_split_root(
    split_root: Path, variant: str, train_n: int, val_n: int
) -> dict[str, Any]:
    variant_root = split_root / variant
    train = _load(variant_root / "train_prompts")
    val = _load(variant_root / "val_prompts")
    if len(train) != train_n or len(val) != val_n:
        raise RuntimeError(
            f"Expected {train_n}/{val_n} records, found {len(train)}/{len(val)}"
        )
    for record in (*train, *val):
        _validate_record(record, variant)
    train_keys = {_key(record) for record in train}
    val_keys = {_key(record) for record in val}
    if (
        len(train_keys) != len(train)
        or len(val_keys) != len(val)
        or train_keys & val_keys
    ):
        raise RuntimeError("Duplicate tasks or train/validation overlap detected.")
    return {
        "variant": variant,
        "train_n": len(train),
        "val_n": len(val),
        "train_val_overlap": len(train_keys & val_keys),
        "unique_initial_states": len(
            {record["spec"]["initial_state"] for record in (*train, *val)}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=[S5_VARIANT_PIGGYBACK, S5_VARIANT_CONTROL], required=True
    )
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--val-n", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_split_root(
                args.split_root, args.variant, args.train_n, args.val_n
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
