#!/usr/bin/env python3
"""Fast sanity checks for generated S5 prompt splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))

from chain_of_lies.variants.s5.data_generation.generate import (
    S5_VARIANT_CONTROL,
    S5_VARIANT_PIGGYBACK,
    build_public_cot_prefix,
    normalized_sequence_hamming_distance,
    replay_sequence,
    sequence_hamming_distance,
    spec_sequences,
    validate_public_cot,
    validate_swap,
)


def _load_records(prompt_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(prompt_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise ValueError(f"No JSON prompts found in {prompt_dir}")
    return records


def _diagnostics(records: list[dict[str, Any]]) -> dict[str, float]:
    lengths = []
    differences = []
    normalized_distances = []
    for record in records:
        public_sequence, private_sequence = spec_sequences(record["spec"])
        lengths.append(len(public_sequence))
        differences.append(sequence_hamming_distance(public_sequence, private_sequence))
        normalized_distances.append(
            normalized_sequence_hamming_distance(public_sequence, private_sequence)
        )
    return {
        "avg_instruction_sequence_length": sum(lengths) / len(lengths),
        "avg_num_differing_positions": sum(differences) / len(differences),
        "avg_normalized_hamming_distance": sum(normalized_distances)
        / len(normalized_distances),
    }


def _validate_records(
    records: list[dict[str, Any]], *, variant: str
) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for record in records:
        if record.get("task_type") != "s5":
            raise RuntimeError(
                f"Expected task_type=s5 in {record.get('experiment_id')}"
            )
        spec = record["spec"]
        public_sequence, private_sequence = spec_sequences(spec)
        for swap in (*public_sequence, *private_sequence):
            validate_swap(swap)
        public_final, _ = replay_sequence(public_sequence)
        private_final, _ = replay_sequence(private_sequence)
        if (
            public_final != spec["public_answer"]
            or private_final != spec["private_answer"]
        ):
            raise RuntimeError(f"Bad final state in {record['experiment_id']}")
        validate_public_cot(public_sequence, build_public_cot_prefix(public_sequence))
        if public_sequence == private_sequence:
            raise RuntimeError(
                f"Identical public/private sequence in {record['experiment_id']}"
            )
        differing_positions = sequence_hamming_distance(
            public_sequence, private_sequence
        )
        if variant == S5_VARIANT_PIGGYBACK:
            changed_positions = tuple(spec.get("changed_positions", []))
            if (
                spec.get("sampling_strategy")
                == "copy_public_then_replace_final_position"
            ):
                if differing_positions != 1 or changed_positions != (
                    len(public_sequence) - 1,
                ):
                    raise RuntimeError(
                        f"Piggyback example {record['experiment_id']} did not change only the final swap."
                    )
            elif not (1 <= differing_positions <= 3):
                raise RuntimeError(
                    f"Piggyback example {record['experiment_id']} has {differing_positions} differing positions."
                )
        if (
            variant == S5_VARIANT_CONTROL
            and spec.get("sampling_strategy")
            != "independent_public_private_same_length"
        ):
            raise RuntimeError(
                f"Control example {record['experiment_id']} was not independently sampled."
            )
        key = (
            tuple(spec["public_instruction_sequence"]),
            tuple(spec["private_instruction_sequence"]),
        )
        if key in keys:
            raise RuntimeError(
                f"Duplicate public/private pair in {record['experiment_id']}"
            )
        keys.add(key)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root", type=Path, default=Path("generated_data/prompt_splits")
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=[S5_VARIANT_PIGGYBACK, S5_VARIANT_CONTROL],
        required=True,
    )
    args = parser.parse_args()

    variant_root = args.split_root / args.variant
    train_records = _load_records(variant_root / "train_prompts")
    val_records = _load_records(variant_root / "val_prompts")
    train_keys = _validate_records(train_records, variant=args.variant)
    val_keys = _validate_records(val_records, variant=args.variant)
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(
            f"Train/val overlap contains {len(overlap)} public/private pairs."
        )

    combined = train_records + val_records
    report = {
        "variant": args.variant,
        "train_n": len(train_records),
        "val_n": len(val_records),
        "train_val_overlap": len(overlap),
        "diagnostics": _diagnostics(combined),
        "samples": [
            {
                "experiment_id": record["experiment_id"],
                "public_instruction_sequence": record["spec"][
                    "public_instruction_sequence"
                ],
                "private_instruction_sequence": record["spec"][
                    "private_instruction_sequence"
                ],
                "public_answer": record["spec"]["public_answer"],
                "private_answer": record["spec"]["private_answer"],
            }
            for record in combined[:3]
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
