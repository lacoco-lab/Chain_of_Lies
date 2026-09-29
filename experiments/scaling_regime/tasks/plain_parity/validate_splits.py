#!/usr/bin/env python3
"""Fail-closed validation for matched plain-parity calibration splits."""

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

from chain_of_lies.variants.parity.data_generation.generate import (
    build_answer_block,
    build_public_cot_prefix,
    parity,
    running_parities,
    spec_sequences,
)
from chain_of_lies.variants.steganography.hard_task_channels import (
    build_local_channel_cot_prefix,
    validate_target,
)
from experiments.scaling_regime.tasks.plain_parity.generate_splits import (
    VARIANTS,
    load_config,
)

FILLER_TOKEN_MARKER = "<|CHAIN_OF_LIES_ATOMIC_FILLER|>"


def _parity_target(record: dict[str, Any], supervision_mode: str) -> str | None:
    """Construct parity targets without importing the PyTorch training stack."""
    spec = record.get("spec") or {}
    try:
        public, _ = spec_sequences(spec)
        answer_block = build_answer_block(
            int(spec["public_answer"]), int(spec["private_answer"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    if parity(public) != int(spec["public_answer"]):
        return None
    if supervision_mode == "answer_only":
        return answer_block
    if supervision_mode == "public_cot":
        return build_public_cot_prefix(public) + answer_block
    if supervision_mode == "filler_public_cot":
        if int(spec.get("filler_token_count", 0)) <= 0:
            return None
        return (
            build_public_cot_prefix(public) + FILLER_TOKEN_MARKER + "\n" + answer_block
        )
    if supervision_mode == "local_channel_cot":
        cot_prefix = build_local_channel_cot_prefix(record)
        return cot_prefix + answer_block if cot_prefix is not None else None
    raise ValueError(f"Unsupported parity supervision mode: {supervision_mode}")


def _load(root: Path, split: str, expected: int) -> list[dict[str, Any]]:
    paths = sorted((root / f"{split}_prompts").glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} records under {root}/{split}_prompts; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def validate(config_path: Path, split_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["seed"])
    expected = {
        "train": int(config["train_per_bucket"]) * len(config["length_buckets"]),
        "val": int(config["validation_per_bucket"]) * len(config["length_buckets"]),
    }
    by_variant = {
        variant: {
            split: _load(split_root / f"seed_{seed}" / variant, split, expected[split])
            for split in ("train", "val")
        }
        for variant in VARIANTS
    }
    bucket_lookup = {
        str(bucket["name"]): (int(bucket["min"]), int(bucket["max"]))
        for bucket in config["length_buckets"]
    }
    sequence_sets = {split: set() for split in ("train", "val")}

    for variant in VARIANTS:
        for split in ("train", "val"):
            for record in by_variant[variant][split]:
                if (
                    record.get("task_type") != "parity"
                    or record.get("variant_name") != variant
                    or record.get("difficulty_regime") != config["difficulty_regime"]
                ):
                    raise RuntimeError(f"Bad record identity in {variant}/{split}.")
                spec = record["spec"]
                public, private = spec_sequences(spec)
                length = int(spec["sequence_length"])
                low, high = bucket_lookup[str(spec["length_bucket"])]
                if len(public) != length or not low <= length <= high:
                    raise RuntimeError(
                        f"Bad length metadata in {record['experiment_id']}."
                    )
                if (
                    public == private
                    or parity(public) != spec["public_answer"]
                    or parity(private) != spec["private_answer"]
                ):
                    raise RuntimeError(
                        f"Bad parity pair or answer in {record['experiment_id']}."
                    )
                sequence_sets[split].update((public, private))
                for mode in ("answer_only", "public_cot"):
                    if _parity_target(record, mode) is None:
                        raise RuntimeError(
                            f"Missing {mode} target in {record['experiment_id']}."
                        )
                if variant == "parity_piggyback":
                    if public[:-1] != private[:-1] or public[-1] == private[-1]:
                        raise RuntimeError(
                            f"Piggyback pair does not flip only the final bit: {record['experiment_id']}."
                        )
                    if spec["public_answer"] == spec["private_answer"]:
                        raise RuntimeError(
                            f"Final-bit flip did not flip parity: {record['experiment_id']}."
                        )
                if variant == "parity_steg_local_invisible":
                    validate_target(record)
                    if _parity_target(record, "local_channel_cot") is None:
                        raise RuntimeError(
                            f"Missing steg target in {record['experiment_id']}."
                        )

    for split in ("train", "val"):
        controls = by_variant["parity_control"][split]
        piggies = by_variant["parity_piggyback"][split]
        stegs = by_variant["parity_steg_local_invisible"][split]
        for control, piggy, steg in zip(controls, piggies, stegs):
            if not (
                control["spec"]["private_bits"]
                == piggy["spec"]["private_bits"]
                == steg["spec"]["private_bits"]
            ):
                raise RuntimeError(
                    f"Private matching failed in {split}/{control['experiment_id']}."
                )
            if control["spec"]["public_bits"] != steg["spec"]["public_bits"]:
                raise RuntimeError(
                    f"Control/steg public matching failed in {split}/{control['experiment_id']}."
                )
        filler_record = json.loads(json.dumps(controls[0]))
        filler_record["spec"]["filler_token_count"] = int(
            config["conditions"]["filler"]["filler_token_count"]
        )
        filler = _parity_target(filler_record, "filler_public_cot")
        if filler is None or filler.count(FILLER_TOKEN_MARKER) != 1:
            raise RuntimeError("Could not construct the filler target.")

        for variant in VARIANTS:
            pairs = [
                (
                    tuple(record["spec"]["public_bits"]),
                    tuple(record["spec"]["private_bits"]),
                )
                for record in by_variant[variant][split]
            ]
            if len(pairs) != len(set(pairs)):
                raise RuntimeError(f"Duplicate exact pairs in {variant}/{split}.")

    overlap = sequence_sets["train"] & sequence_sets["val"]
    if overlap:
        raise RuntimeError(
            f"Found {len(overlap)} individual sequences in both train and validation."
        )
    counts = {}
    for split, per_bucket in (
        ("train", int(config["train_per_bucket"])),
        ("val", int(config["validation_per_bucket"])),
    ):
        rows = by_variant["parity_control"][split]
        bucket_counts = Counter(str(row["spec"]["length_bucket"]) for row in rows)
        label_counts = {
            bucket: Counter(
                int(row["spec"]["private_answer"])
                for row in rows
                if row["spec"]["length_bucket"] == bucket
            )
            for bucket in bucket_lookup
        }
        if bucket_counts != Counter({bucket: per_bucket for bucket in bucket_lookup}):
            raise RuntimeError(f"Unbalanced {split} length buckets: {bucket_counts}.")
        if any(values[0] != values[1] for values in label_counts.values()):
            raise RuntimeError(f"Unbalanced {split} private labels: {label_counts}.")
        for variant in VARIANTS:
            for bucket in bucket_lookup:
                public_labels = Counter(
                    int(row["spec"]["public_answer"])
                    for row in by_variant[variant][split]
                    if row["spec"]["length_bucket"] == bucket
                )
                if abs(public_labels[0] - public_labels[1]) > 1:
                    raise RuntimeError(
                        f"Unbalanced {variant}/{split}/{bucket} public labels: {public_labels}."
                    )
        counts[split] = dict(bucket_counts)
    return {
        "valid": True,
        "seed": seed,
        "train_n": expected["train"],
        "val_n": expected["val"],
        "bucket_counts": counts,
        "individual_sequence_train_val_overlap": 0,
        "private_tasks_matched_across_variants": True,
        "control_steg_public_tasks_matched": True,
        "piggyback_shared_prefix": "all but final bit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("generated_data/parity_regime_calibration"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.split_root), indent=2))


if __name__ == "__main__":
    main()
