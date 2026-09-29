#!/usr/bin/env python3
"""Fail-closed validation of S5 hard-regime split and target invariants."""

from __future__ import annotations

import argparse
import hashlib
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

from chain_of_lies.variants.s5.data_generation.generate import (
    parse_swap,
    replay_sequence,
)
from chain_of_lies.variants.steganography.hard_task_channels import validate_target
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.s5.generate_splits import (
    CONTROL,
    PIGGYBACK,
    STEG,
    _digest,
    calibration_reserved_keys,
    load_config,
)


def _records(root: Path, split: str, count: int) -> list[dict[str, Any]]:
    paths = sorted((root / f"{split}_prompts").glob("*.json"))
    if len(paths) != count:
        raise RuntimeError(
            f"Expected {count} records in {root}/{split}_prompts; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _key(record: dict[str, Any], side: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    spec = record["spec"]
    return str(spec["initial_state"]), tuple(
        parse_swap(text) for text in spec[f"{side}_instruction_sequence"]
    )


def _check(record: dict[str, Any], variant: str, low: int, high: int) -> None:
    if record.get("task_type") != "s5" or record.get("variant_name") != variant:
        raise RuntimeError("S5 record identity mismatch.")
    spec = record["spec"]
    initial = str(spec["initial_state"])
    if sorted(initial) != list("ABCDE"):
        raise RuntimeError("Invalid initial permutation.")
    public = _key(record, "public")[1]
    private = _key(record, "private")[1]
    if not (low <= len(public) <= high) or len(public) != len(private):
        raise RuntimeError("Invalid S5 sequence length.")
    if (
        replay_sequence(public, initial_state=initial)[0] != spec["public_answer"]
        or replay_sequence(private, initial_state=initial)[0] != spec["private_answer"]
    ):
        raise RuntimeError("Incorrect S5 answer.")
    differences = [
        index for index, pair in enumerate(zip(public, private)) if pair[0] != pair[1]
    ]
    if differences != spec["changed_positions"]:
        raise RuntimeError("Changed-position metadata mismatch.")
    if variant == PIGGYBACK and differences != [len(public) - 1]:
        raise RuntimeError("S5 piggyback must replace only the final swap.")
    if variant == STEG:
        validate_target(record)


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    global_manifest = json.loads(
        (data_root / "split_manifest.json").read_text(encoding="utf-8")
    )
    required_global = {
        "experiment": config["experiment_name"],
        "seeds": config["seeds"],
        "train_n": config["train_n"],
        "val_n": config["val_n"],
        "sequence_length_range": config["sequence_length_range"],
        "cross_seed_component_overlap": 0,
        "component_key": "(initial_state, complete_swap_sequence)",
    }
    if any(global_manifest.get(key) != value for key, value in required_global.items()):
        raise RuntimeError("Global S5 leakage contract is missing.")
    seeds = [seed] if seed is not None else config["seeds"]
    calibration_keys = calibration_reserved_keys()
    calibration_manifest = global_manifest.get("calibration_exclusion") or {}
    if (
        calibration_manifest.get("experiment") != "s5_random_initial_replication"
        or calibration_manifest.get("component_count") != len(calibration_keys)
        or calibration_manifest.get("component_digest_sha256")
        != _digest(calibration_keys)
        or calibration_manifest.get("observed_overlap") != 0
    ):
        raise RuntimeError("S5 calibration-exclusion provenance is missing or stale.")
    low, high = config["sequence_length_range"]
    all_keys: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    reports = {}
    for current_seed in seeds:
        split_keys: dict[str, set[Any]] = {"train": set(), "val": set()}
        by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for variant in (PIGGYBACK, CONTROL, STEG):
            by_variant[variant] = {}
            for split, count in (
                ("train", config["train_n"]),
                ("val", config["val_n"]),
            ):
                records = _records(
                    data_root / f"seed_{current_seed}" / variant, split, int(count)
                )
                by_variant[variant][split] = records
                for record in records:
                    _check(record, variant, low, high)
                    split_keys[split].update(
                        (_key(record, "public"), _key(record, "private"))
                    )
        expected_split_components = {
            "train": 5 * int(config["train_n"]),
            "val": 5 * int(config["val_n"]),
        }
        for split, expected_count in expected_split_components.items():
            if len(split_keys[split]) != expected_count:
                raise RuntimeError(
                    f"Expected {expected_count} unique S5 components in seed {current_seed} {split}; "
                    f"found {len(split_keys[split])}."
                )
        for split in ("train", "val"):
            pig = by_variant[PIGGYBACK][split]
            control = by_variant[CONTROL][split]
            if [(_key(r, "public"), r["spec"]["public_answer"]) for r in pig] != [
                (_key(r, "public"), r["spec"]["public_answer"]) for r in control
            ]:
                raise RuntimeError(
                    "Piggyback/control public S5 records are not matched."
                )
        if split_keys["train"] & split_keys["val"]:
            raise RuntimeError(
                f"S5 train/eval component overlap in seed {current_seed}."
            )
        seed_keys = split_keys["train"] | split_keys["val"]
        if seed_keys & calibration_keys:
            raise RuntimeError(
                f"S5 confirmatory/calibration overlap at seed {current_seed}."
            )
        if seed_keys & all_keys:
            raise RuntimeError(
                f"S5 cross-seed component overlap at seed {current_seed}."
            )
        all_keys.update(seed_keys)
        reports[str(current_seed)] = {
            "train_unique_components": len(split_keys["train"]),
            "val_unique_components": len(split_keys["val"]),
            "train_val_overlap": 0,
        }
    if seed is None:
        expected_global = (
            5 * (int(config["train_n"]) + int(config["val_n"])) * len(config["seeds"])
        )
        if len(all_keys) != expected_global:
            raise RuntimeError(
                f"Expected {expected_global} globally unique S5 components; found {len(all_keys)}."
            )
        digest = _digest(all_keys)
        if len(all_keys) != global_manifest.get(
            "unique_components"
        ) or digest != global_manifest.get("component_digest_sha256"):
            raise RuntimeError("S5 global component count/digest mismatch.")
    result = {
        "valid": True,
        "validated_seeds": seeds,
        "cross_seed_component_overlap": 0,
        "seed_reports": reports,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate(args.config, args.data_root, args.seed)
