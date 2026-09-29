#!/usr/bin/env python3
"""Independently fail closed on leakage or malformed confirmatory records."""

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

from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.confirmatory.generate_splits import (
    calibration_reserved_keys,
    load_config,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.range_calibration.generate_splits import (
    _audit_variant,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.range_calibration.validate_splits import (
    _question_keys,
    _records,
    _validate_record,
)


def _variants(config: dict[str, Any], family: str) -> list[str]:
    return [str(config[family]["variant"])]


def _validate_seed(
    config: dict[str, Any], data_root: Path, family: str, seed: int
) -> tuple[dict[str, Any], set[tuple[str, int, int, int]]]:
    section = config[family]
    selected_range = {"operand_range": config["operand_range"]}
    variants = _variants(config, family)
    seed_root = data_root / family / f"seed_{seed}"
    if not seed_root.is_dir():
        raise RuntimeError(f"Missing generated seed directory: {seed_root}")

    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audits: dict[str, Any] = {}
    for variant in variants:
        variant_root = seed_root / variant
        audit = _audit_variant(
            variant_root, int(config["train_n"]), int(config["val_n"])
        )
        manifest_path = variant_root / "split_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing split manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "experiment": config["experiment_name"],
            "family": family,
            "operand_range": config["operand_range"],
            "seed": seed,
            "variant": variant,
            "train_n": config["train_n"],
            "val_n": config["val_n"],
            "strict_component_disjointness_required": True,
            "cross_seed_component_disjointness_required": True,
            "train_val_individual_question_overlap": 0,
            "train_val_commutative_equivalent_overlap": 0,
            "train_val_exact_pair_overlap": 0,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Manifest mismatch in {manifest_path}: {mismatches}")
        for key in (
            "train_question_digest_sha256",
            "val_question_digest_sha256",
            "train_record_digest_sha256",
            "val_record_digest_sha256",
        ):
            if manifest.get(key) != audit[key]:
                raise RuntimeError(f"Digest mismatch for {key} in {manifest_path}.")

        records[variant] = {
            split: _records(variant_root, split) for split in ("train", "val")
        }
        for split_records in records[variant].values():
            for record in split_records:
                _validate_record(
                    record,
                    family=family,
                    variant=variant,
                    selected_range=selected_range,
                    section=section,
                )
        audits[variant] = audit

    train_keys = set().union(*(_question_keys(records[v]["train"]) for v in variants))
    val_keys = set().union(*(_question_keys(records[v]["val"]) for v in variants))
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(
            f"Suite-wide train/validation overlap for {family} seed {seed}: {len(overlap)}"
        )

    return (
        {
            "valid": True,
            "family": family,
            "seed": seed,
            "global_train_unique_questions": len(train_keys),
            "global_val_unique_questions": len(val_keys),
            "global_train_val_question_overlap": 0,
            "variant_audits": audits,
        },
        train_keys | val_keys,
    )


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    seeds = [seed] if seed is not None else [int(value) for value in config["seeds"]]
    if any(value not in config["seeds"] for value in seeds):
        raise ValueError(f"Unconfigured seed requested: {seed}")

    seed_reports: dict[str, Any] = {}
    family_seed_keys: dict[str, list[tuple[int, set[tuple[str, int, int, int]]]]] = {
        "steganography": []
    }
    for family in family_seed_keys:
        reserved = calibration_reserved_keys(config, family)
        for current_seed in seeds:
            report, keys = _validate_seed(config, data_root, family, current_seed)
            calibration_overlap = keys & reserved
            if calibration_overlap:
                raise RuntimeError(
                    f"Calibration/confirmatory component overlap in {family} seed {current_seed}: "
                    f"{len(calibration_overlap)}"
                )
            seed_reports[f"{family}/seed_{current_seed}"] = report
            family_seed_keys[family].append((current_seed, keys))

    if seed is None:
        for family, keyed_seeds in family_seed_keys.items():
            for index, (left_seed, left) in enumerate(keyed_seeds):
                for right_seed, right in keyed_seeds[index + 1 :]:
                    overlap = left & right
                    if overlap:
                        raise RuntimeError(
                            f"Cross-seed component overlap in {family}, seeds {left_seed}/{right_seed}: "
                            f"{len(overlap)}"
                        )

    result = {
        "valid": True,
        "experiment": config["experiment_name"],
        "operand_range": config["operand_range"],
        "validated_seeds": seeds,
        "cross_seed_component_overlap": (
            0 if seed is None else "certified_by_generation_manifest"
        ),
        "seed_reports": seed_reports,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate(args.config, args.data_root, args.seed)


if __name__ == "__main__":
    main()
