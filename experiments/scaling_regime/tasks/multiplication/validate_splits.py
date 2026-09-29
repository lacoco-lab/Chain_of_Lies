#!/usr/bin/env python3
"""Fail-closed validation for balanced multiplication-length splits."""

from __future__ import annotations

import argparse
import copy
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

from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)
from chain_of_lies.variants.steganography.data_generation.generate import (
    validate_steganographic_target,
)
from experiments.scaling_regime.tasks.multiplication.generate_splits import (
    CONTROL,
    PIGGYBACK,
    STEG,
    VARIANTS,
    _bounds,
    _wrap_shift,
    load_config,
    roots,
)


def _records(root: Path, expected: int) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} records in {root}; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _operands(record: dict[str, Any], side: str) -> tuple[int, int]:
    values = tuple(int(value) for value in record["spec"][f"{side}_operands"])
    if len(values) != 2:
        raise RuntimeError(
            "Multiplication records require exactly two operands per side."
        )
    return values  # type: ignore[return-value]


def _key(record: dict[str, Any], side: str) -> tuple[int, int]:
    return tuple(sorted(_operands(record, side)))


def _digest(keys: set[tuple[int, int]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(repr(key) for key in keys)).encode()
    ).hexdigest()


def _check_record(
    record: dict[str, Any], variant: str, lengths: set[int], shifts: set[int]
) -> None:
    if record.get("task_type") != "arithmetic" or record.get("variant_name") != variant:
        raise RuntimeError("Multiplication record identity mismatch.")
    spec = record["spec"]
    if spec.get("question_format") != "multiplication_only":
        raise RuntimeError("Unexpected multiplication question format.")
    digits = int(spec["operand_digits"])
    if digits not in lengths or int(spec["evaluation_length"]) != digits:
        raise RuntimeError("Invalid digit-length metadata.")
    for side in ("public", "private"):
        operands = _operands(record, side)
        if any(len(str(value)) != digits for value in operands):
            raise RuntimeError(
                f"Non-{digits}-digit operand in {record['experiment_id']}."
            )
        if f"{operands[0]} * {operands[1]}" != spec[f"{side}_question"]:
            raise RuntimeError("Question text and operand metadata disagree.")
        if operands[0] * operands[1] != int(spec[f"{side}_answer"]):
            raise RuntimeError("Incorrect multiplication answer.")
    public, private = _operands(record, "public"), _operands(record, "private")
    overlap = len(set(public) & set(private))
    expected_overlap = 1 if variant == PIGGYBACK else 0
    if (
        overlap != expected_overlap
        or int(spec["shared_operand_count"]) != expected_overlap
    ):
        raise RuntimeError(f"Incorrect operand overlap for {variant}: {overlap}.")
    if variant == PIGGYBACK:
        shift = int(spec.get("configured_public_to_private_factor_shift", 0))
        low, high = _bounds(digits)
        if shift not in shifts:
            raise RuntimeError(
                "Piggy record uses a shift outside the frozen shift distribution."
            )
        if public[0] != private[0] or private[1] != _wrap_shift(
            public[1], shift, low, high
        ):
            raise RuntimeError(
                "Piggy must use the same multiplier and configured nearby-factor shift."
            )
    if float(spec["shared_operand_fraction"]) != expected_overlap / 2:
        raise RuntimeError("Incorrect shared-operand fraction metadata.")
    if spec["public_answer"] == spec["private_answer"]:
        raise RuntimeError("Public and private answers must differ.")
    if variant == STEG:
        validate_steganographic_target(record)


def _validate_matching(by_variant: dict[str, list[dict[str, Any]]]) -> None:
    ids = [
        [record["experiment_id"] for record in by_variant[variant]]
        for variant in VARIANTS
    ]
    if not (ids[0] == ids[1] == ids[2]):
        raise RuntimeError("Variant rows are not aligned by experiment id.")
    for control, piggy, steg in zip(*(by_variant[variant] for variant in VARIANTS)):
        if not (
            _operands(control, "private")
            == _operands(piggy, "private")
            == _operands(steg, "private")
        ):
            raise RuntimeError(
                "Private multiplications are not matched across mechanisms."
            )
        if _operands(control, "public") != _operands(steg, "public"):
            raise RuntimeError(
                "Control and Steg public multiplications are not matched."
            )


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    lengths = [int(value) for value in config["operand_digits"]]
    length_set = set(lengths)
    shifts = {int(value) for value in config["piggyback_factor_shifts"]}
    train_per_length = int(config["train_examples_per_length"])
    eval_per_length = int(config["eval_examples_per_length"])
    train_total, eval_total = train_per_length * len(lengths), eval_per_length * len(
        lengths
    )
    manifest = json.loads(
        (data_root / "split_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "experiment": config["experiment_name"],
        "operand_digits": lengths,
        "train_examples_per_length": train_per_length,
        "eval_examples_per_length": eval_per_length,
        "piggyback_relation": "same_multiplier_and_small_wrapped_factor_shift",
        "piggyback_factor_shifts": list(config["piggyback_factor_shifts"]),
        "control_private_shared_operand_fraction": 0.0,
        "evaluation_set_shared_across_training_seeds": True,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("Global split manifest violates the frozen design contract.")

    eval_by_variant = {
        variant: _records(data_root / "shared_eval" / variant, eval_total)
        for variant in VARIANTS
    }
    for variant, records in eval_by_variant.items():
        for record in records:
            _check_record(record, variant, length_set, shifts)
    _validate_matching(eval_by_variant)
    eval_keys: dict[int, set[tuple[int, int]]] = {length: set() for length in lengths}
    for length in lengths:
        subset = {
            variant: [
                record
                for record in rows
                if int(record["spec"]["operand_digits"]) == length
            ]
            for variant, rows in eval_by_variant.items()
        }
        if any(len(rows) != eval_per_length for rows in subset.values()):
            raise RuntimeError(f"Unbalanced evaluation set at {length} digits.")
        for variant in VARIANTS:
            views = sorted(
                (data_root / "shared_eval" / f"length_{length}" / variant).glob(
                    "*.json"
                )
            )
            if len(views) != eval_per_length or any(
                not path.is_file() for path in views
            ):
                raise RuntimeError(
                    f"Broken per-length evaluation view for {variant}/L{length}."
                )
        keys = {
            _key(record, side)
            for record in subset[CONTROL]
            for side in ("public", "private")
        }
        keys.update(_key(record, "public") for record in subset[PIGGYBACK])
        if len(keys) != 3 * eval_per_length:
            raise RuntimeError(f"Duplicate evaluation questions at {length} digits.")
        if (
            _digest(keys)
            != manifest["evaluation_question_digest_by_length"][str(length)]
        ):
            raise RuntimeError(f"Evaluation digest mismatch at {length} digits.")
        eval_keys[length] = keys

    reports: dict[str, Any] = {}
    selected_seeds = [seed] if seed is not None else config["seeds"]
    for current_seed in selected_seeds:
        train_by_variant = {
            variant: _records(data_root / f"seed_{current_seed}" / variant, train_total)
            for variant in VARIANTS
        }
        for variant, records in train_by_variant.items():
            for record in records:
                _check_record(record, variant, length_set, shifts)
        _validate_matching(train_by_variant)
        length_reports = {}
        for length in lengths:
            subset = {
                variant: [
                    record
                    for record in rows
                    if int(record["spec"]["operand_digits"]) == length
                ]
                for variant, rows in train_by_variant.items()
            }
            if any(len(rows) != train_per_length for rows in subset.values()):
                raise RuntimeError(
                    f"Unbalanced training set for seed {current_seed}, length {length}."
                )
            keys = {
                _key(record, side)
                for record in subset[CONTROL]
                for side in ("public", "private")
            }
            keys.update(_key(record, "public") for record in subset[PIGGYBACK])
            if len(keys) != 3 * train_per_length:
                raise RuntimeError(
                    f"Duplicate training questions for seed {current_seed}, length {length}."
                )
            if keys & eval_keys[length]:
                raise RuntimeError(
                    f"Train/evaluation leakage for seed {current_seed}, length {length}."
                )
            length_reports[str(length)] = {
                "train_rows_per_variant": train_per_length,
                "train_eval_overlap": 0,
            }

        samples = {variant: rows[0] for variant, rows in train_by_variant.items()}
        for condition, condition_spec in config["conditions"].items():
            record = copy.deepcopy(samples[str(condition_spec["variant"])])
            mode = str(condition_spec["supervision_mode"])
            if mode == "filler_public_cot":
                record["spec"]["filler_token_count"] = int(
                    condition_spec["filler_token_count"]
                )
            suffix = _canonical_public_cot_suffix(record, supervision_mode=mode)
            if not suffix:
                raise RuntimeError(f"Could not construct the {condition} target.")
            if mode == "filler_public_cot" and suffix.count(FILLER_TOKEN_MARKER) != 1:
                raise RuntimeError("Filler target must contain exactly one marker.")
        reports[str(current_seed)] = {"valid": True, "lengths": length_reports}

    result = {
        "valid": True,
        "validated_seeds": selected_seeds,
        "operand_digits": lengths,
        "private_questions_matched_across_all_five_conditions": True,
        "control_steg_public_questions_matched": True,
        "piggyback_relation": "same_multiplier_and_small_wrapped_factor_shift",
        "piggyback_factor_shifts": sorted(shifts),
        "control_private_shared_operand_fraction": 0.0,
        "train_eval_question_overlap": 0,
        "seed_reports": reports,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    p.add_argument("--data-root", type=Path)
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
    validate(a.config, a.data_root or roots(cfg)[0], a.seed)
