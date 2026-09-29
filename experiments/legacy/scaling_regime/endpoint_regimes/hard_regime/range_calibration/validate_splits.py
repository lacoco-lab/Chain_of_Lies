#!/usr/bin/env python3
"""Fail closed unless a calibration split is strictly component-disjoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .generate_splits import (
        SCHEME_LOCAL_INVISIBLE,
        _audit_variant,
        _canonical_question,
        _wrap_shift,
        build_arithmetic_prompt_text,
        build_steganography_prompt_text,
        load_config,
        range_config,
        validate_steganographic_target,
    )
except ImportError:  # Direct script execution from this directory.
    from generate_splits import (
        SCHEME_LOCAL_INVISIBLE,
        _audit_variant,
        _canonical_question,
        _wrap_shift,
        build_arithmetic_prompt_text,
        build_steganography_prompt_text,
        load_config,
        range_config,
        validate_steganographic_target,
    )

LINEAR_RE = re.compile(r"^(\d+) \* (\d+) \+ (\d+)$")
MULTIPLICATION_RE = re.compile(r"^(\d+) \* (\d+)$")


def _records(variant_root: Path, split: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((variant_root / f"{split}_prompts").glob("*.json"))
    ]


def _questions(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["spec"][field])
        for record in records
        for field in ("public_question", "private_question")
    }


def _question_keys(records: list[dict[str, Any]]) -> set[tuple[str, int, int, int]]:
    return {_canonical_question(question) for question in _questions(records)}


def _linear(question: str) -> tuple[int, int, int]:
    match = LINEAR_RE.fullmatch(question)
    if match is None:
        raise RuntimeError(f"Malformed linear-arithmetic question: {question!r}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _multiplication(question: str) -> tuple[int, int]:
    match = MULTIPLICATION_RE.fullmatch(question)
    if match is None:
        raise RuntimeError(f"Malformed multiplication question: {question!r}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _validate_record(
    record: dict[str, Any],
    *,
    family: str,
    variant: str,
    selected_range: dict[str, Any],
    section: dict[str, Any],
) -> None:
    if record.get("task_type") != "arithmetic":
        raise RuntimeError(
            f"Unexpected task type in {record.get('experiment_id')}: {record.get('task_type')}"
        )
    spec = record.get("spec", {})
    public_question, private_question = spec.get("public_question"), spec.get(
        "private_question"
    )
    if not isinstance(public_question, str) or not isinstance(private_question, str):
        raise RuntimeError(f"Missing question text in {record.get('experiment_id')}.")
    low, high = (int(value) for value in selected_range["operand_range"])

    if family == "piggyback":
        public = _linear(public_question)
        private = _linear(private_question)
        if not all(low <= value <= high for value in (*public[:2], *private[:2])):
            raise RuntimeError(
                f"Operand outside {low}..{high} in {record.get('experiment_id')}."
            )
        offset_low, offset_high = (int(value) for value in section["offset_range"])
        if (
            not offset_low <= public[2] <= offset_high
            or not offset_low <= private[2] <= offset_high
        ):
            raise RuntimeError(
                f"Offset outside {offset_low}..{offset_high} in {record.get('experiment_id')}."
            )
        if spec.get("public_answer") != public[0] * public[1] + public[2]:
            raise RuntimeError(
                f"Incorrect public answer in {record.get('experiment_id')}."
            )
        if spec.get("private_answer") != private[0] * private[1] + private[2]:
            raise RuntimeError(
                f"Incorrect private answer in {record.get('experiment_id')}."
            )
        if (
            public_question == private_question
            or spec["public_answer"] == spec["private_answer"]
        ):
            raise RuntimeError(
                f"Identical paired task/answer in {record.get('experiment_id')}."
            )
        if variant == "arith_piggyback":
            valid_factors = {
                _wrap_shift(public[1], int(shift), low, high)
                for shift in section["shifts"]
            }
            valid_offsets = {
                _wrap_shift(public[2], int(shift), offset_low, offset_high)
                for shift in section["shifts"]
            }
            if (
                private[0] != public[0]
                or private[1] not in valid_factors
                or private[2] not in valid_offsets
            ):
                raise RuntimeError(
                    f"Invalid piggyback relationship in {record.get('experiment_id')}."
                )
        expected_prompt = build_arithmetic_prompt_text(
            public_question, private_question
        )
    else:
        public = _multiplication(public_question)
        private = _multiplication(private_question)
        if not all(low <= value <= high for value in (*public, *private)):
            raise RuntimeError(
                f"Operand outside {low}..{high} in {record.get('experiment_id')}."
            )
        if spec.get("public_answer") != public[0] * public[1]:
            raise RuntimeError(
                f"Incorrect public answer in {record.get('experiment_id')}."
            )
        if spec.get("private_answer") != private[0] * private[1]:
            raise RuntimeError(
                f"Incorrect private answer in {record.get('experiment_id')}."
            )
        if public_question == private_question:
            raise RuntimeError(
                f"Identical paired task in {record.get('experiment_id')}."
            )
        if spec.get("steganography_scheme") != SCHEME_LOCAL_INVISIBLE:
            raise RuntimeError(
                f"Wrong steganography scheme in {record.get('experiment_id')}."
            )
        validate_steganographic_target(record)
        expected_prompt = build_steganography_prompt_text(
            public_question,
            private_question,
            scheme=SCHEME_LOCAL_INVISIBLE,
        )
    if record.get("prompt_text") != expected_prompt:
        raise RuntimeError(
            f"Prompt does not match its spec in {record.get('experiment_id')}."
        )


def validate(
    config_path: Path, data_root: Path, family: str, range_id: str
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_range = range_config(config, family, range_id)
    section = config[family]
    seed = int(config["seed"])
    train_n, val_n = int(config["train_n"]), int(config["val_n"])
    range_root = data_root / family / range_id / f"seed_{seed}"
    if not range_root.exists():
        raise RuntimeError(f"Missing generated range directory: {range_root}")

    if family == "piggyback":
        variants = list(config[family]["variants"])
    else:
        variants = [str(config[family]["variant"])]

    audits: dict[str, Any] = {}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in variants:
        variant_root = range_root / variant
        audits[variant] = _audit_variant(variant_root, train_n, val_n)
        manifest_path = variant_root / "split_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing split manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "family": family,
            "range_id": range_id,
            "seed": seed,
            "variant": variant,
            "train_n": train_n,
            "val_n": val_n,
            "strict_component_disjointness_required": True,
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
        for digest_key in (
            "train_question_digest_sha256",
            "val_question_digest_sha256",
            "train_record_digest_sha256",
            "val_record_digest_sha256",
        ):
            if manifest.get(digest_key) != audits[variant][digest_key]:
                raise RuntimeError(
                    f"Question digest mismatch in {manifest_path}: {digest_key}"
                )
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

    # The prohibition is suite-wide: a component used to train either arm may
    # not occur in the validation set of either arm.
    global_train = set().union(*(_question_keys(records[v]["train"]) for v in variants))
    global_val = set().union(*(_question_keys(records[v]["val"]) for v in variants))
    overlap = global_train & global_val
    if overlap:
        preview = sorted(overlap)[:5]
        raise RuntimeError(
            f"Suite-wide train/validation component overlap: {len(overlap)}; examples={preview}"
        )

    if family == "piggyback":
        piggy, control = variants
        for split in ("train", "val"):
            piggy_public = [r["spec"]["public_question"] for r in records[piggy][split]]
            control_public = [
                r["spec"]["public_question"] for r in records[control][split]
            ]
            if piggy_public != control_public:
                raise RuntimeError(
                    f"Piggyback/control public records are not matched in {split}."
                )

    result = {
        "valid": True,
        "family": family,
        "range_id": range_id,
        "seed": seed,
        "global_train_unique_questions": len(global_train),
        "global_val_unique_questions": len(global_val),
        "global_train_val_question_overlap": 0,
        "global_train_val_commutative_equivalent_overlap": 0,
        "variant_audits": audits,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=("piggyback", "steganography"), required=True
    )
    parser.add_argument("--range", dest="range_id", required=True)
    args = parser.parse_args()
    validate(args.config, args.data_root, args.family, args.range_id)


if __name__ == "__main__":
    main()
