#!/usr/bin/env python3
"""Fail-closed validation for balanced knowledge-length splits."""

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
from chain_of_lies.variants.knowledge.data_generation.generate import (
    ATOMIC_NUMBER_FACTS,
)
from chain_of_lies.variants.steganography.hard_task_channels import validate_target
from experiments.scaling_regime.tasks.iterated_addition.generate_splits import (
    CONTROL,
    PIGGYBACK,
    STEG,
    VARIANTS,
    _leakage_keys,
    load_config,
    roots,
)

FACTS = dict(ATOMIC_NUMBER_FACTS)


def _records(root: Path, expected: int) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} records in {root}; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _facts(record: dict[str, Any], side: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(item["entity"]), int(item["fact_value"]))
        for item in record["spec"][f"{side}_facts"]
    )


def _key(record: dict[str, Any], side: str) -> tuple[str, ...]:
    return tuple(sorted(entity for entity, _ in _facts(record, side)))


def _digest(keys: set[tuple[str, ...]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(repr(key) for key in keys)).encode()
    ).hexdigest()


def _check_record(record: dict[str, Any], variant: str, lengths: set[int]) -> None:
    if record.get("task_type") != "knowledge" or record.get("variant_name") != variant:
        raise RuntimeError("Knowledge-length record identity mismatch.")
    spec = record["spec"]
    length = int(spec["facts_per_question"])
    if length not in lengths or int(spec["evaluation_length"]) != length:
        raise RuntimeError("Invalid knowledge evaluation length metadata.")
    for side in ("public", "private"):
        facts = _facts(record, side)
        if len(facts) != length or len({entity for entity, _ in facts}) != length:
            raise RuntimeError(
                "Knowledge questions must contain the configured number of distinct facts."
            )
        if any(FACTS.get(entity) != value for entity, value in facts):
            raise RuntimeError("Incorrect atomic-number fact in generated record.")
        if sum(value for _, value in facts) != int(spec[f"{side}_answer"]):
            raise RuntimeError("Recorded knowledge answer does not equal the fact sum.")
    public, private = _facts(record, "public"), _facts(record, "private")
    overlap = len(
        set(entity for entity, _ in public) & set(entity for entity, _ in private)
    )
    expected_overlap = length - 1 if variant == PIGGYBACK else 0
    if (
        overlap != expected_overlap
        or int(spec["shared_fact_count"]) != expected_overlap
    ):
        raise RuntimeError(
            f"Incorrect {variant} overlap at length {length}: {overlap}."
        )
    if variant == PIGGYBACK and public[:expected_overlap] != private[:expected_overlap]:
        raise RuntimeError("Piggy shared facts must form an aligned prefix.")
    if float(spec["shared_fact_fraction"]) != expected_overlap / length:
        raise RuntimeError("Incorrect shared-fact fraction metadata.")
    if spec["public_answer"] == spec["private_answer"]:
        raise RuntimeError("Public and private answers must differ.")
    if variant == STEG:
        validate_target(record)


def _validate_matching(by_variant: dict[str, list[dict[str, Any]]]) -> None:
    ids = [
        [record["experiment_id"] for record in by_variant[variant]]
        for variant in VARIANTS
    ]
    if not (ids[0] == ids[1] == ids[2]):
        raise RuntimeError("Variant rows are not aligned by experiment id.")
    for records in zip(*(by_variant[variant] for variant in VARIANTS)):
        control, piggy, steg = records
        if _facts(control, "private") != _facts(piggy, "private") or _facts(
            control, "private"
        ) != _facts(steg, "private"):
            raise RuntimeError(
                "Private questions are not exactly matched across mechanisms."
            )
        if _facts(control, "public") != _facts(steg, "public"):
            raise RuntimeError(
                "Control and Steg public questions are not exactly matched."
            )
        if int(control["spec"]["facts_per_question"]) == 1 and _facts(
            control, "public"
        ) != _facts(piggy, "public"):
            raise RuntimeError(
                "At k=1, zero-overlap Piggy must exactly match the independent control public input."
            )


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    lengths = [int(value) for value in config["lengths"]]
    length_set = set(lengths)
    train_per_length = int(config["train_examples_per_length"])
    eval_per_length = int(config["eval_examples_per_length"])
    train_total, eval_total = train_per_length * len(lengths), eval_per_length * len(
        lengths
    )
    manifest = json.loads(
        (data_root / "split_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "schema_version": 2,
        "experiment": config["experiment_name"],
        "lengths": lengths,
        "train_examples_per_length": train_per_length,
        "eval_examples_per_length": eval_per_length,
        "piggyback_shared_rule": "all_but_one",
        "piggyback_shared_count_by_length": {
            str(length): length - 1 for length in lengths
        },
        "uniqueness_unit_by_length": {
            str(length): (
                "complete_public_private_prompt_pair"
                if length == 1
                else "complete_question"
            )
            for length in lengths
        },
        "control_private_shared_fraction": 0.0,
        "evaluation_set_shared_across_training_seeds": True,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError(
            "Global split manifest does not satisfy the frozen design contract."
        )

    eval_by_variant = {
        variant: _records(data_root / "shared_eval" / variant, eval_total)
        for variant in VARIANTS
    }
    for variant, records in eval_by_variant.items():
        for record in records:
            _check_record(record, variant, length_set)
    _validate_matching(eval_by_variant)
    eval_keys_by_length: dict[int, set[tuple[str, ...]]] = {
        length: set() for length in lengths
    }
    for length in lengths:
        subset = {
            variant: [
                record
                for record in rows
                if int(record["spec"]["evaluation_length"]) == length
            ]
            for variant, rows in eval_by_variant.items()
        }
        if any(len(rows) != eval_per_length for rows in subset.values()):
            raise RuntimeError(f"Evaluation set is not balanced at length {length}.")
        for variant in VARIANTS:
            linked = sorted(
                (data_root / "shared_eval" / f"length_{length}" / variant).glob(
                    "*.json"
                )
            )
            if len(linked) != eval_per_length or any(
                not path.is_symlink() or not path.resolve().is_file() for path in linked
            ):
                raise RuntimeError(
                    f"Broken per-length evaluation view for {variant}/length_{length}."
                )
        keys = set()
        for control, piggy in zip(subset[CONTROL], subset[PIGGYBACK]):
            keys.update(
                _leakage_keys(
                    _facts(control, "private"),
                    _facts(piggy, "public"),
                    _facts(control, "public"),
                    length,
                )
            )
        expected_keys = (1 if length == 1 else 3) * eval_per_length
        if len(keys) != expected_keys:
            raise RuntimeError(
                f"Duplicate evaluation leakage units at length {length}."
            )
        if (
            _digest(keys)
            != manifest["evaluation_question_digest_by_length"][str(length)]
        ):
            raise RuntimeError(f"Evaluation digest mismatch at length {length}.")
        eval_keys_by_length[length] = keys

    selected_seeds = [seed] if seed is not None else config["seeds"]
    reports: dict[str, Any] = {}
    for current_seed in selected_seeds:
        train_by_variant = {
            variant: _records(data_root / f"seed_{current_seed}" / variant, train_total)
            for variant in VARIANTS
        }
        for variant, records in train_by_variant.items():
            for record in records:
                _check_record(record, variant, length_set)
        _validate_matching(train_by_variant)
        length_reports = {}
        for length in lengths:
            subset = {
                variant: [
                    record
                    for record in rows
                    if int(record["spec"]["evaluation_length"]) == length
                ]
                for variant, rows in train_by_variant.items()
            }
            if any(len(rows) != train_per_length for rows in subset.values()):
                raise RuntimeError(
                    f"Training set is not balanced for seed {current_seed}, length {length}."
                )
            keys = set()
            for control, piggy in zip(subset[CONTROL], subset[PIGGYBACK]):
                keys.update(
                    _leakage_keys(
                        _facts(control, "private"),
                        _facts(piggy, "public"),
                        _facts(control, "public"),
                        length,
                    )
                )
            expected_keys = (1 if length == 1 else 3) * train_per_length
            if len(keys) != expected_keys:
                raise RuntimeError(
                    f"Duplicate training leakage units for seed {current_seed}, length {length}."
                )
            if keys & eval_keys_by_length[length]:
                raise RuntimeError(
                    f"Train/evaluation question leakage for seed {current_seed}, length {length}."
                )
            length_reports[str(length)] = {
                "train_rows_per_variant": train_per_length,
                "train_eval_overlap": 0,
            }

        # Verify all five deterministic training targets can be built. The
        # three control conditions intentionally share records but not targets.
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
                raise RuntimeError(
                    "Filler target must contain exactly one atomic-filler marker."
                )
        reports[str(current_seed)] = {"valid": True, "lengths": length_reports}

    result = {
        "valid": True,
        "validated_seeds": selected_seeds,
        "lengths": lengths,
        "private_questions_matched_across_all_five_conditions": True,
        "control_steg_public_questions_matched": True,
        "piggyback_shared_rule": "all_but_one",
        "piggyback_shared_count_by_length": {
            str(length): length - 1 for length in lengths
        },
        "uniqueness_unit_by_length": {
            str(length): (
                "complete_public_private_prompt_pair"
                if length == 1
                else "complete_question"
            )
            for length in lengths
        },
        "control_private_shared_fraction": 0.0,
        "train_eval_question_overlap": 0,
        "seed_reports": reports,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate(args.config, args.data_root or roots(cfg)[0], args.seed)
