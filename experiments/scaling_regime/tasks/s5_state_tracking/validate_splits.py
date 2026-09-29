#!/usr/bin/env python3
"""Fail-closed validation for the balanced full-range S5 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from experiments.scaling_regime.tasks.s5_state_tracking.generate_splits import (
    CONTROL,
    PIGGYBACK,
    STEG,
    VARIANTS,
    _component_key,
    _digest,
    _pair_key,
    load_config,
)


def _records(root: Path, expected: int) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} records in {root}; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _sequences(
    record: dict[str, Any],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    spec = record["spec"]
    return (
        tuple(parse_swap(value) for value in spec["public_instruction_sequence"]),
        tuple(parse_swap(value) for value in spec["private_instruction_sequence"]),
    )


def _check_record(
    record: dict[str, Any], variant: str, expected_length: int | None = None
) -> None:
    if record.get("task_type") != "s5" or record.get("variant_name") != variant:
        raise RuntimeError("S5 record identity mismatch.")
    spec = record["spec"]
    length = int(spec["evaluation_length"])
    if expected_length is not None and length != expected_length:
        raise RuntimeError("S5 per-length view contains the wrong length.")
    if int(spec["sequence_length"]) != length or length not in range(1, 20):
        raise RuntimeError("Invalid S5 sequence length.")
    initial = str(spec["initial_state"])
    if sorted(initial) != list("ABCDE"):
        raise RuntimeError("Invalid S5 initial permutation.")
    public, private = _sequences(record)
    if len(public) != length or len(private) != length:
        raise RuntimeError("Public/private S5 lengths are not matched.")
    public_answer = replay_sequence(public, initial_state=initial)[0]
    private_answer = replay_sequence(private, initial_state=initial)[0]
    if (
        public_answer != spec["public_answer"]
        or private_answer != spec["private_answer"]
    ):
        raise RuntimeError("Stored S5 answer does not match replay.")
    if public_answer == private_answer:
        raise RuntimeError(
            "Public/private final states must differ to prevent answer-copy shortcuts."
        )
    differences = [
        index for index, pair in enumerate(zip(public, private)) if pair[0] != pair[1]
    ]
    if differences != spec["changed_positions"] or len(differences) != int(
        spec["num_differing_positions"]
    ):
        raise RuntimeError("S5 difference metadata mismatch.")
    shared_prefix = next(
        (
            index
            for index, pair in enumerate(zip(public, private))
            if pair[0] != pair[1]
        ),
        length,
    )
    if shared_prefix != int(spec["shared_prefix_length"]):
        raise RuntimeError("S5 shared-prefix metadata mismatch.")
    if variant == PIGGYBACK:
        if differences != [length - 1] or shared_prefix != length - 1:
            raise RuntimeError("S5 Piggy must differ only in the final swap.")
    if variant == STEG:
        validate_target(record)


def _index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(record["experiment_id"]): record for record in records}
    if len(indexed) != len(records):
        raise RuntimeError("Duplicate experiment IDs in S5 split.")
    return indexed


def _validate_matched_split(
    by_variant: dict[str, list[dict[str, Any]]],
    expected_per_length: int,
) -> tuple[dict[int, set[Any]], dict[int, set[Any]]]:
    indexed = {variant: _index(records) for variant, records in by_variant.items()}
    if any(set(indexed[variant]) != set(indexed[CONTROL]) for variant in VARIANTS):
        raise RuntimeError("S5 experiment IDs are not matched across variants.")
    length_counts: dict[str, Counter[int]] = {
        variant: Counter() for variant in VARIANTS
    }
    components: dict[int, set[Any]] = defaultdict(set)
    pairs: dict[int, set[Any]] = defaultdict(set)
    row_count: Counter[int] = Counter()
    for experiment_id in sorted(indexed[CONTROL]):
        control = indexed[CONTROL][experiment_id]
        piggy = indexed[PIGGYBACK][experiment_id]
        steg = indexed[STEG][experiment_id]
        for variant, record in ((CONTROL, control), (PIGGYBACK, piggy), (STEG, steg)):
            _check_record(record, variant)
            length_counts[variant][int(record["spec"]["evaluation_length"])] += 1
        control_public, control_private = _sequences(control)
        piggy_public, piggy_private = _sequences(piggy)
        steg_public, steg_private = _sequences(steg)
        initials = {
            str(record["spec"]["initial_state"]) for record in (control, piggy, steg)
        }
        if len(initials) != 1:
            raise RuntimeError("Initial state is not matched across mechanisms.")
        initial = initials.pop()
        if not (control_private == piggy_private == steg_private):
            raise RuntimeError(
                "Private S5 question is not matched across all five conditions."
            )
        if control_public != steg_public:
            raise RuntimeError("Control and Steg public S5 questions are not matched.")
        length = len(control_private)
        if length == 1 and control_public != piggy_public:
            raise RuntimeError(
                "At length one, Piggy and control public questions must match exactly."
            )
        row_count[length] += 1
        pairs[length].update(
            {
                _pair_key(initial, control_public, control_private),
                _pair_key(initial, piggy_public, piggy_private),
            }
        )
        if length >= 2:
            before = len(components[length])
            components[length].update(
                {
                    _component_key(initial, control_private),
                    _component_key(initial, piggy_public),
                    _component_key(initial, control_public),
                }
            )
            if len(components[length]) != before + 3:
                raise RuntimeError(
                    f"Repeated complete S5 component at length {length}."
                )
    expected_counts = Counter({length: expected_per_length for length in range(1, 20)})
    if row_count != expected_counts:
        raise RuntimeError(f"S5 row distribution is not balanced: {row_count}.")
    for variant in VARIANTS:
        if length_counts[variant] != expected_counts:
            raise RuntimeError(f"S5 {variant} length distribution is not balanced.")
    for length in range(1, 20):
        expected_pairs = expected_per_length if length == 1 else 2 * expected_per_length
        if len(pairs[length]) != expected_pairs:
            raise RuntimeError(f"Unexpected pair-key count at length {length}.")
        if length >= 2 and len(components[length]) != 3 * expected_per_length:
            raise RuntimeError(f"Unexpected component-key count at length {length}.")
    return components, pairs


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest_path = data_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing split manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "experiment": config["experiment_name"],
        "lengths": config["lengths"],
        "seeds": config["seeds"],
        "piggyback_shared_rule": "all_but_final_swap",
        "private_questions_matched_across_all_five_conditions": True,
        "control_public_matched_across_vanilla_filler_public_only_and_steg": True,
        "length_one_piggyback_control_public_matched": True,
        "evaluation_set_shared_across_training_seeds": True,
        "train_eval_overlap": 0,
        "cross_seed_train_overlap": 0,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("Global S5 split contract is missing or stale.")

    eval_n = int(config["eval_examples_per_length"])
    train_n = int(config["train_examples_per_length"])
    lengths = [int(value) for value in config["lengths"]]
    eval_total = eval_n * len(lengths)
    eval_records = {
        variant: _records(data_root / "shared_eval" / variant, eval_total)
        for variant in VARIANTS
    }
    eval_components, eval_pairs = _validate_matched_split(eval_records, eval_n)
    for length in lengths:
        for variant in VARIANTS:
            view = _records(
                data_root / "shared_eval" / f"length_{length}" / variant, eval_n
            )
            _validate_ids = {record["experiment_id"] for record in view}
            expected_ids = {
                record["experiment_id"]
                for record in eval_records[variant]
                if int(record["spec"]["evaluation_length"]) == length
            }
            if _validate_ids != expected_ids:
                raise RuntimeError(
                    f"S5 length view mismatch for {variant}/length_{length}."
                )
            for record in view:
                _check_record(record, variant, length)
        if (
            _digest(eval_pairs[length])
            != manifest["evaluation_pair_digest_by_length"][str(length)]
        ):
            raise RuntimeError(f"Evaluation digest mismatch at length {length}.")

    seeds = [seed] if seed is not None else [int(value) for value in config["seeds"]]
    prior_components: dict[int, set[Any]] = defaultdict(set)
    prior_pairs: dict[int, set[Any]] = defaultdict(set)
    reports: dict[str, Any] = {}
    for current_seed in seeds:
        train_records = {
            variant: _records(
                data_root / f"seed_{current_seed}" / variant, train_n * len(lengths)
            )
            for variant in VARIANTS
        }
        train_components, train_pairs = _validate_matched_split(train_records, train_n)
        for length in lengths:
            leakage_train = (
                train_pairs[length] if length == 1 else train_components[length]
            )
            leakage_eval = (
                eval_pairs[length] if length == 1 else eval_components[length]
            )
            leakage_prior = (
                prior_pairs[length] if length == 1 else prior_components[length]
            )
            if leakage_train & leakage_eval:
                raise RuntimeError(
                    f"S5 train/evaluation overlap at seed {current_seed}, length {length}."
                )
            if leakage_train & leakage_prior:
                raise RuntimeError(
                    f"S5 cross-seed train overlap at seed {current_seed}, length {length}."
                )
            if length == 1:
                prior_pairs[length].update(train_pairs[length])
            else:
                prior_components[length].update(train_components[length])
        reports[str(current_seed)] = {
            "train_rows_per_variant": train_n * len(lengths),
            "train_examples_per_length": train_n,
            "train_eval_overlap": 0,
        }
    result = {
        "valid": True,
        "validated_seeds": seeds,
        "lengths": lengths,
        "private_questions_matched_across_all_five_conditions": True,
        "control_steg_public_questions_matched": True,
        "piggyback_shared_rule": "all_but_final_swap",
        "piggyback_shared_prefix_by_length": {
            str(length): length - 1 for length in lengths
        },
        "length_one_piggyback_control_public_matched": True,
        "train_eval_overlap": 0,
        "cross_seed_train_overlap": 0,
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
    arguments = parser.parse_args()
    loaded = load_config(arguments.config)
    validate(
        arguments.config,
        arguments.data_root or Path(loaded["paths"]["data_root"]),
        arguments.seed,
    )
