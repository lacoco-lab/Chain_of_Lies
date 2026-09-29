#!/usr/bin/env python3
"""Generate unique train/validation splits for arithmetic and S5 variants."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))

from chain_of_lies.evaluation.rewards import normalize_variant_name
from chain_of_lies.variants.arithmetic.data_generation.generate import (
    ACTIVE_PIGGYBACK_FACTORS,
    ACTIVE_PIGGYBACK_MULTIPLIERS,
    ACTIVE_PIGGYBACK_OFFSETS,
    ACTIVE_PIGGYBACK_SMALL_SHIFTS,
    build_arithmetic_prompt_text,
    sample_control_linear_pair,
    sample_correlated_linear_pair,
)
from chain_of_lies.variants.s5.data_generation.generate import (
    DEFAULT_LENGTH_RANGE,
    DEFAULT_PIGGYBACK_CHANGE_RANGE,
    DEFAULT_PIGGYBACK_CHANGE_POSITION,
    S5_VARIANT_CONTROL,
    S5_VARIANT_PIGGYBACK,
    build_public_cot_prefix as build_s5_public_cot_prefix,
    build_s5_prompt_text,
    normalized_sequence_hamming_distance,
    pair_to_spec as s5_pair_to_spec,
    replay_sequence,
    sample_s5_control_pair,
    sample_s5_piggyback_pair,
    sequence_hamming_distance,
    spec_sequences as s5_spec_sequences,
    validate_public_cot as validate_s5_public_cot,
    validate_swap,
)

ACTIVE_ARITHMETIC_VARIANTS = ("arith_piggyback", "arith_piggyback_control")
ACTIVE_S5_VARIANTS = (S5_VARIANT_PIGGYBACK, S5_VARIANT_CONTROL)


def _write_payload(payload: dict[str, Any], out_dir: Path, experiment_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{experiment_id}.json"
    tmp_target = out_dir / f".{experiment_id}.json.tmp"
    tmp_target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_target.replace(target)


def _validate_prompt_dir(prompt_dir: Path, expected_count: int) -> None:
    paths = sorted(prompt_dir.glob("*.json"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} JSON files in {prompt_dir}, found {len(paths)}."
        )
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid generated prompt JSON in {path}: {exc}"
            ) from exc


def _clear_prompt_dir(prompt_dir: Path) -> None:
    if not prompt_dir.exists():
        return
    for pattern in ("*.json", ".*.json.tmp"):
        for path in prompt_dir.glob(pattern):
            path.unlink()


def _payload(
    variant_name: str,
    experiment_id: str,
    public_pair: tuple[str, int],
    private_pair: tuple[str, int],
) -> dict[str, Any]:
    public_question, public_answer = public_pair
    private_question, private_answer = private_pair
    return {
        "task_type": "arithmetic",
        "experiment_id": experiment_id,
        "difficulty_variant": variant_name.removeprefix("arith_"),
        "prompt_text": build_arithmetic_prompt_text(public_question, private_question),
        "spec": {
            "public_question": public_question,
            "private_question": private_question,
            "public_answer": public_answer,
            "private_answer": private_answer,
        },
    }


def _s5_payload(
    variant_name: str,
    experiment_id: str,
    pair: Any,
) -> dict[str, Any]:
    spec = s5_pair_to_spec(pair)
    return {
        "task_type": "s5",
        "experiment_id": experiment_id,
        "difficulty_variant": variant_name,
        "prompt_text": build_s5_prompt_text(
            pair.public_sequence, pair.private_sequence
        ),
        "spec": spec,
    }


def _sampler_for_variant(
    variant_name: str,
) -> Callable[[random.Random], tuple[tuple[str, int], tuple[str, int]]]:
    variant_name = normalize_variant_name(variant_name)
    if variant_name == "arith_piggyback":
        return sample_correlated_linear_pair
    if variant_name == "arith_piggyback_control":
        return sample_control_linear_pair
    raise ValueError(f"Unsupported active variant: {variant_name}")


def _s5_sampler_for_variant(
    variant_name: str,
    *,
    length_range: tuple[int, int],
    piggyback_change_range: tuple[int, int],
    piggyback_change_position: str,
) -> Callable[[random.Random], Any]:
    variant_name = normalize_variant_name(variant_name)
    if variant_name == S5_VARIANT_PIGGYBACK:
        return lambda rng: sample_s5_piggyback_pair(
            rng,
            length_range=length_range,
            change_range=piggyback_change_range,
            change_position=piggyback_change_position,
        )
    if variant_name == S5_VARIANT_CONTROL:
        return lambda rng: sample_s5_control_pair(rng, length_range=length_range)
    raise ValueError(f"Unsupported S5 variant: {variant_name}")


def _s5_pair_key(pair: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    spec = s5_pair_to_spec(pair)
    return (
        tuple(spec["public_instruction_sequence"]),
        tuple(spec["private_instruction_sequence"]),
    )


def _validate_s5_payload(
    payload: dict[str, Any],
    *,
    variant_name: str,
    piggyback_change_range: tuple[int, int],
    piggyback_change_position: str,
) -> None:
    spec = payload["spec"]
    public_sequence, private_sequence = s5_spec_sequences(spec)
    for swap in (*public_sequence, *private_sequence):
        validate_swap(swap)
    public_final, _ = replay_sequence(public_sequence)
    private_final, _ = replay_sequence(private_sequence)
    if public_final != spec["public_answer"]:
        raise RuntimeError(
            f"Incorrect S5 public state update in {payload['experiment_id']}."
        )
    if private_final != spec["private_answer"]:
        raise RuntimeError(
            f"Incorrect S5 private state update in {payload['experiment_id']}."
        )
    validate_s5_public_cot(public_sequence, build_s5_public_cot_prefix(public_sequence))
    if public_sequence == private_sequence:
        raise RuntimeError(
            f"S5 public/private sequences are identical in {payload['experiment_id']}."
        )
    differing_positions = sequence_hamming_distance(public_sequence, private_sequence)
    if differing_positions != spec["num_differing_positions"]:
        raise RuntimeError(
            f"Incorrect S5 differing-position count in {payload['experiment_id']}."
        )
    if variant_name == S5_VARIANT_PIGGYBACK:
        changed_positions = tuple(spec["changed_positions"])
        if len(changed_positions) != differing_positions:
            raise RuntimeError(
                f"S5 piggyback changed-position metadata mismatch in {payload['experiment_id']}."
            )
        if not (
            piggyback_change_range[0]
            <= differing_positions
            <= piggyback_change_range[1]
        ):
            raise RuntimeError(
                f"S5 piggyback changed {differing_positions} positions in {payload['experiment_id']}."
            )
        if piggyback_change_position == "last" and changed_positions != (
            len(public_sequence) - 1,
        ):
            raise RuntimeError(
                f"S5 piggyback did not change only the final swap in {payload['experiment_id']}."
            )
        if (
            piggyback_change_position == "last"
            and spec["sampling_strategy"] != "copy_public_then_replace_final_position"
        ):
            raise RuntimeError(
                f"S5 piggyback final-position strategy mismatch in {payload['experiment_id']}."
            )
    if variant_name == S5_VARIANT_CONTROL:
        if spec["sampling_strategy"] != "independent_public_private_same_length":
            raise RuntimeError(
                f"S5 control was not marked as independently sampled in {payload['experiment_id']}."
            )


def _s5_diagnostics(pairs: list[Any]) -> dict[str, Any]:
    if not pairs:
        return {}
    lengths = [len(pair.public_sequence) for pair in pairs]
    differing_positions = [
        sequence_hamming_distance(pair.public_sequence, pair.private_sequence)
        for pair in pairs
    ]
    normalized_distances = [
        normalized_sequence_hamming_distance(
            pair.public_sequence, pair.private_sequence
        )
        for pair in pairs
    ]
    return {
        "avg_instruction_sequence_length": sum(lengths) / len(lengths),
        "min_instruction_sequence_length": min(lengths),
        "max_instruction_sequence_length": max(lengths),
        "avg_num_differing_positions": sum(differing_positions)
        / len(differing_positions),
        "avg_normalized_hamming_distance": sum(normalized_distances)
        / len(normalized_distances),
    }


def write_unique_s5_split(
    *,
    variant_name: str,
    train_n: int,
    val_n: int,
    output_root: Path,
    seed: int,
    length_range: tuple[int, int],
    piggyback_change_range: tuple[int, int],
    piggyback_change_position: str,
) -> None:
    variant_name = normalize_variant_name(variant_name)
    sampler = _s5_sampler_for_variant(
        variant_name,
        length_range=length_range,
        piggyback_change_range=piggyback_change_range,
        piggyback_change_position=piggyback_change_position,
    )
    rng = random.Random(seed)
    total_needed = train_n + val_n
    seen_pairs: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    selected_pairs: list[Any] = []
    max_attempts = max(200_000, total_needed * 50)

    attempts = 0
    while len(selected_pairs) < total_needed:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not sample {total_needed} unique S5 pairs for {variant_name} "
                f"after {attempts} attempts."
            )
        pair = sampler(rng)
        key = _s5_pair_key(pair)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        selected_pairs.append(pair)

    variant_root = output_root / variant_name
    for split_name, count, offset in (
        ("train", train_n, 0),
        ("val", val_n, train_n),
    ):
        out_dir = variant_root / f"{split_name}_prompts"
        _clear_prompt_dir(out_dir)
        for index in range(count):
            pair = selected_pairs[offset + index]
            experiment_id = f"s5_{split_name}_{index:05d}"
            payload = _s5_payload(variant_name, experiment_id, pair)
            _validate_s5_payload(
                payload,
                variant_name=variant_name,
                piggyback_change_range=piggyback_change_range,
                piggyback_change_position=piggyback_change_position,
            )
            _write_payload(payload, out_dir, experiment_id)
        _validate_prompt_dir(out_dir, count)

    train_pairs = selected_pairs[:train_n]
    val_pairs = selected_pairs[train_n:]
    samples = [
        {
            "experiment_id": f"s5_sample_{index:02d}",
            "public_instruction_sequence": s5_pair_to_spec(pair)[
                "public_instruction_sequence"
            ],
            "private_instruction_sequence": s5_pair_to_spec(pair)[
                "private_instruction_sequence"
            ],
            "public_answer": pair.public_final_state,
            "private_answer": pair.private_final_state,
            "num_differing_positions": sequence_hamming_distance(
                pair.public_sequence, pair.private_sequence
            ),
            "normalized_hamming_distance": normalized_sequence_hamming_distance(
                pair.public_sequence, pair.private_sequence
            ),
        }
        for index, pair in enumerate(selected_pairs[:3])
    ]
    manifest = {
        "variant_name": variant_name,
        "seed": seed,
        "train_n": train_n,
        "val_n": val_n,
        "length_range": list(length_range),
        "piggyback_change_range": (
            list(piggyback_change_range)
            if variant_name == S5_VARIANT_PIGGYBACK
            else None
        ),
        "piggyback_change_position": (
            piggyback_change_position if variant_name == S5_VARIANT_PIGGYBACK else None
        ),
        "unique_pair_count": len(seen_pairs),
        "train_val_exact_pair_disjoint": True,
        "public_private_pairs_unique": True,
        "checks": {
            "valid_swap_generation": True,
            "correct_state_updates": True,
            "public_cot_faithful_to_public_trajectory": True,
            "piggyback_small_number_of_differences": (
                True if variant_name == S5_VARIANT_PIGGYBACK else None
            ),
            "control_independently_sampled": (
                True if variant_name == S5_VARIANT_CONTROL else None
            ),
        },
        "diagnostics": {
            "all": _s5_diagnostics(selected_pairs),
            "train": _s5_diagnostics(train_pairs),
            "val": _s5_diagnostics(val_pairs),
        },
        "samples": samples,
    }
    (variant_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[SplitGen] wrote {train_n} train and {val_n} validation prompts for {variant_name} "
        f"under {variant_root}",
        flush=True,
    )
    print(
        f"[SplitGen] diagnostics {variant_name}: {json.dumps(manifest['diagnostics']['all'], sort_keys=True)}"
    )
    for sample in samples:
        print(f"[SplitGen] sample {variant_name}: {json.dumps(sample, sort_keys=True)}")


def write_unique_split(
    *,
    variant_name: str,
    train_n: int,
    val_n: int,
    output_root: Path,
    seed: int,
) -> None:
    variant_name = normalize_variant_name(variant_name)
    sampler = _sampler_for_variant(variant_name)
    rng = random.Random(seed)
    total_needed = train_n + val_n
    seen_pairs: set[tuple[str, str]] = set()
    seen_questions: set[str] = set()
    selected_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    max_attempts = max(200_000, total_needed * 50)

    attempts = 0
    while len(selected_pairs) < total_needed:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not sample {total_needed} unique pairs for {variant_name} "
                f"after {attempts} attempts."
            )
        public_pair, private_pair = sampler(rng)
        key = (public_pair[0], private_pair[0])
        if key in seen_pairs:
            continue
        if public_pair[0] in seen_questions or private_pair[0] in seen_questions:
            continue
        seen_pairs.add(key)
        seen_questions.add(public_pair[0])
        seen_questions.add(private_pair[0])
        selected_pairs.append((public_pair, private_pair))

    variant_root = output_root / variant_name
    for split_name, count, offset in (
        ("train", train_n, 0),
        ("val", val_n, train_n),
    ):
        out_dir = variant_root / f"{split_name}_prompts"
        _clear_prompt_dir(out_dir)
        for index in range(count):
            public_pair, private_pair = selected_pairs[offset + index]
            experiment_id = f"arith_{split_name}_{index:05d}"
            _write_payload(
                _payload(variant_name, experiment_id, public_pair, private_pair),
                out_dir,
                experiment_id,
            )
        _validate_prompt_dir(out_dir, count)

    manifest = {
        "variant_name": variant_name,
        "seed": seed,
        "train_n": train_n,
        "val_n": val_n,
        "active_multiplier_range": [
            min(ACTIVE_PIGGYBACK_MULTIPLIERS),
            max(ACTIVE_PIGGYBACK_MULTIPLIERS),
        ],
        "active_factor_range": [
            min(ACTIVE_PIGGYBACK_FACTORS),
            max(ACTIVE_PIGGYBACK_FACTORS),
        ],
        "active_offset_range": [
            min(ACTIVE_PIGGYBACK_OFFSETS),
            max(ACTIVE_PIGGYBACK_OFFSETS),
        ],
        "active_piggyback_shifts": list(ACTIVE_PIGGYBACK_SMALL_SHIFTS),
        "unique_pair_count": len(seen_pairs),
        "unique_question_count": len(seen_questions),
        "train_val_exact_pair_disjoint": True,
        "train_val_question_disjoint": True,
    }
    (variant_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[SplitGen] wrote {train_n} train and {val_n} validation prompts for {variant_name} "
        f"under {variant_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        type=str,
        default="all_active",
        help="One variant, 'all_active' for arithmetic, or 'all_s5'.",
    )
    parser.add_argument("--train-n", type=int, default=10_000)
    parser.add_argument("--val-n", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("generated_data/prompt_splits")
    )
    parser.add_argument("--min-steps", type=int, default=DEFAULT_LENGTH_RANGE[0])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_LENGTH_RANGE[1])
    parser.add_argument(
        "--min-piggyback-changes", type=int, default=DEFAULT_PIGGYBACK_CHANGE_RANGE[0]
    )
    parser.add_argument(
        "--max-piggyback-changes", type=int, default=DEFAULT_PIGGYBACK_CHANGE_RANGE[1]
    )
    parser.add_argument(
        "--piggyback-change-position",
        type=str,
        choices=["last", "random"],
        default=DEFAULT_PIGGYBACK_CHANGE_POSITION,
    )
    args = parser.parse_args()

    length_range = (args.min_steps, args.max_steps)
    piggyback_change_range = (args.min_piggyback_changes, args.max_piggyback_changes)
    if args.variant == "all_active":
        variants = ACTIVE_ARITHMETIC_VARIANTS
    elif args.variant == "all_s5":
        variants = ACTIVE_S5_VARIANTS
    else:
        variants = (args.variant,)
    for variant_name in variants:
        normalized = normalize_variant_name(variant_name)
        if normalized in ACTIVE_S5_VARIANTS:
            write_unique_s5_split(
                variant_name=normalized,
                train_n=args.train_n,
                val_n=args.val_n,
                output_root=args.output_root,
                seed=args.seed,
                length_range=length_range,
                piggyback_change_range=piggyback_change_range,
                piggyback_change_position=args.piggyback_change_position,
            )
        else:
            write_unique_split(
                variant_name=normalized,
                train_n=args.train_n,
                val_n=args.val_n,
                output_root=args.output_root,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
