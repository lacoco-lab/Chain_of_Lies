#!/usr/bin/env python3
"""Generate unique train/validation splits for the active arithmetic pair."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl.rewards import normalize_variant_name
from chain_of_lies.stage1_arithmetic.generate import (
    ACTIVE_PIGGYBACK_FACTORS,
    ACTIVE_PIGGYBACK_MULTIPLIERS,
    ACTIVE_PIGGYBACK_OFFSETS,
    ACTIVE_PIGGYBACK_SMALL_SHIFTS,
    build_arithmetic_prompt_text,
    sample_control_linear_pair,
    sample_correlated_linear_pair,
)


ACTIVE_VARIANTS = ("arith_piggyback", "arith_piggyback_control")


def _write_payload(payload: dict[str, Any], out_dir: Path, experiment_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{experiment_id}.json"
    tmp_target = out_dir / f".{experiment_id}.json.tmp"
    tmp_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_target.replace(target)


def _validate_prompt_dir(prompt_dir: Path, expected_count: int) -> None:
    paths = sorted(prompt_dir.glob("*.json"))
    if len(paths) != expected_count:
        raise RuntimeError(f"Expected {expected_count} JSON files in {prompt_dir}, found {len(paths)}.")
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid generated prompt JSON in {path}: {exc}") from exc


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


def _sampler_for_variant(
    variant_name: str,
) -> Callable[[random.Random], tuple[tuple[str, int], tuple[str, int]]]:
    variant_name = normalize_variant_name(variant_name)
    if variant_name == "arith_piggyback":
        return sample_correlated_linear_pair
    if variant_name == "arith_piggyback_control":
        return sample_control_linear_pair
    raise ValueError(f"Unsupported active variant: {variant_name}")


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
        "active_multiplier_range": [min(ACTIVE_PIGGYBACK_MULTIPLIERS), max(ACTIVE_PIGGYBACK_MULTIPLIERS)],
        "active_factor_range": [min(ACTIVE_PIGGYBACK_FACTORS), max(ACTIVE_PIGGYBACK_FACTORS)],
        "active_offset_range": [min(ACTIVE_PIGGYBACK_OFFSETS), max(ACTIVE_PIGGYBACK_OFFSETS)],
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
        help="One active variant or 'all_active'.",
    )
    parser.add_argument("--train-n", type=int, default=10_000)
    parser.add_argument("--val-n", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("data/RL_splits"))
    args = parser.parse_args()

    variants = ACTIVE_VARIANTS if args.variant == "all_active" else (args.variant,)
    for variant_name in variants:
        write_unique_split(
            variant_name=variant_name,
            train_n=args.train_n,
            val_n=args.val_n,
            output_root=args.output_root,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
