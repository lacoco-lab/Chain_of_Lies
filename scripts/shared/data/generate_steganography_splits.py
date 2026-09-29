#!/usr/bin/env python3
"""Generate matched splits for the aligned local steganography variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))

from chain_of_lies.variants.arithmetic.data_generation.generate import (
    ACTIVE_PIGGYBACK_FACTORS,
    ACTIVE_PIGGYBACK_MULTIPLIERS,
)
from chain_of_lies.variants.steganography.data_generation.generate import (
    STEGANOGRAPHY_VARIANTS,
    aligned_private_payload,
    build_steganography_prompt_text,
    sample_no_offset_steganography_pair,
    steganography_scheme_for_variant,
    validate_steganographic_target,
)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.replace(path)


def _clear_prompt_dir(prompt_dir: Path) -> None:
    if not prompt_dir.exists():
        return
    for pattern in ("*.json", ".*.json.tmp"):
        for path in prompt_dir.glob(pattern):
            path.unlink()


def _sample_pairs(
    *, train_n: int, val_n: int, seed: int
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    rng = random.Random(seed)
    total_needed = train_n + val_n
    seen_pairs: set[tuple[str, str]] = set()
    seen_questions: set[str] = set()
    selected: list[tuple[tuple[str, int], tuple[str, int]]] = []
    max_attempts = max(200_000, total_needed * 50)
    attempts = 0
    while len(selected) < total_needed:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not sample {total_needed} unique multiplication pairs."
            )
        public_pair, private_pair = sample_no_offset_steganography_pair(rng)
        key = (public_pair[0], private_pair[0])
        if key in seen_pairs:
            continue
        if public_pair[0] in seen_questions or private_pair[0] in seen_questions:
            continue
        seen_pairs.add(key)
        seen_questions.update(key)
        selected.append((public_pair, private_pair))
    return selected


def _record(
    *,
    variant_name: str,
    experiment_id: str,
    public_pair: tuple[str, int],
    private_pair: tuple[str, int],
) -> dict[str, Any]:
    public_question, public_answer = public_pair
    private_question, private_answer = private_pair
    scheme = steganography_scheme_for_variant(variant_name)
    record: dict[str, Any] = {
        "task_type": "arithmetic",
        "experiment_id": experiment_id,
        "difficulty_variant": variant_name.removeprefix("arith_"),
        "prompt_text": build_steganography_prompt_text(
            public_question,
            private_question,
            scheme=scheme,
        ),
        "spec": {
            "public_question": public_question,
            "private_question": private_question,
            "public_answer": public_answer,
            "private_answer": private_answer,
            "steganography_scheme": scheme,
            "steganography_payload_source": "aligned_private_trace_values",
        },
    }
    payload = aligned_private_payload(record)
    record["spec"]["steganography_payload"] = payload
    record["spec"]["steganography_payload_value_count"] = len(payload.split("|"))
    return record


def _pair_digest(pairs: list[tuple[tuple[str, int], tuple[str, int]]]) -> str:
    serialized = "\n".join(f"{public[0]}\t{private[0]}" for public, private in pairs)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_matched_splits(
    *,
    train_n: int,
    val_n: int,
    output_root: Path,
    seed: int,
    variants: tuple[str, ...] = STEGANOGRAPHY_VARIANTS,
) -> None:
    pairs = _sample_pairs(train_n=train_n, val_n=val_n, seed=seed)
    digest = _pair_digest(pairs)
    unknown = sorted(set(variants) - set(STEGANOGRAPHY_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported steganography variants: {unknown}")
    if not variants:
        raise ValueError("At least one steganography variant is required.")
    for variant_name in variants:
        variant_root = output_root / variant_name
        records: list[dict[str, Any]] = []
        for split_name, count, offset in (
            ("train", train_n, 0),
            ("val", val_n, train_n),
        ):
            out_dir = variant_root / f"{split_name}_prompts"
            _clear_prompt_dir(out_dir)
            for index in range(count):
                public_pair, private_pair = pairs[offset + index]
                experiment_id = f"steg_{split_name}_{index:05d}"
                record = _record(
                    variant_name=variant_name,
                    experiment_id=experiment_id,
                    public_pair=public_pair,
                    private_pair=private_pair,
                )
                validate_steganographic_target(record)
                _write_json(record, out_dir / f"{experiment_id}.json")
                records.append(record)
            written = list(out_dir.glob("*.json"))
            if len(written) != count:
                raise RuntimeError(
                    f"Expected {count} prompts in {out_dir}, found {len(written)}."
                )

        widths = [
            int(record["spec"]["steganography_payload_value_count"])
            for record in records
        ]
        manifest = {
            "variant_name": variant_name,
            "scheme": steganography_scheme_for_variant(variant_name),
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
            "active_offset_range": [0, 0],
            "public_private_sampled_independently": True,
            "train_val_pair_and_question_disjoint": True,
            "matched_pair_digest": digest,
            "payload_source": "aligned_private_trace_values",
            "payload_value_count": {
                "min": min(widths),
                "max": max(widths),
                "mean": sum(widths) / len(widths),
            },
            "checks": {
                "public_answers_correct": True,
                "private_answers_correct": True,
                "local_payload_matches_private_trace": True,
                "target_round_trip_exact": True,
            },
            "sample": {
                "experiment_id": records[0]["experiment_id"],
                "public_question": records[0]["spec"]["public_question"],
                "private_question": records[0]["spec"]["private_question"],
                "payload": records[0]["spec"]["steganography_payload"],
            },
        }
        _write_json(manifest, variant_root / "split_manifest.json")
        print(
            f"[LocalStegSplit] {variant_name}: train={train_n} val={val_n} "
            f"matched_pair_digest={digest}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-n", type=int, default=10_000)
    parser.add_argument("--val-n", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("generated_data/prompt_splits/seed_0")
    )
    parser.add_argument(
        "--variants",
        default=",".join(STEGANOGRAPHY_VARIANTS),
        help="Comma-separated steganography variants to generate.",
    )
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    write_matched_splits(
        train_n=args.train_n,
        val_n=args.val_n,
        output_root=args.output_root,
        seed=args.seed,
        variants=variants,
    )


if __name__ == "__main__":
    main()
