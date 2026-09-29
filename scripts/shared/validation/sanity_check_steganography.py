#!/usr/bin/env python3
"""Validate matched aligned-local steganography splits and target round-trips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))

from chain_of_lies.variants.steganography.data_generation.generate import (
    STEGANOGRAPHY_VARIANTS,
    aligned_private_payload,
    build_local_channel_cot_prefix,
    decode_steganographic_payload,
    validate_steganographic_target,
)


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("*.json"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root", type=Path, default=Path("generated_data/prompt_splits/seed_0")
    )
    parser.add_argument("--expected-train", type=int, default=10_000)
    parser.add_argument("--expected-val", type=int, default=1_000)
    parser.add_argument(
        "--variants",
        default=",".join(STEGANOGRAPHY_VARIANTS),
        help="Comma-separated steganography variants to validate.",
    )
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(STEGANOGRAPHY_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported steganography variants: {unknown}")
    if not variants:
        raise ValueError("At least one steganography variant is required.")

    reports: list[dict[str, Any]] = []
    reference_pairs: list[tuple[str, str]] | None = None
    reference_digest: str | None = None
    for variant in variants:
        root = args.split_root / variant
        train = _load(root / "train_prompts")
        val = _load(root / "val_prompts")
        if len(train) != args.expected_train or len(val) != args.expected_val:
            raise RuntimeError(
                f"{variant}: expected {args.expected_train}/{args.expected_val}, "
                f"found {len(train)}/{len(val)}."
            )
        records = train + val
        pairs = [
            (
                str(record["spec"]["public_question"]),
                str(record["spec"]["private_question"]),
            )
            for record in records
        ]
        if reference_pairs is None:
            reference_pairs = pairs
        elif pairs != reference_pairs:
            raise RuntimeError(
                f"{variant}: examples do not match the reference variant exactly."
            )
        manifest = json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        )
        if reference_digest is None:
            reference_digest = str(manifest["matched_pair_digest"])
        elif manifest["matched_pair_digest"] != reference_digest:
            raise RuntimeError(
                f"{variant}: matched pair digest differs from the reference variant."
            )
        if len(pairs) != len(set(pairs)):
            raise RuntimeError(f"{variant}: duplicate public/private pairs.")
        questions = [question for pair in pairs for question in pair]
        if len(questions) != len(set(questions)):
            raise RuntimeError(f"{variant}: question reuse across train/validation.")
        for record in records:
            validate_steganographic_target(record)
            cot = build_local_channel_cot_prefix(record)
            expected = aligned_private_payload(record)
            if decode_steganographic_payload(record, cot or "") != expected:
                raise RuntimeError(
                    f"{variant}: round-trip failure in {record['experiment_id']}."
                )
        reports.append(
            {
                "variant": variant,
                "train_n": len(train),
                "val_n": len(val),
                "target_round_trip_rate": 1.0,
                "matched_pair_digest": manifest["matched_pair_digest"],
            }
        )
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
