#!/usr/bin/env python3
"""Fail fast if an S5/knowledge invisible target does not survive tokenization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))

from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix
from chain_of_lies.variants.steganography.hard_task_channels import (
    build_local_channel_cot_prefix,
    decode_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--max-target-tokens", type=int, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths = []
    checked = 0
    for path in sorted(args.prompts_dir.glob("*.json"))[: args.sample_size]:
        record = json.loads(path.read_text(encoding="utf-8"))
        cot = build_local_channel_cot_prefix(record)
        suffix = _canonical_public_cot_suffix(
            record, supervision_mode="local_channel_cot"
        )
        if (
            cot is None
            or suffix is None
            or decode_payload(record, cot) != record["spec"]["steganography_payload"]
        ):
            raise RuntimeError(f"Invalid local-channel target in {path}.")
        ids = tokenizer.encode(suffix, add_special_tokens=False)
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != suffix:
            raise RuntimeError(
                f"Invisible target failed tokenizer round-trip in {path}."
            )
        lengths.append(len(ids))
        checked += 1
    if not checked:
        raise RuntimeError(f"No prompt files found in {args.prompts_dir}.")
    if max(lengths) > args.max_target_tokens:
        raise RuntimeError(
            f"Target length {max(lengths)} exceeds limit {args.max_target_tokens}."
        )
    print(
        json.dumps(
            {
                "model": args.model,
                "checked": checked,
                "round_trip_exact": True,
                "target_tokens_min": min(lengths),
                "target_tokens_max": max(lengths),
                "configured_max_target_tokens": args.max_target_tokens,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
