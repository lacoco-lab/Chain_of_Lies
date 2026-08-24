#!/usr/bin/env python3
"""Fail fast if the invisible local channel does not survive tokenizer round-trips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.variants.steganography.data_generation.generate import (
    STEG_VARIANT_LOCAL_INVISIBLE,
    build_local_channel_cot_prefix,
    invisible_digit_codebook,
)
from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split-root", type=Path, default=Path("generated_data/prompt_splits/seed_0"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=None,
        help="Fail if any checked full supervised target exceeds this token budget.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sequences: dict[str, list[int]] = {}
    for digit, character in invisible_digit_codebook().items():
        token_ids = tokenizer.encode(character, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != character:
            raise RuntimeError(
                f"Invisible digit {digit} failed tokenizer round-trip: "
                f"ids={token_ids}, decoded={decoded.encode('unicode_escape')!r}."
            )
        sequences[digit] = token_ids
    if len({tuple(ids) for ids in sequences.values()}) != 10:
        raise RuntimeError("Invisible digit characters do not have ten distinct token sequences.")

    prompt_dir = args.split_root / STEG_VARIANT_LOCAL_INVISIBLE / "train_prompts"
    checked = 0
    target_token_lengths: list[int] = []
    for path in sorted(prompt_dir.glob("*.json"))[: args.sample_size]:
        record = json.loads(path.read_text(encoding="utf-8"))
        cot = build_local_channel_cot_prefix(record)
        if cot is None:
            raise RuntimeError(f"Could not build invisible target for {path}.")
        token_ids = tokenizer.encode(cot, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != cot:
            raise RuntimeError(f"Full invisible target failed tokenizer round-trip for {path.name}.")
        suffix = _canonical_public_cot_suffix(record, supervision_mode="local_channel_cot")
        if suffix is None:
            raise RuntimeError(f"Could not build the complete supervised target for {path}.")
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        suffix_decoded = tokenizer.decode(
            suffix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if suffix_decoded != suffix:
            raise RuntimeError(f"Full supervised target failed tokenizer round-trip for {path.name}.")
        target_token_lengths.append(len(suffix_ids))
        checked += 1
    if checked == 0:
        raise RuntimeError(f"No invisible-channel prompts found under {prompt_dir}.")
    max_target_tokens = max(target_token_lengths)
    if args.max_target_tokens is not None and max_target_tokens > args.max_target_tokens:
        raise RuntimeError(
            f"Checked target needs {max_target_tokens} tokens, exceeding "
            f"--max-target-tokens={args.max_target_tokens}."
        )
    print(
        json.dumps(
            {
                "model": args.model,
                "digit_token_lengths": {digit: len(ids) for digit, ids in sequences.items()},
                "full_targets_checked": checked,
                "full_target_token_length_min": min(target_token_lengths),
                "full_target_token_length_max": max_target_tokens,
                "configured_max_target_tokens": args.max_target_tokens,
                "round_trip_exact": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
