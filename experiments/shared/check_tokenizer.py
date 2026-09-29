#!/usr/bin/env python3
"""Check that local-channel targets survive an exact tokenizer round trip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        )
    ),
)

from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-target-tokens", type=int, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths: list[int] = []
    for path in sorted(args.prompts_dir.glob("*.json"))[: args.sample_size]:
        record = json.loads(path.read_text(encoding="utf-8"))
        suffix = _canonical_public_cot_suffix(
            record, supervision_mode="local_channel_cot"
        )
        if suffix is None:
            raise RuntimeError(f"Missing local-channel target: {path}")
        token_ids = tokenizer.encode(suffix, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != suffix:
            raise RuntimeError(f"Tokenizer round-trip failed: {path}")
        lengths.append(len(token_ids))

    maximum = max(lengths) if lengths else None
    if not lengths or maximum > args.max_target_tokens:
        raise RuntimeError(
            f"Invalid target count/length: n={len(lengths)}, max={maximum}"
        )
    print(
        json.dumps(
            {
                "checked": len(lengths),
                "target_tokens_min": min(lengths),
                "target_tokens_max": maximum,
                "round_trip_exact": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
