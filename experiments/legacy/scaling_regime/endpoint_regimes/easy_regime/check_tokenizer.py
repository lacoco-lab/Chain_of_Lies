#!/usr/bin/env python3
"""Check every Easy steganographic target survives a tokenizer round-trip."""

from __future__ import annotations
import argparse, json, sys
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts-dir", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--max-target-tokens", type=int, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    lengths = []
    for path in sorted(a.prompts_dir.glob("*.json"))[: a.sample_size]:
        record = json.loads(path.read_text(encoding="utf-8"))
        suffix = _canonical_public_cot_suffix(
            record, supervision_mode="local_channel_cot"
        )
        if suffix is None:
            raise RuntimeError(f"Missing local-channel target: {path}")
        ids = tokenizer.encode(suffix, add_special_tokens=False)
        if (
            tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            != suffix
        ):
            raise RuntimeError(f"Tokenizer round-trip failed: {path}")
        lengths.append(len(ids))
    if not lengths or max(lengths) > a.max_target_tokens:
        raise RuntimeError(
            f"Invalid target count/length: n={len(lengths)}, max={max(lengths) if lengths else None}"
        )
    print(
        json.dumps(
            {
                "checked": len(lengths),
                "target_tokens_min": min(lengths),
                "target_tokens_max": max(lengths),
                "round_trip_exact": True,
            },
            indent=2,
        )
    )
