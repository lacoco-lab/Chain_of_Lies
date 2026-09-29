#!/usr/bin/env python3
"""Print the paper-ready table for the matched PARITY controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/parity_standard_cot_controls")
    )
    args = parser.parse_args()
    print(
        "| Condition | N | CoT token accuracy (%) | Entire CoT (%) | Final PARITY (%) | Expected pattern |"
    )
    print("| :-- | --: | --: | --: | --: | :--: |")
    lengths = {
        "no_cot": (4, 8, 16, 32, 64),
        "filler_cot": (4, 8, 16, 32, 64),
        "normal_cot": (4, 8, 16, 32, 64, 256, 512, 1024),
    }
    for condition, condition_lengths in lengths.items():
        for length in condition_lengths:
            path = args.root / condition / f"n{length}" / "metrics.json"
            if not path.exists():
                print(f"| {condition} | {length:,} | MISSING | -- | -- | -- |")
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            test = result["test_softmax_attention"]
            print(
                f"| {condition} | {length:,} | {fmt(test['cot_bit_accuracy'])} | "
                f"{fmt(test['cot_exact'])} | {fmt(test['final_parity_accuracy'])} | "
                f"{'Yes' if result['matches_expected_pattern'] else 'No'} |"
            )


if __name__ == "__main__":
    main()
