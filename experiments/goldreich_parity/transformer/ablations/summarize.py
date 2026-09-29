#!/usr/bin/env python3
"""Compare standard-architecture supervision ablations with full baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(label: str, length: int, path: Path) -> None:
    if not path.exists():
        print(f"| {label} | {length} | MISSING | -- | -- | -- |")
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    test = result["test_softmax_attention"]
    print(
        f"| {label} | {length} | {100*test['state_bit_accuracy']:.2f} | "
        f"{100*test['trace_exact']:.2f} | {100*test['final_parity_accuracy']:.2f} | "
        f"{100*test['joint_exact']:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/parity_goldreich_standard_ablations"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/parity_goldreich_standard_transformer_v2"),
    )
    args = parser.parse_args()
    print(
        "| Training supervision | N | State bits (%) | Entire CoT (%) | Final PARITY (%) | Joint (%) |"
    )
    print("| :-- | --: | --: | --: | --: | --: |")
    for length in (32, 64):
        row("Full", length, args.baseline_root / "d128" / f"n{length}" / "metrics.json")
        row(
            "No route labels",
            length,
            args.root / "no_route" / f"n{length}" / "metrics.json",
        )
        row(
            "End-to-end only",
            length,
            args.root / "end_to_end" / f"n{length}" / "metrics.json",
        )


if __name__ == "__main__":
    main()
