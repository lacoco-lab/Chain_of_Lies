#!/usr/bin/env python3
"""Summarize full-test softmax-attention evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/parity_goldreich_toy_transformer_softmax"),
    )
    args = parser.parse_args()
    rows = []
    for length in (32, 64, 256, 512, 1024, 2048):
        path = args.root / f"n{length}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["test_softmax_attention"]
        rows.append(
            (
                length,
                result["seed_bits"],
                metrics["n"],
                metrics["state_bit_accuracy"],
                metrics["trace_exact"],
                metrics["final_parity_accuracy"],
                metrics["joint_exact"],
                result["passes_original_task_gate"],
                result["elapsed_seconds"],
            )
        )
    print(
        "| N | r | Test | State bits | Entire CoT | Final PARITY | Joint | Pass | Seconds |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for row in rows:
        print(
            f"| {row[0]:,} | {row[1]} | {row[2]:,} | {row[3]:.6f} | {row[4]:.6f} "
            f"| {row[5]:.6f} | {row[6]:.6f} | {str(row[7])} | {row[8]:.1f} |"
        )


if __name__ == "__main__":
    main()
