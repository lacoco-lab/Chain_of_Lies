#!/usr/bin/env python3
"""Summarize the retained standard-Transformer paper experiment.

The training code uses a deliberately strict internal gate (including 99.9%
state-bit and final-answer accuracy).  That gate is useful for stopping failed
training runs, but it is too easy to misread as the scientific outcome.  This
summary instead reports whether every test metric reaches an explicit,
user-configurable accuracy threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/parity_goldreich_standard_transformer"),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.99,
        help="Minimum required value for every reported test metric (default: 0.99).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.success_threshold <= 1.0:
        parser.error("--success-threshold must be between 0 and 1")

    threshold_label = f"All metrics >= {100 * args.success_threshold:g}%"
    print(
        "| d_model | N | r | Parameters | State bits | Entire CoT | "
        f"Final PARITY | Joint | {threshold_label} |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for model_dim in (128,):
        for length in (4, 8, 16, 32, 64, 256, 512, 1024):
            path = args.root / f"d{model_dim}" / f"n{length}" / "metrics.json"
            if not path.is_file():
                print(
                    f"| {model_dim} | {length:,} | -- | -- | MISSING | -- | -- | -- | -- |"
                )
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            metrics = result["test_softmax_attention"]
            reported_metrics = (
                metrics["state_bit_accuracy"],
                metrics["trace_exact"],
                metrics["final_parity_accuracy"],
                metrics["joint_exact"],
            )
            meets_threshold = all(
                value >= args.success_threshold for value in reported_metrics
            )
            print(
                f"| {model_dim} | {length:,} | {result['seed_bits']} | "
                f"{result['architecture']['trainable_parameters']:,} | "
                f"{metrics['state_bit_accuracy']:.6f} | {metrics['trace_exact']:.6f} | "
                f"{metrics['final_parity_accuracy']:.6f} | {metrics['joint_exact']:.6f} | "
                f"{'Yes' if meets_threshold else 'No'} |"
            )


if __name__ == "__main__":
    main()
