#!/usr/bin/env python3
"""Summarize the Goldreich supervision-ablation matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODES = ("mask_state", "cot_answer_only")
LENGTHS = (32, 64, 256)


def summarize(root: Path) -> dict:
    rows = []
    for mode in MODES:
        for length in LENGTHS:
            metrics = json.loads(
                (root / mode / f"n{length}" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            test = metrics["test_hard_attention"]
            rows.append(
                {
                    "supervision": mode,
                    "length": length,
                    "test_n": test["n"],
                    "state_bit_accuracy": test["state_bit_accuracy"],
                    "trace_exact": test["trace_exact"],
                    "final_parity_accuracy": test["final_parity_accuracy"],
                    "joint_exact": test["joint_exact"],
                    "routing_accuracy": metrics["final_routing_metrics"]["accuracy"],
                    "minimum_target_attention": metrics["final_routing_metrics"][
                        "minimum_target_attention"
                    ],
                    "training_steps": metrics["integration_stage"]["steps"],
                    "passes": metrics["passes_gate"],
                }
            )
    result = {
        "schema_version": 2,
        "experiment_name": "parity_goldreich_toy_transformer_v2_supervision_ablation",
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Goldreich supervision ablations",
        "",
        "The public predicate and XOR truth tables are supervised in both modes.",
        "",
        "- `mask_state`: mask, encrypted-state, and final-answer losses; no position labels.",
        "- `cot_answer_only`: encrypted-CoT and final-answer losses only; no mask or position labels.",
        "",
        "| Mode | N | State bits | Entire CoT | Final PARITY | CoT + answer | Routing | Result |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['supervision']} | {row['length']} | "
            f"{row['state_bit_accuracy']:.4f} | {row['trace_exact']:.4f} | "
            f"{row['final_parity_accuracy']:.4f} | {row['joint_exact']:.4f} | "
            f"{row['routing_accuracy']:.4f} | {'PASS' if row['passes'] else 'FAIL'} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/parity_goldreich_toy_transformer_ablations"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
