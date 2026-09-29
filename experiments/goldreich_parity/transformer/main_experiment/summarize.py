#!/usr/bin/env python3
"""Summarize one or more Goldreich toy-transformer runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(artifacts_root: Path, lengths: list[int]) -> dict:
    rows, details = [], {}
    for length in lengths:
        metrics = json.loads(
            (artifacts_root / f"n{length}" / "metrics.json").read_text(encoding="utf-8")
        )
        test = metrics["test_hard_attention"]
        row = {
            "length": length,
            "seed_bits": metrics["seed_bits"],
            "mask_bits": metrics["mask_bits"],
            "test_n": test["n"],
            "state_bit_accuracy": test["state_bit_accuracy"],
            "trace_exact": test["trace_exact"],
            "final_parity_accuracy": test["final_parity_accuracy"],
            "joint_exact": test["joint_exact"],
            "routing_accuracy": metrics["final_routing_metrics"]["accuracy"],
            "minimum_target_attention": metrics["final_routing_metrics"][
                "minimum_target_attention"
            ],
            "passes": metrics["passes_gate"],
        }
        rows.append(row)
        details[str(length)] = metrics
    result = {
        "schema_version": 2,
        "experiment_name": details[str(lengths[0])]["experiment_name"],
        "all_lengths_pass": all(row["passes"] for row in rows),
        "rows": rows,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (artifacts_root / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Purpose-built Goldreich transformer",
        "",
        f"Overall: **{'PASS' if result['all_lengths_pass'] else 'FAIL'}**",
        "",
        "| Input length | Seed bits | Mask/CoT bits | Test N | State bits | Entire CoT | Final PARITY | CoT + answer | Routing | Min target attention | Result |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['length']} | {row['seed_bits']} | {row['mask_bits']} | "
            f"{row['test_n']} | {row['state_bit_accuracy']:.4f} | "
            f"{row['trace_exact']:.4f} | {row['final_parity_accuracy']:.4f} | "
            f"{row['joint_exact']:.4f} | {row['routing_accuracy']:.4f} | "
            f"{row['minimum_target_attention']:.4f} | "
            f"{'PASS' if row['passes'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Validation and test PRG seeds are disjoint from training seeds. The hard-attention",
            "completion contains only encrypted state bits and the final PARITY answer; masks and",
            "unencrypted states are internal and are never emitted.",
            "",
        ]
    )
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/parity_goldreich_toy_transformer"),
    )
    parser.add_argument("--length", type=int, action="append", dest="lengths")
    args = parser.parse_args()
    lengths = args.lengths or [32, 64, 256, 512]
    print(json.dumps(summarize(args.artifacts_root, lengths), indent=2))


if __name__ == "__main__":
    main()
