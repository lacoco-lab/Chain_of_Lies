#!/usr/bin/env python3
"""Summarize all matched seed-free PARITY control runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONDITIONS = ("no_cot", "filler_cot", "normal_cot")


def discover_lengths(root: Path) -> list[int]:
    per_condition = []
    for condition in CONDITIONS:
        lengths = {
            int(path.parent.name.removeprefix("n"))
            for path in (root / condition).glob("n*/metrics.json")
        }
        per_condition.append(lengths)
    complete = set.intersection(*per_condition) if per_condition else set()
    if not complete:
        raise FileNotFoundError(
            f"No complete three-condition lengths found under {root}"
        )
    return sorted(complete)


def summarize(root: Path, lengths: list[int]) -> dict:
    rows, experiment_names, data_sources = [], set(), set()
    for condition in CONDITIONS:
        for length in lengths:
            path = root / condition / f"n{length}" / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            experiment_names.add(metrics["experiment_name"])
            default_source = (
                "exhaustive finite input domain partitioned into disjoint balanced splits"
                if "_easy_" in metrics["experiment_name"]
                else "exact input examples from the encrypted experiment; seed-related fields ignored"
            )
            data_sources.add(metrics.get("data_source", default_source))
            test = metrics["test_hard_attention"]
            rows.append(
                {
                    "condition": condition,
                    "length": length,
                    "test_n": test["n"],
                    "cot_length": test["cot_length"],
                    "cot_bit_accuracy": test["cot_bit_accuracy"],
                    "cot_exact": test["cot_exact"],
                    "final_parity_accuracy": test["final_parity_accuracy"],
                    "parameters": metrics["architecture"]["trainable_parameters"],
                    "matches_expected_behavior": metrics["matches_expected_behavior"],
                }
            )
    result = {
        "schema_version": 1,
        "experiment_names": sorted(experiment_names),
        "data_sources": sorted(data_sources),
        "all_runs_match_expected_behavior": all(
            row["matches_expected_behavior"] for row in rows
        ),
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
        "# Seed-free PARITY controls",
        "",
        f"Overall expected pattern: **{'PASS' if result['all_runs_match_expected_behavior'] else 'FAIL'}**",
        "",
        "| Condition | N | CoT length | CoT bits | Entire CoT | Final PARITY | Parameters | Expected pattern |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        bit = (
            "—" if row["cot_bit_accuracy"] is None else f"{row['cot_bit_accuracy']:.4f}"
        )
        exact = "—" if row["cot_exact"] is None else f"{row['cot_exact']:.4f}"
        lines.append(
            f"| {row['condition']} | {row['length']} | {row['cot_length']} | "
            f"{bit} | {exact} | {row['final_parity_accuracy']:.4f} | "
            f"{row['parameters']} | {'PASS' if row['matches_expected_behavior'] else 'FAIL'} |"
        )
    lines.extend(["", "Data sources:"])
    lines.extend(f"- {source}" for source in sorted(data_sources))
    lines.extend(
        [
            "",
            "The control loader reads only IDs, input bits, and parity answers. No private seed,",
            "Goldreich mask, or encrypted state is supplied to any control model.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/parity_cot_controls")
    )
    parser.add_argument("--length", type=int, action="append", dest="lengths")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                args.artifacts_root,
                args.lengths or discover_lengths(args.artifacts_root),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
