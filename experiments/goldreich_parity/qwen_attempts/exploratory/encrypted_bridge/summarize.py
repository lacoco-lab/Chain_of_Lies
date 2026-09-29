#!/usr/bin/env python3
"""Summarize supplied controls, derived bridge tasks, and retained mechanisms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(config_path: Path, training_root: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(
        (training_root / "training_metrics.json").read_text(encoding="utf-8")
    )
    final = metrics["final_evaluations"]
    gate = config["success_gate"]
    checks = {
        "supplied_start": final["supplied_start"]["accuracy"]["greedy_accuracy"]
        >= gate["minimum_supplied_accuracy"],
        "derived_start": final["derived_start"]["accuracy"]["greedy_accuracy"]
        >= gate["minimum_derived_accuracy"],
        "supplied_update": final["supplied_update"]["accuracy"]["greedy_accuracy"]
        >= gate["minimum_supplied_accuracy"],
        "derived_update": final["derived_update"]["accuracy"]["greedy_accuracy"]
        >= gate["minimum_derived_accuracy"],
        "local_retention": final["local"]["accuracy"]["greedy_accuracy"]
        >= gate["minimum_local_retention"],
        "start_retention": final["derived_start"]["accuracy"]["greedy_accuracy"]
        >= gate["minimum_start_retention"],
    }
    rows = []
    for task in (
        "local",
        "supplied_start",
        "derived_start",
        "supplied_update",
        "derived_update",
    ):
        accuracy, attention = final[task]["accuracy"], final[task]["attention"]
        rows.append(
            {
                "task": task,
                "n": accuracy["n"],
                "greedy_accuracy": accuracy["greedy_accuracy"],
                "binary_accuracy": accuracy["binary_accuracy"],
                "bit_ce": accuracy["bit_ce"],
                "selected_attention_mass": (
                    attention["selected_attention_mass"] if attention else None
                ),
                "selected_seed_ratio": (
                    attention["selected_seed_ratio"] if attention else None
                ),
            }
        )
    first_failure = next(
        (
            task
            for task in (
                "supplied_start",
                "derived_start",
                "supplied_update",
                "derived_update",
            )
            if not checks[task]
        ),
        None,
    )
    if first_failure is None:
        diagnosis = "Both one-bit encryption bridges generalize; proceed to a gated two-step trace."
    elif first_failure == "supplied_start":
        diagnosis = "The model cannot learn even the supplied-mask start XOR."
    elif first_failure == "derived_start":
        diagnosis = "Supplied XOR works, but the learned mask cannot be composed into the first encrypted state."
    elif first_failure == "supplied_update":
        diagnosis = (
            "Encrypted start works, but the five-input supplied-mask update fails."
        )
    else:
        diagnosis = "Supplied update works, but deriving and composing two masks in one update fails."
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "baseline_local_accuracy": metrics["baseline_local"]["greedy_accuracy"],
        "checks": checks,
        "first_failure": first_failure,
        "diagnosis": diagnosis,
        "passes_all_bridges": all(checks.values()),
        "elapsed_seconds": metrics["elapsed_seconds"],
        "metrics": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Goldreich encrypted-state bridge",
        "",
        f"Overall: **{'PASS' if result['passes_all_bridges'] else 'FAIL'}**",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "| Task | N | Accuracy | Bit CE | Selected attention | Selected-seed ratio | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        attention = (
            "n/a"
            if row["selected_attention_mass"] is None
            else f"{row['selected_attention_mass']:.3f}"
        )
        ratio = (
            "n/a"
            if row["selected_seed_ratio"] is None
            else f"{row['selected_seed_ratio']:.3f}"
        )
        gate_name = "local_retention" if row["task"] == "local" else row["task"]
        lines.append(
            f"| {row['task']} | {row['n']} | {row['greedy_accuracy']:.3f} | "
            f"{row['bit_ce']:.3f} | {attention} | {ratio} | "
            f"{'pass' if checks[gate_name] else 'fail'} |"
        )
    lines.append("")
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_encrypted_bridge/training"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_encrypted_bridge"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.config, args.training_root, args.output_root), indent=2
        )
    )


if __name__ == "__main__":
    main()
