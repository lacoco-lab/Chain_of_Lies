#!/usr/bin/env python3
"""Summarize the paired one-derived-mask update ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONDITIONS = ("derive_previous", "derive_current")


def summarize(config_path: Path, artifacts_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = config["success_gate"]
    base = json.loads(
        (artifacts_root / "base" / "metrics.json").read_text(encoding="utf-8")
    )
    branches = {
        condition: json.loads(
            (artifacts_root / condition / "metrics.json").read_text(encoding="utf-8")
        )
        for condition in CONDITIONS
    }
    rows, checks = [], {}
    for condition, metrics in branches.items():
        final = metrics["final_evaluations"]
        own = final[condition]["greedy_accuracy"]
        local = final["local"]["greedy_accuracy"]
        supplied = final["supplied_update"]["greedy_accuracy"]
        passes = bool(
            own >= float(gate["minimum_condition_accuracy"])
            and local >= float(gate["minimum_local_retention"])
            and supplied >= float(gate["minimum_supplied_update_accuracy"])
        )
        checks[condition] = passes
        attention = metrics["final_attention"]
        rows.append(
            {
                "condition": condition,
                "n": final[condition]["n"],
                "accuracy": own,
                "bit_ce": final[condition]["bit_ce"],
                "local_retention": local,
                "supplied_update_retention": supplied,
                "other_condition_accuracy": final[
                    (
                        "derive_current"
                        if condition == "derive_previous"
                        else "derive_previous"
                    )
                ]["greedy_accuracy"],
                "selected_attention_mass": attention["selected_attention_mass"],
                "selected_seed_ratio": attention["selected_seed_ratio"],
                "passes": passes,
            }
        )
    if all(checks.values()):
        diagnosis = (
            "Both one-derived-mask updates work. The remaining bottleneck is deriving and "
            "composing two Goldreich masks in the same update."
        )
        next_step = (
            "Train the full update with two separate internal mask targets and separate "
            "attention supervision for the previous and current selectors."
        )
    elif checks["derive_previous"]:
        diagnosis = "The update works when only z_prev is derived, but not when only z_cur is derived."
        next_step = "Add direct auxiliary supervision for the current-mask computation before recombining both masks."
    elif checks["derive_current"]:
        diagnosis = "The update works when only z_cur is derived, but not when only z_prev is derived."
        next_step = "Add direct auxiliary supervision for the previous-mask computation before recombining both masks."
    else:
        diagnosis = (
            "Neither one-derived-mask update works. The bottleneck is composing even one "
            "derived Goldreich mask with the encrypted-state update."
        )
        next_step = (
            "Use a training-only intermediate mask target, then train the final XOR while "
            "keeping that intermediate computation supervised."
        )
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "base_supplied_update_accuracy": base["final_evaluations"]["supplied_update"][
            "greedy_accuracy"
        ],
        "base_local_accuracy": base["final_evaluations"]["local"]["greedy_accuracy"],
        "checks": checks,
        "diagnosis": diagnosis,
        "recommended_next_step": next_step,
        "metrics": rows,
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
        "# Goldreich one-derived-mask update ablation",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        f"Common supplied-update base: {result['base_supplied_update_accuracy']:.3f}; "
        f"local retention: {result['base_local_accuracy']:.3f}.",
        "",
        "| Condition | N | Accuracy | Bit CE | Local | Supplied update | Attention | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['bit_ce']:.3f} | {row['local_retention']:.3f} | "
            f"{row['supplied_update_retention']:.3f} | {row['selected_attention_mass']:.3f} | "
            f"{'pass' if row['passes'] else 'fail'} |"
        )
    lines.extend(["", f"Next: {next_step}", ""])
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_update_ablation"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.config, args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
