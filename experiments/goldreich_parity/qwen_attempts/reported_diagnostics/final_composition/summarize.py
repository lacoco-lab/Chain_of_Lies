#!/usr/bin/env python3
"""Produce the final Goldreich encrypted-update report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASKS = (
    "local",
    "supplied_update",
    "derive_previous",
    "derive_current",
    "mask_delta",
    "full_update",
)
CURRICULUM_TASKS = ("full_zero", "full_one", "full_two")


def summarize(config_path: Path, artifacts_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    positions = json.loads(
        (artifacts_root / "positions" / "metrics.json").read_text(encoding="utf-8")
    )
    full = json.loads(
        (artifacts_root / "full" / "metrics.json").read_text(encoding="utf-8")
    )
    gate = config["success_gate"]
    final = full["final_evaluations"]
    rows = []
    for task in TASKS:
        source = final if task in final else positions["final_evaluations"]
        metrics = source[task]
        threshold = (
            gate["minimum_supplied_update_retention"]
            if task == "supplied_update"
            else (
                gate["minimum_local_retention"]
                if task == "local"
                else gate["minimum_composition_accuracy"]
            )
        )
        rows.append(
            {
                "task": task,
                "n": metrics["n"],
                "accuracy": metrics["greedy_accuracy"],
                "bit_ce": metrics["bit_ce"],
                "threshold": threshold,
                "passes": metrics["greedy_accuracy"] >= threshold,
            }
        )
    curriculum_rows = []
    for task in CURRICULUM_TASKS:
        metrics = final[task]
        curriculum_rows.append(
            {
                "task": task,
                "n": metrics["n"],
                "accuracy": metrics["greedy_accuracy"],
                "bit_ce": metrics["bit_ce"],
            }
        )
    passes = all(row["passes"] for row in rows)
    diagnosis = (
        "The complete two-mask encrypted update generalizes while every prerequisite is retained."
        if passes
        else "The final two-mask update did not pass all accuracy and retention gates."
    )
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "architecture": (
            "frozen mechanism + rank-32 positions + rank-32 mask-delta/public-XOR curriculum"
        ),
        "positions_gate_passes": positions["passes_gate"],
        "full_gate_passes": full["passes_gate"],
        "passes_all_gates": passes and positions["passes_gate"] and full["passes_gate"],
        "diagnosis": diagnosis,
        "position_elapsed_seconds": positions["elapsed_seconds"],
        "full_elapsed_seconds": full["elapsed_seconds"],
        "full_attention": full["final_attention"],
        "metrics": rows,
        "curriculum_diagnostics": curriculum_rows,
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
        "# Final Goldreich encrypted update",
        "",
        f"Overall: **{'PASS' if result['passes_all_gates'] else 'FAIL'}**",
        "",
        diagnosis,
        "",
        "Architecture: frozen mechanism → fresh rank-32 position adapter → "
        "fresh rank-32 mask-delta/public-XOR curriculum adapter.",
        "",
        "| Task | N | Accuracy | Bit CE | Gate |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['bit_ce']:.3f} | {'pass' if row['passes'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "Curriculum diagnostics:",
            "",
            "| Bridge | N | Accuracy | Bit CE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in curriculum_rows:
        lines.append(
            f"| {row['task']} | {row['n']} | {row['accuracy']:.3f} | {row['bit_ce']:.3f} |"
        )
    lines.append("")
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
        default=Path("artifacts/ce_parity_goldreich_final"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.config, args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
