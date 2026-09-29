#!/usr/bin/env python3
"""Compare composition accuracy and retained skills across fresh LoRA ranks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(config_path: Path, artifacts_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    for rank in map(int, config["ranks"]):
        metrics = json.loads(
            (artifacts_root / f"rank_{rank}" / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        final = metrics["final_evaluations"]
        rows.append(
            {
                "rank": rank,
                "trainable_parameters": metrics["trainable_parameters"],
                "best_epoch": metrics["best_epoch"],
                "composition_accuracy": final["composition"]["greedy_accuracy"],
                "composition_bit_ce": final["composition"]["bit_ce"],
                "local_retention": final["local"]["greedy_accuracy"],
                "supplied_update_retention": final["supplied_update"][
                    "greedy_accuracy"
                ],
                "other_position_accuracy": final["other_position"]["greedy_accuracy"],
                "selected_attention_mass": metrics["final_attention"][
                    "selected_attention_mass"
                ],
                "selected_seed_ratio": metrics["final_attention"][
                    "selected_seed_ratio"
                ],
                "passes": metrics["passes_gate"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )
    passing = [row for row in rows if row["passes"]]
    if passing:
        selected = min(passing, key=lambda row: row["rank"])
        diagnosis = f"Composition succeeds at rank {selected['rank']}; adapter capacity was a limiting factor."
        next_step = f"Use rank {selected['rank']} with the frozen mechanism for the two-derived-mask update."
    else:
        selected = max(rows, key=lambda row: row["composition_accuracy"])
        diagnosis = (
            "No tested rank learns the composition while retaining the established mechanisms; "
            "rank alone is not the solution."
        )
        next_step = "Keep the frozen mechanism and add training-only supervision for the intermediate mask."
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "condition": config["condition"],
        "mechanism_merged_and_frozen": True,
        "lowest_passing_rank": min((row["rank"] for row in passing), default=None),
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
        "# Goldreich composition LoRA-rank sweep",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "The successful mechanism adapter was merged into frozen base weights before each fresh LoRA was trained.",
        "",
        "| Rank | Trainable parameters | Composition | Bit CE | Local | Supplied update | Attention | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['trainable_parameters']:,} | "
            f"{row['composition_accuracy']:.3f} | {row['composition_bit_ce']:.3f} | "
            f"{row['local_retention']:.3f} | {row['supplied_update_retention']:.3f} | "
            f"{row['selected_attention_mass']:.3f} | {'pass' if row['passes'] else 'fail'} |"
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
        default=Path("artifacts/ce_parity_goldreich_composition_rank"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.config, args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
