#!/usr/bin/env python3
"""Validate and package the paper-facing staged PARITY controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CONDITIONS = ("no_cot", "filler_cot")
LENGTHS = (4, 8, 16, 32, 64)
LABELS = {"no_cot": "No CoT", "filler_cot": "Filler-only CoT"}


def source_path(easy_root: Path, large_root: Path, condition: str, length: int) -> Path:
    root = easy_root if length <= 16 else large_root
    return root / condition / f"n{length}" / "metrics.json"


def load_staged(path: Path, condition: str, length: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing staged control result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["condition"] != condition or int(result["length"]) != length:
        raise RuntimeError(f"Metadata mismatch in {path}")
    for key in ("routing_training", "representation_training", "public_training"):
        if not result[key].get("enabled", False):
            raise RuntimeError(
                f"Paper-facing result is not staged ({key} disabled): {path}"
            )
    if length > 16 and result.get("training_regime") != "staged":
        raise RuntimeError(f"Large-N result lacks staged provenance: {path}")
    if (
        int(result["architecture"]["heads"]) != 16
        or int(result["architecture"]["model_dim"]) != 128
    ):
        raise RuntimeError(f"Architecture mismatch in {path}")
    return result


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--easy-root", type=Path, default=Path("artifacts/parity_standard_cot_controls")
    )
    parser.add_argument(
        "--large-root",
        type=Path,
        default=Path("artifacts/parity_standard_cot_controls_staged"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/parity_standard_cot_controls_staged_summary"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for length in LENGTHS:
            path = source_path(args.easy_root, args.large_root, condition, length)
            result = load_staged(path, condition, length)
            test = result["test_softmax_attention"]
            recipe = result.get("staged_recipe") or {}
            rows.append(
                {
                    "condition": condition,
                    "condition_label": LABELS[condition],
                    "N": length,
                    "train_examples": int(result["train_examples"]),
                    "validation_examples": int(result["validation_examples"]),
                    "test_examples": int(result["test_examples"]),
                    "filler_token_accuracy": (
                        test["cot_bit_accuracy"] if condition == "filler_cot" else None
                    ),
                    "filler_tokens_evaluated": (
                        int(result["test_examples"]) * length
                        if condition == "filler_cot"
                        else None
                    ),
                    "entire_filler_cot_accuracy": (
                        test["cot_exact"] if condition == "filler_cot" else None
                    ),
                    "final_parity_accuracy": float(test["final_parity_accuracy"]),
                    "training_regime": "staged",
                    "directly_supervised_input_bits": int(
                        recipe.get("directly_supervised_input_bits", length)
                    ),
                    "deviation_from_easy_recipe": recipe.get(
                        "deviation_from_easy_recipe"
                    ),
                    "source_metrics": str(path),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    csv_path = args.output_root / "staged_control_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Condition | N | Test examples | Filler tokens | Filler token accuracy (%) | "
        "Entire filler CoT (%) | Final PARITY (%) |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition_label']} | {row['N']} | {row['test_examples']:,} | "
            f"{row['filler_tokens_evaluated'] or '--'} | "
            f"{fmt(row['filler_token_accuracy'])} | {fmt(row['entire_filler_cot_accuracy'])} | "
            f"{fmt(row['final_parity_accuracy'])} |"
        )
    table_path = args.output_root / "staged_control_results.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    deviations = sorted(
        {
            row["deviation_from_easy_recipe"]
            for row in rows
            if row["deviation_from_easy_recipe"]
        }
    )
    summary = {
        "schema_version": 1,
        "paper_facing_training_regime": "staged",
        "conditions": list(CONDITIONS),
        "lengths": list(LENGTHS),
        "end_to_end_diagnostics_included": False,
        "architecture": {"layers": 1, "heads": 16, "model_dim": 128},
        "deviations_from_easy_recipe": deviations,
        "rows": rows,
    }
    summary_path = args.output_root / "staged_control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(table_path.read_text(encoding="utf-8"), end="")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "table": str(table_path),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
