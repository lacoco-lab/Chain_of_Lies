#!/usr/bin/env python3
"""Combine the focused-bit control and attention-regularized results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(config_path: Path, artifacts_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = {}
    for condition in config["conditions"]:
        path = artifacts_root / condition / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing condition metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        best = metrics["history"][metrics["best_epoch"] - 1]
        conditions[condition] = {
            "attention_regularization": metrics["attention_regularization"],
            "baseline_greedy_accuracy": metrics["baseline_validation"][
                "greedy_accuracy"
            ],
            "best_epoch": metrics["best_epoch"],
            "best_validation_greedy_accuracy": metrics[
                "best_validation_greedy_accuracy"
            ],
            "best_validation_binary_accuracy": best["validation"]["binary_accuracy"],
            "best_train_greedy_accuracy": best["train_evaluation"]["greedy_accuracy"],
            "validation_attention": best["validation_attention"],
            "passes_gate": metrics["passes_gate"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        }
    control = conditions["bit_only"]["best_validation_greedy_accuracy"]
    main = conditions["bit_attention"]["best_validation_greedy_accuracy"]
    conclusion = (
        "The explicit lookup primitive passes on unseen seeds; proceed to the full encrypted CoT."
        if conditions["bit_attention"]["passes_gate"]
        else "The explicit lookup primitive still fails on unseen seeds; do not start full encrypted-CoT training."
    )
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "train_validation_seed_overlap": 0,
        "gate_threshold": config["gate_validation_accuracy"],
        "conditions": conditions,
        "attention_accuracy_delta": main - control,
        "main_passes_gate": conditions["bit_attention"]["passes_gate"],
        "conclusion": conclusion,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (artifacts_root / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "attention_regularization",
                "baseline_greedy_accuracy",
                "best_epoch",
                "train_greedy_accuracy",
                "validation_greedy_accuracy",
                "validation_binary_accuracy",
                "passes_gate",
                "elapsed_seconds",
            ]
        )
        for name, values in conditions.items():
            writer.writerow(
                [
                    name,
                    values["attention_regularization"],
                    values["baseline_greedy_accuracy"],
                    values["best_epoch"],
                    values["best_train_greedy_accuracy"],
                    values["best_validation_greedy_accuracy"],
                    values["best_validation_binary_accuracy"],
                    values["passes_gate"],
                    values["elapsed_seconds"],
                ]
            )
    lines = [
        "# Goldreich one-bit retrieval experiment",
        "",
        "The train and validation sets use disjoint 16-bit seeds. Each of the 64 fixed public "
        "selectors is exactly balanced between mask bits 0 and 1.",
        "",
        "| Condition | Baseline | Best train | Best validation | Epoch | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, values in conditions.items():
        lines.append(
            f"| {name} | {values['baseline_greedy_accuracy']:.3f} | "
            f"{values['best_train_greedy_accuracy']:.3f} | "
            f"{values['best_validation_greedy_accuracy']:.3f} | "
            f"{values['best_epoch']} | {'pass' if values['passes_gate'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            f"Attention accuracy delta: {result['attention_accuracy_delta']:+.3f}.",
            "",
            conclusion,
            "",
        ]
    )
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
        default=Path("artifacts/ce_parity_goldreich_bit_attention"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.config, args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
