#!/usr/bin/env python3
"""Summarize paired paper-style filler sweeps and test the preregistered lengths."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _mcnemar_exact(wrong_to_right: int, right_to_wrong: int) -> float:
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_right, right_to_wrong)
    numerator = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def _paired_bootstrap_ci(
    *,
    wrong_to_right: int,
    right_to_wrong: int,
    n: int,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    unchanged = n - wrong_to_right - right_to_wrong
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n,
        [wrong_to_right / n, right_to_wrong / n, unchanged / n],
        size=samples,
    )
    deltas = (counts[:, 0] - counts[:, 1]) / n
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(deltas, tail)), float(np.quantile(deltas, 1.0 - tail))


def _comparison(
    baseline_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    baseline = {str(row["item_id"]): row for row in baseline_rows}
    condition = {str(row["item_id"]): row for row in condition_rows}
    if set(baseline) != set(condition):
        raise RuntimeError("Filler conditions do not contain identical item IDs.")
    n = len(baseline)
    baseline_correct = sum(bool(row["correct"]) for row in baseline.values())
    condition_correct = sum(bool(row["correct"]) for row in condition.values())
    wrong_to_right = sum(
        not bool(baseline[item_id]["correct"]) and bool(condition[item_id]["correct"])
        for item_id in baseline
    )
    right_to_wrong = sum(
        bool(baseline[item_id]["correct"]) and not bool(condition[item_id]["correct"])
        for item_id in baseline
    )
    ci_low, ci_high = _paired_bootstrap_ci(
        wrong_to_right=wrong_to_right,
        right_to_wrong=right_to_wrong,
        n=n,
        samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "n": n,
        "baseline_accuracy": baseline_correct / n,
        "accuracy": condition_correct / n,
        "delta_vs_k0": (condition_correct - baseline_correct) / n,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "mcnemar_exact_p": _mcnemar_exact(wrong_to_right, right_to_wrong),
        "paired_bootstrap_ci_low": ci_low,
        "paired_bootstrap_ci_high": ci_high,
        "parse_rate": sum(row.get("predicted") is not None for row in condition.values()) / n,
        "mean_prompt_tokens": sum(int(row["prompt_tokens"]) for row in condition.values()) / n,
        "filler_model_tokens_per_region": int(
            next(iter(condition.values())).get("filler_model_tokens_per_region", 0)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(*, config_path: Path, responses_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs: dict[str, dict[str, Any]] = {}
    for task in config["tasks"]:
        path = responses_dir / task / f"{task}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing task result: {path}")
        outputs[task] = json.loads(path.read_text(encoding="utf-8"))

    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    task_gates: dict[str, Any] = {}
    low, high = (float(value) for value in config["medium_baseline_accuracy_interval"])
    alpha = float(config["alpha"])
    component_minimum = float(config["component_accuracy_minimum"])

    for task, output in outputs.items():
        main_rows = output["rows"]
        components = output["component_rows"]
        component_accuracy = sum(bool(row["correct"]) for row in components) / len(components)
        component_rows.append(
            {"task": task, "subtask": "all", "n": len(components), "accuracy": component_accuracy}
        )
        component_subtasks = sorted({str(row["subtask"]) for row in components})
        for subtask in component_subtasks:
            selected = [row for row in components if row["subtask"] == subtask]
            component_rows.append(
                {
                    "task": task,
                    "subtask": subtask,
                    "n": len(selected),
                    "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected),
                }
            )

        scored_groups = [(task, main_rows)]
        if task == "letter_position":
            for subtask in ("element_letter", "capital_letter"):
                scored_groups.append(
                    (subtask, [row for row in main_rows if row["subtask"] == subtask])
                )
        for group_name, group_rows in scored_groups:
            baseline_rows = [row for row in group_rows if int(row["filler_length"]) == 0]
            for k in (int(value) for value in config["filler_lengths"]):
                condition_rows = [row for row in group_rows if int(row["filler_length"]) == k]
                comparison = _comparison(
                    baseline_rows,
                    condition_rows,
                    bootstrap_samples=int(config["bootstrap_samples"]),
                    confidence=float(config["confidence_level"]),
                    seed=int(config["seed"]) + k + sum(ord(char) for char in group_name),
                )
                metric_rows.append(
                    {
                        "task": task,
                        "group": group_name,
                        "filler_length": k,
                        **comparison,
                    }
                )

        primary_k = int(config["primary_filler_length"][task])
        primary = next(
            row
            for row in metric_rows
            if row["task"] == task and row["group"] == task and row["filler_length"] == primary_k
        )
        checks = {
            "component_knowledge": component_accuracy >= component_minimum,
            "baseline_in_medium_band": low <= float(primary["baseline_accuracy"]) <= high,
            "positive_primary_delta": float(primary["delta_vs_k0"]) > 0.0,
            "primary_mcnemar": float(primary["mcnemar_exact_p"]) < alpha,
            "positive_primary_ci": float(primary["paired_bootstrap_ci_low"]) > 0.0,
        }
        task_gates[task] = {
            "primary_filler_length": primary_k,
            "component_accuracy": component_accuracy,
            "checks": checks,
            "medium_filler_signal": all(checks.values()),
        }

    summary = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model_id": config["model_id"],
        "seed": config["seed"],
        "inference_only": True,
        "paper_protocol": {
            "filler_type": "dots",
            "filler_lengths": config["filler_lengths"],
            "few_shot_per_task": 5,
            "placement": "between each question and Answer: in demonstrations and target",
        },
        "task_gates": task_gates,
        "any_medium_filler_signal": any(row["medium_filler_signal"] for row in task_gates.values()),
        "metrics": metric_rows,
        "component_metrics": component_rows,
        "claim_boundary": (
            "This is a seed-0 inference screen on one 8B model family. A negative result does not "
            "show that knowledge tasks or Llama models cannot use filler tokens."
        ),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_dir / "accuracy_by_task_and_filler.csv", metric_rows)
    _write_csv(artifacts_dir / "component_accuracy.csv", component_rows)
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Llama 3.1 8B knowledge filler screen",
        "",
        "Inference-only reproduction of the paper's five-shot dot-filler protocol.",
        "",
        "| Task/group | K | N | Accuracy | Delta vs K=0 | Wrong→right | Right→wrong | McNemar p | Paired 95% CI | Prompt tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['group']}` | {row['filler_length']} | {row['n']} | "
            f"{row['accuracy']:.3f} | {row['delta_vs_k0']:+.3f} | "
            f"{row['wrong_to_right']} | {row['right_to_wrong']} | "
            f"{row['mcnemar_exact_p']:.4g} | "
            f"[{row['paired_bootstrap_ci_low']:+.3f}, {row['paired_bootstrap_ci_high']:+.3f}] | "
            f"{row['mean_prompt_tokens']:.1f} |"
        )
    lines.extend(["", "## Component knowledge", "", "| Task | Probe type | N | Accuracy |", "|---|---|---:|---:|"])
    for row in component_rows:
        lines.append(
            f"| `{row['task']}` | `{row['subtask']}` | {row['n']} | {row['accuracy']:.3f} |"
        )
    lines.extend(["", "## Preregistered Medium signal", ""])
    for task, gate in task_gates.items():
        lines.append(
            f"- `{task}` at K={gate['primary_filler_length']}: "
            f"**{'PASS' if gate['medium_filler_signal'] else 'NO SIGNAL'}**"
        )
    lines.extend(["", f"> {summary['claim_boundary']}", ""])
    (artifacts_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report under {artifacts_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=Path("generated_data/llama31_8b_knowledge_filler_responses"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/llama31_8b_knowledge_filler_screen"),
    )
    args = parser.parse_args()
    summarize(
        config_path=args.config,
        responses_dir=args.responses_dir,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
