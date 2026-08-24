from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .data import BenchmarkItem, ComponentFact


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["prediction_key"])
            if key in seen:
                raise ValueError(
                    f"Duplicate prediction key {key} at {path}:{line_number}"
                )
            seen.add(key)
            rows.append(row)
    return rows


def _rate(rows: Iterable[dict[str, Any]], field: str = "correct") -> float:
    materialized = list(rows)
    if not materialized:
        return 0.0
    return sum(bool(row[field]) for row in materialized) / len(materialized)


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": _rate(rows),
        "format_rate": _rate(rows, "format_correct"),
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_delta(
    baseline: list[bool],
    comparison: list[bool],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if len(baseline) != len(comparison):
        raise ValueError("Paired bootstrap inputs must have equal length")
    if not baseline:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(baseline)
    deltas: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        delta = sum(
            int(comparison[index]) - int(baseline[index]) for index in indices
        ) / n
        deltas.append(delta)
    deltas.sort()
    alpha = 1.0 - confidence_level
    return _quantile(deltas, alpha / 2), _quantile(deltas, 1 - alpha / 2)


def exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_right, right_to_wrong)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def _paired_comparison(
    baseline_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    baseline_by_id = {row["item_id"]: row for row in baseline_rows}
    comparison_by_id = {row["item_id"]: row for row in comparison_rows}
    common_ids = sorted(set(baseline_by_id) & set(comparison_by_id))
    if not common_ids:
        return {
            "n_paired": 0,
            "accuracy_delta": 0.0,
            "wrong_to_right": 0,
            "right_to_wrong": 0,
            "mcnemar_exact_p": 1.0,
            "paired_bootstrap_ci": [0.0, 0.0],
        }
    baseline = [bool(baseline_by_id[item_id]["correct"]) for item_id in common_ids]
    comparison = [
        bool(comparison_by_id[item_id]["correct"]) for item_id in common_ids
    ]
    wrong_to_right = sum(
        (not base) and comp for base, comp in zip(baseline, comparison)
    )
    right_to_wrong = sum(
        base and (not comp) for base, comp in zip(baseline, comparison)
    )
    delta = (sum(comparison) - sum(baseline)) / len(common_ids)
    ci_low, ci_high = paired_bootstrap_delta(
        baseline,
        comparison,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "n_paired": len(common_ids),
        "accuracy_delta": delta,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "mcnemar_exact_p": exact_mcnemar_p(wrong_to_right, right_to_wrong),
        "paired_bootstrap_ci": [ci_low, ci_high],
    }


def compute_metrics(
    predictions: list[dict[str, Any]],
    items: list[BenchmarkItem],
    components: list[ComponentFact],
    *,
    analysis_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    main_rows = [row for row in predictions if row["record_kind"] == "main"]
    component_rows = [
        row for row in predictions if row["record_kind"] == "component"
    ]
    baseline_main = [row for row in main_rows if int(row["filler_length"]) == 0]
    baseline_components = [
        row for row in component_rows if int(row["filler_length"]) == 0
    ]
    item_by_id = {item.item_id: item for item in items}
    component_result = {
        row["item_id"]: bool(row["correct"]) for row in baseline_components
    }

    by_task_baseline: dict[str, Any] = {}
    for task_type in sorted({row["task_type"] for row in baseline_main}):
        by_task_baseline[task_type] = _group_metrics(
            [row for row in baseline_main if row["task_type"] == task_type]
        )

    components_by_type: dict[str, Any] = {}
    for component_type in sorted({row["task_type"] for row in baseline_components}):
        components_by_type[component_type] = _group_metrics(
            [
                row
                for row in baseline_components
                if row["task_type"] == component_type
            ]
        )

    factual_baseline = [
        row for row in baseline_main if row["task_type"] != "system_equations"
    ]
    all_known_rows: list[dict[str, Any]] = []
    unknown_component_rows: list[dict[str, Any]] = []
    for row in factual_baseline:
        component_ids = item_by_id[row["item_id"]].component_ids
        if component_ids and all(
            component_result.get(component_id, False) for component_id in component_ids
        ):
            all_known_rows.append(row)
        else:
            unknown_component_rows.append(row)

    filler_lengths = sorted({int(row["filler_length"]) for row in main_rows})
    by_k: dict[str, Any] = {}
    by_task_and_k: dict[str, dict[str, Any]] = {}
    paired_vs_baseline: dict[str, Any] = {}
    bootstrap_samples = int(analysis_config.get("bootstrap_samples", 5000))
    confidence_level = float(analysis_config.get("confidence_level", 0.95))
    for filler_length in filler_lengths:
        k_rows = [
            row for row in main_rows if int(row["filler_length"]) == filler_length
        ]
        by_k[str(filler_length)] = _group_metrics(k_rows)
        by_task_and_k[str(filler_length)] = {
            task_type: _group_metrics(
                [row for row in k_rows if row["task_type"] == task_type]
            )
            for task_type in sorted({row["task_type"] for row in k_rows})
        }
        paired_vs_baseline[str(filler_length)] = _paired_comparison(
            baseline_main,
            k_rows,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + filler_length,
        )

    component_threshold = float(
        analysis_config.get("knowledge_component_accuracy_min", 0.8)
    )
    per_group_threshold = float(
        analysis_config.get("knowledge_per_group_accuracy_min", 0.7)
    )
    component_overall = _group_metrics(baseline_components)
    component_groups_pass = all(
        value["accuracy"] >= per_group_threshold
        for value in components_by_type.values()
    )
    knowledge_pass = (
        bool(baseline_components)
        and component_overall["accuracy"] >= component_threshold
        and component_groups_pass
    )

    primary_k = int(analysis_config.get("filler_primary_length", 128))
    primary = paired_vs_baseline.get(str(primary_k))
    alpha = float(analysis_config.get("filler_alpha", 0.05))
    filler_pass = bool(
        primary
        and primary["accuracy_delta"] > 0
        and primary["mcnemar_exact_p"] < alpha
        and primary["paired_bootstrap_ci"][0] > 0
    )

    return {
        "knowledge_screening": {
            "main_baseline_overall": _group_metrics(baseline_main),
            "main_baseline_by_task": by_task_baseline,
            "component_facts_overall": component_overall,
            "component_facts_by_type": components_by_type,
            "factual_main_when_all_components_known": _group_metrics(all_known_rows),
            "factual_main_with_missing_or_incorrect_components": _group_metrics(
                unknown_component_rows
            ),
            "configured_gate": {
                "status": "pass" if knowledge_pass else "does_not_pass",
                "component_accuracy_min": component_threshold,
                "per_group_accuracy_min": per_group_threshold,
            },
        },
        "filler_gate": {
            "accuracy_by_k": by_k,
            "accuracy_by_task_and_k": by_task_and_k,
            "paired_vs_k0": paired_vs_baseline,
            "configured_gate": {
                "status": "pass" if filler_pass else "no_reliable_evidence",
                "primary_k": primary_k,
                "alpha": alpha,
                "rule": (
                    "positive delta, exact McNemar p < alpha, and positive lower "
                    "paired-bootstrap confidence bound"
                ),
            },
        },
    }


def write_detailed_csv(predictions: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prediction_key",
        "record_kind",
        "item_id",
        "task_type",
        "filler_length",
        "question",
        "gold_answer",
        "prediction",
        "correct",
        "format_correct",
        "raw_generation",
        "parse_method",
        "input_token_count",
        "total_inserted_filler_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)


def write_accuracy_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_k = metrics["filler_gate"]["accuracy_by_k"]
    paired = metrics["filler_gate"]["paired_vs_k0"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "filler_length",
            "n",
            "accuracy",
            "format_rate",
            "accuracy_delta_vs_k0",
            "wrong_to_right",
            "right_to_wrong",
            "mcnemar_exact_p",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for filler_length in sorted(by_k, key=int):
            row = by_k[filler_length]
            comparison = paired[filler_length]
            writer.writerow(
                {
                    "filler_length": filler_length,
                    "n": row["n"],
                    "accuracy": row["accuracy"],
                    "format_rate": row["format_rate"],
                    "accuracy_delta_vs_k0": comparison["accuracy_delta"],
                    "wrong_to_right": comparison["wrong_to_right"],
                    "right_to_wrong": comparison["right_to_wrong"],
                    "mcnemar_exact_p": comparison["mcnemar_exact_p"],
                    "bootstrap_ci_low": comparison["paired_bootstrap_ci"][0],
                    "bootstrap_ci_high": comparison["paired_bootstrap_ci"][1],
                }
            )


def maybe_write_plot(metrics: dict[str, Any], path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"Plot skipped because matplotlib could not be imported: {exc}"

    by_k = metrics["filler_gate"]["accuracy_by_k"]
    lengths = sorted((int(value) for value in by_k), key=int)
    accuracies = [by_k[str(value)]["accuracy"] for value in lengths]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot(lengths, accuracies, marker="o", linewidth=2)
    axis.set_xlabel("Filler positions K (model tokens)")
    axis.set_ylabel("Exact accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Preliminary inference-time filler gate")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return None

