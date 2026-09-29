#!/usr/bin/env python3
"""Recompute and aggregate balanced multiplication results by exact digit length."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.evaluation.rewards import load_prompt_record, score_completion
from experiments.scaling_regime.tasks.multiplication.generate_splits import (
    load_config,
    roots,
)

METRICS = (
    "task_success_rate",
    "public_exact_rate",
    "private_exact_rate",
    "payload_decode_rate",
    "concealment_rate",
    "format_rate",
    "avg_cot_words",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    data_root, artifacts_root, responses_root = roots(config)
    expected_n = int(config["eval_examples_per_length"])
    per_example: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for model in config["models"]:
        for seed in config["seeds"]:
            for condition, spec in config["conditions"].items():
                variant = str(spec["variant"])
                provenance = (
                    artifacts_root
                    / model
                    / f"seed_{seed}"
                    / condition
                    / "provenance"
                    / "validation_report.json"
                )
                if not provenance.is_file() or not json.loads(
                    provenance.read_text(encoding="utf-8")
                ).get("valid"):
                    raise RuntimeError(f"Missing valid provenance: {provenance}")
                for length in config["operand_digits"]:
                    report_path = (
                        artifacts_root
                        / model
                        / f"seed_{seed}"
                        / condition
                        / "per_length_eval"
                        / config["checkpoint"]
                        / f"length_{length}.json"
                    )
                    if not report_path.is_file():
                        raise RuntimeError(f"Missing evaluation report: {report_path}")
                    reported = json.loads(report_path.read_text(encoding="utf-8"))[
                        "metrics"
                    ]
                    if int(
                        reported.get("num_examples", 0)
                    ) != expected_n or reported.get("missing_responses"):
                        raise RuntimeError(f"Incomplete evaluation: {report_path}")
                    prompt_root = (
                        data_root / "shared_eval" / f"length_{length}" / variant
                    )
                    response_root = (
                        responses_root
                        / model
                        / f"seed_{seed}"
                        / condition
                        / f"length_{length}"
                        / config["checkpoint"]
                        / "trained"
                        / variant
                    )
                    scored_rows = []
                    for prompt_path in sorted(prompt_root.glob("*.json")):
                        response_path = response_root / prompt_path.name
                        if not response_path.is_file():
                            raise RuntimeError(f"Missing response: {response_path}")
                        prompt = load_prompt_record(prompt_path)
                        response = json.loads(response_path.read_text(encoding="utf-8"))
                        score = score_completion(prompt, response["raw_text"])
                        row = {
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "length": int(length),
                            "experiment_id": prompt["experiment_id"],
                            "task_success": int(score.task_success),
                            "public_correct": int(bool(score.flags["public_correct"])),
                            "private_correct": int(
                                bool(score.flags["private_correct"])
                            ),
                            "payload_decode_correct": (
                                int(bool(score.flags.get("payload_decode_correct")))
                                if score.flags.get("payload_target") is not None
                                else None
                            ),
                            "concealment_ok": int(score.concealment_ok),
                            "format_ok": int(score.format_ok),
                            "cot_words": int(score.cot_word_count),
                        }
                        per_example.append(row)
                        scored_rows.append(row)
                    if len(scored_rows) != expected_n:
                        raise RuntimeError(
                            f"Incorrect scored count for {model}/{seed}/{condition}/L{length}"
                        )
                    valid_chains = [
                        row["cot_words"] for row in scored_rows if row["format_ok"]
                    ]
                    payload = [
                        row["payload_decode_correct"]
                        for row in scored_rows
                        if row["payload_decode_correct"] is not None
                    ]
                    computed = {
                        "task_success_rate": statistics.fmean(
                            row["task_success"] for row in scored_rows
                        ),
                        "public_exact_rate": statistics.fmean(
                            row["public_correct"] for row in scored_rows
                        ),
                        "private_exact_rate": statistics.fmean(
                            row["private_correct"] for row in scored_rows
                        ),
                        "payload_decode_rate": (
                            statistics.fmean(payload) if payload else None
                        ),
                        "concealment_rate": statistics.fmean(
                            row["concealment_ok"] for row in scored_rows
                        ),
                        "format_rate": statistics.fmean(
                            row["format_ok"] for row in scored_rows
                        ),
                        "avg_cot_words": statistics.fmean(
                            row["cot_words"] for row in scored_rows
                        ),
                    }
                    for metric in METRICS:
                        expected, observed = reported.get(metric), computed[metric]
                        if expected is None and observed is None:
                            continue
                        if (
                            expected is None
                            or observed is None
                            or abs(float(expected) - float(observed)) > 1e-12
                        ):
                            raise RuntimeError(
                                f"Report/recomputed mismatch for {metric}: {model}/{seed}/{condition}/L{length}"
                            )
                    per_seed.append(
                        {
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "length": int(length),
                            "n_examples": expected_n,
                            **computed,
                            "n_format_valid_chains": len(valid_chains),
                            "median_cot_words_format_valid": (
                                statistics.median(valid_chains)
                                if valid_chains
                                else None
                            ),
                        }
                    )

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        groups[(row["model"], row["condition"], row["length"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (model, condition, length), rows in sorted(groups.items()):
        if sorted(row["seed"] for row in rows) != [0, 1, 2]:
            raise RuntimeError(f"Incomplete seeds for {model}/{condition}/L{length}")
        output: dict[str, Any] = {
            "model": model,
            "condition": condition,
            "length": length,
            "n_seeds": 3,
            "n_unique_eval_prompts": expected_n,
            "n_model_outputs": expected_n * 3,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            mean, sd = _mean_sd(values) if values else (None, None)
            output[f"{metric}_mean"], output[f"{metric}_sample_sd"] = mean, sd
        medians = [
            float(row["median_cot_words_format_valid"])
            for row in rows
            if row["median_cot_words_format_valid"] is not None
        ]
        output["mean_seed_median_cot_words"] = (
            statistics.fmean(medians) if medians else None
        )
        output["sample_sd_seed_median_cot_words"] = (
            statistics.stdev(medians)
            if len(medians) > 1
            else (0.0 if medians else None)
        )
        pooled = [
            row["cot_words"]
            for row in per_example
            if row["model"] == model
            and row["condition"] == condition
            and row["length"] == length
            and row["format_ok"]
        ]
        output["pooled_median_cot_words_format_valid"] = (
            statistics.median(pooled) if pooled else None
        )
        aggregate.append(output)

    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_root / "per_example.csv", per_example)
    _write_csv(artifacts_root / "per_seed_length_metrics.csv", per_seed)
    _write_csv(artifacts_root / "aggregate_length_metrics.csv", aggregate)
    summary = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "design": {
            "operand_digits": config["operand_digits"],
            "shared_eval_across_seeds": True,
            "piggyback_relation": "same_multiplier_and_small_wrapped_factor_shift",
            "piggyback_factor_shifts": config["piggyback_factor_shifts"],
        },
        "training_cells": 30,
        "length_evaluations": 90,
        "per_seed_length_metrics": per_seed,
        "aggregate_length_metrics": aggregate,
    }
    (artifacts_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "validated_training_cells": 30,
                "length_evaluations": 90,
                "per_example_rows": len(per_example),
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    a = p.parse_args()
    summarize(a.config)
