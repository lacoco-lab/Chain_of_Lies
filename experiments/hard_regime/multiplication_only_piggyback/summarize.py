#!/usr/bin/env python3
"""Fail-closed summary and affine-reference comparison for the replacement run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "task_success_rate", "public_exact_rate", "private_exact_rate",
    "concealment_rate", "format_rate", "avg_cot_words",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_n = int(config["val_n"])
    for model in config["models"]:
        for seed in config["seeds"]:
            provenance = root / model / f"seed_{seed}" / "provenance" / "validation_report.json"
            if not provenance.exists() or not json.loads(provenance.read_text(encoding="utf-8")).get("valid"):
                raise RuntimeError(f"Missing valid split provenance: {provenance}")
            for mode in config["piggyback"]["modes"]:
                for variant in config["piggyback"]["variants"]:
                    path = root / model / f"seed_{seed}" / mode / "per_variant_eval" / config["checkpoint"] / f"{variant}.json"
                    if not path.exists():
                        raise RuntimeError(f"Missing evaluation report: {path}")
                    reports = json.loads(path.read_text(encoding="utf-8")).get("reports", [])
                    if len(reports) != 1:
                        raise RuntimeError(f"Expected one report in {path}, found {len(reports)}")
                    trained = reports[0].get("rl", {})
                    if trained.get("num_examples") != expected_n or trained.get("missing_responses"):
                        raise RuntimeError(f"Incomplete trained evaluation in {path}")
                    row: dict[str, Any] = {
                        "model": model, "seed": seed, "mode": mode, "variant": variant,
                    }
                    for metric in METRICS:
                        row[metric] = trained.get(metric)
                    rows.append(row)
    return rows


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({(row["model"], row["mode"], row["variant"]) for row in rows})
    for model, mode, variant in groups:
        subset = [row for row in rows if (row["model"], row["mode"], row["variant"]) == (model, mode, variant)]
        if len(subset) != 3:
            raise RuntimeError(f"Expected three seeds for {model}/{mode}/{variant}")
        result: dict[str, Any] = {"model": model, "mode": mode, "variant": variant, "n_seeds": 3}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(result)
    return output


def _contrasts(config: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["model"], row["seed"], row["mode"], row["variant"]): row for row in rows}
    per_seed: list[dict[str, Any]] = []
    for model in config["models"]:
        for seed in config["seeds"]:
            pig = lookup[(model, seed, "public_cot", "arith_piggyback")]
            answer = lookup[(model, seed, "answer_only", "arith_piggyback")]
            control = lookup[(model, seed, "public_cot", "arith_piggyback_control")]
            per_seed.append({
                "model": model,
                "seed": seed,
                "piggyback_public_exact": pig["public_exact_rate"],
                "piggyback_private_exact": pig["private_exact_rate"],
                "answer_only_private_exact": answer["private_exact_rate"],
                "control_private_exact": control["private_exact_rate"],
                "piggyback_minus_answer_only": pig["private_exact_rate"] - answer["private_exact_rate"],
                "piggyback_minus_control": pig["private_exact_rate"] - control["private_exact_rate"],
            })
    aggregate: list[dict[str, Any]] = []
    for model in config["models"]:
        subset = [row for row in per_seed if row["model"] == model]
        result: dict[str, Any] = {"model": model, "n_seeds": 3}
        for metric in (
            "piggyback_public_exact", "piggyback_private_exact", "answer_only_private_exact",
            "control_private_exact", "piggyback_minus_answer_only", "piggyback_minus_control",
        ):
            mean, sd = _mean_sd([float(row[metric]) for row in subset])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sample_sd"] = sd
        reference = config["affine_reference"][model]
        result["private_minus_affine_reference"] = result["piggyback_private_exact_mean"] - reference["piggyback_private_exact"]
        result["public_minus_affine_reference"] = result["piggyback_public_exact_mean"] - reference["piggyback_public_exact"]
        aggregate.append(result)
    return per_seed, aggregate


def summarize(config_path: Path, artifacts_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = _load_rows(config, artifacts_root)
    aggregate = _aggregate(rows)
    contrasts, aggregate_contrasts = _contrasts(config, rows)
    summary = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "question_format": "multiplication_only",
        "operand_range": config["operand_range"],
        "models": config["models"],
        "seeds": config["seeds"],
        "affine_reference": config["affine_reference"],
        "per_seed_metrics": rows,
        "aggregate_metrics": aggregate,
        "per_seed_contrasts": contrasts,
        "aggregate_contrasts": aggregate_contrasts,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(artifacts_root / "per_seed_metrics.csv", rows)
    _write_csv(artifacts_root / "aggregate_metrics.csv", aggregate)
    _write_csv(artifacts_root / "per_seed_contrasts.csv", contrasts)
    _write_csv(artifacts_root / "aggregate_contrasts.csv", aggregate_contrasts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.config, args.artifacts_root)
    print(json.dumps({key: summary[key] for key in ("experiment", "question_format", "models", "seeds")}, indent=2))


if __name__ == "__main__":
    main()
