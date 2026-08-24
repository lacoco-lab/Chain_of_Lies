#!/usr/bin/env python3
"""Fail-closed three-seed summaries for final Hard-regime steganography."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _variants(config: dict[str, Any], family: str) -> list[str]:
    return [str(config[family]["variant"])]


def _load_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_n = int(config["val_n"])
    for model in config["models"]:
        for seed in config["seeds"]:
            provenance = root / model / f"seed_{seed}" / "provenance" / "validation_report.json"
            if not provenance.exists() or not json.loads(provenance.read_text(encoding="utf-8")).get("valid"):
                raise RuntimeError(f"Missing valid split provenance: {provenance}")
            for family in ("steganography",):
                for mode in config[family]["modes"]:
                    for variant in _variants(config, family):
                        path = (
                            root
                            / model
                            / f"seed_{seed}"
                            / family
                            / mode
                            / "per_variant_eval"
                            / config["checkpoint"]
                            / f"{variant}.json"
                        )
                        if not path.exists():
                            raise RuntimeError(f"Missing evaluation report: {path}")
                        reports = json.loads(path.read_text(encoding="utf-8")).get("reports", [])
                        if len(reports) != 1:
                            raise RuntimeError(f"Expected one report in {path}, found {len(reports)}")
                        trained = reports[0].get("rl", {})
                        baseline = reports[0].get("baseline", {})
                        if trained.get("num_examples") != expected_n or trained.get("missing_responses"):
                            raise RuntimeError(f"Incomplete trained evaluation in {path}")
                        if baseline.get("num_examples") != expected_n or baseline.get("missing_responses"):
                            raise RuntimeError(f"Incomplete baseline evaluation in {path}")
                        row: dict[str, Any] = {
                            "model": model,
                            "seed": seed,
                            "family": family,
                            "mode": mode,
                            "variant": variant,
                        }
                        for metric in METRICS:
                            row[metric] = trained.get(metric)
                            row[f"baseline_{metric}"] = baseline.get(metric)
                        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({(r["model"], r["family"], r["mode"], r["variant"]) for r in rows})
    for model, family, mode, variant in groups:
        subset = [
            row
            for row in rows
            if (row["model"], row["family"], row["mode"], row["variant"])
            == (model, family, mode, variant)
        ]
        if len(subset) != 3:
            raise RuntimeError(f"Expected three seeds for {model}/{family}/{mode}/{variant}")
        result: dict[str, Any] = {
            "model": model,
            "family": family,
            "mode": mode,
            "variant": variant,
            "n_seeds": 3,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(result)
    return output


def _contrasts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["model"], row["seed"], row["family"], row["mode"], row["variant"]): row
        for row in rows
    }
    per_seed: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows}):
            steg = lookup[(model, seed, "steganography", "local_channel_cot", "arith_steg_local_invisible")]
            steg_answer = lookup[(model, seed, "steganography", "answer_only", "arith_steg_local_invisible")]
            per_seed.append({
                "model": model,
                "seed": seed,
                "family": "steganography",
                "channel_private_exact": steg["private_exact_rate"],
                "answer_only_private_exact": steg_answer["private_exact_rate"],
                "channel_minus_answer_only": steg["private_exact_rate"] - steg_answer["private_exact_rate"],
            })

    aggregate: list[dict[str, Any]] = []
    for model, family in sorted({(r["model"], r["family"]) for r in per_seed}):
        subset = [r for r in per_seed if r["model"] == model and r["family"] == family]
        result: dict[str, Any] = {"model": model, "family": family, "n_seeds": len(subset)}
        for metric in (
            "channel_private_exact",
            "answer_only_private_exact",
            "channel_minus_answer_only",
        ):
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(result)
    return per_seed, aggregate


def summarize(config_path: Path, artifacts_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = _load_rows(config, artifacts_root)
    aggregate = _aggregate(rows)
    contrasts, aggregate_contrasts = _contrasts(rows)
    summary = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "operand_range": config["operand_range"],
        "models": config["models"],
        "seeds": config["seeds"],
        "checkpoint": config["checkpoint"],
        "per_seed_metrics": rows,
        "aggregate_metrics": aggregate,
        "per_seed_contrasts": contrasts,
        "aggregate_contrasts": aggregate_contrasts,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
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
    print(json.dumps({key: summary[key] for key in ("experiment", "operand_range", "models", "seeds")}, indent=2))


if __name__ == "__main__":
    main()
