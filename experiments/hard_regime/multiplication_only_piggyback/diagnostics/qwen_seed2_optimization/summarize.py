#!/usr/bin/env python3
"""Summarize fixed-data Qwen optimization-seed sensitivity."""

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


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing evaluation report: {path}")
    reports = json.loads(path.read_text(encoding="utf-8")).get("reports", [])
    if len(reports) != 1:
        raise RuntimeError(f"Expected one report in {path}, found {len(reports)}")
    trained = reports[0].get("rl", {})
    if trained.get("num_examples") != 1000 or trained.get("missing_responses"):
        raise RuntimeError(f"Incomplete evaluation in {path}")
    return trained


def summarize(
    config_path: Path, diagnostic_root: Path, original_root: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_seed = int(config["original_optimization_seed"])
    rows: list[dict[str, Any]] = []
    original_path = (
        original_root / "qwen" / "seed_2" / "public_cot" / "per_variant_eval"
        / "ckpt_final" / "arith_piggyback.json"
    )
    original = _load_report(original_path)
    rows.append({
        "data_seed": config["data_seed"],
        "optimization_seed": original_seed,
        "source": "original",
        **{metric: original.get(metric) for metric in METRICS},
    })

    for optimization_seed in config["additional_optimization_seeds"]:
        provenance = diagnostic_root / f"opt_seed_{optimization_seed}" / "provenance"
        validation = provenance / "validation_report.json"
        cell = provenance / "diagnostic_cell.json"
        if not validation.exists() or not json.loads(validation.read_text(encoding="utf-8")).get("valid"):
            raise RuntimeError(f"Missing valid provenance: {validation}")
        cell_data = json.loads(cell.read_text(encoding="utf-8")) if cell.exists() else {}
        if cell_data.get("data_seed") != config["data_seed"] or cell_data.get("optimization_seed") != optimization_seed:
            raise RuntimeError(f"Diagnostic cell provenance mismatch: {cell}")
        path = (
            diagnostic_root / f"opt_seed_{optimization_seed}" / "public_cot"
            / "per_variant_eval" / "ckpt_final" / "arith_piggyback.json"
        )
        metrics = _load_report(path)
        rows.append({
            "data_seed": config["data_seed"],
            "optimization_seed": optimization_seed,
            "source": "additional",
            **{metric: metrics.get(metric) for metric in METRICS},
        })

    def aggregate(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
        result: dict[str, Any] = {"group": label, "n_optimization_seeds": len(subset)}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
            result[f"{metric}_min"] = min(values) if values else None
            result[f"{metric}_max"] = max(values) if values else None
        return result

    additional = [row for row in rows if row["source"] == "additional"]
    aggregates = [aggregate(additional, "additional_only"), aggregate(rows, "original_plus_additional")]
    output = {
        "schema_version": 1,
        "diagnostic": config["diagnostic_name"],
        "purpose": "Separate optimization sensitivity from fixed seed-2 dataset difficulty; do not select the best seed.",
        "data_seed": config["data_seed"],
        "per_optimization_seed": rows,
        "aggregates": aggregates,
        "original_private_exact": original["private_exact_rate"],
        "additional_private_exact_values": [row["private_exact_rate"] for row in additional],
    }
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    (diagnostic_root / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = list(rows[0])
    with (diagnostic_root / "per_optimization_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostic-artifacts-root", type=Path, required=True)
    parser.add_argument("--original-artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.config, args.diagnostic_artifacts_root, args.original_artifacts_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
