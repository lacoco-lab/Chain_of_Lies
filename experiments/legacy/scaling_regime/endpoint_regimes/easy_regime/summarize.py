#!/usr/bin/env python3
"""Strict three-seed summary for the five Easy paper conditions."""

from __future__ import annotations
import argparse, csv, json, statistics, sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        )
    ),
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.generate_splits import (
    TASKS,
    active_tasks,
    load_config,
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


def summarize(
    config_path: Path,
    root: Path,
    tasks: tuple[str, ...] | None = None,
    conditions: tuple[str, ...] | None = None,
) -> dict:
    config = load_config(config_path)
    tasks = tasks or active_tasks(config)
    conditions = conditions or tuple(config["conditions"])
    rows = []
    if not tasks or set(tasks) - set(TASKS):
        raise ValueError(f"Invalid tasks: {tasks}")
    if not conditions or set(conditions) - set(config["conditions"]):
        raise ValueError(f"Invalid conditions: {conditions}")
    for task in tasks:
        for model in config["models"]:
            for seed in config["seeds"]:
                provenance = (
                    root
                    / task
                    / model
                    / f"seed_{seed}"
                    / "provenance"
                    / "validation_report.json"
                )
                if not provenance.exists() or not json.loads(
                    provenance.read_text()
                ).get("valid"):
                    raise RuntimeError(f"Missing valid provenance: {provenance}")
                for condition in conditions:
                    spec = config["conditions"][condition]
                    name = spec["variant"].format(task=task)
                    path = (
                        root
                        / task
                        / model
                        / f"seed_{seed}"
                        / condition
                        / "per_variant_eval"
                        / config["checkpoint"]
                        / f"{name}.json"
                    )
                    if not path.exists():
                        raise RuntimeError(f"Missing evaluation: {path}")
                    reports = json.loads(path.read_text()).get("reports", [])
                    if len(reports) != 1:
                        raise RuntimeError(f"Expected one report: {path}")
                    rl = reports[0].get("rl", {})
                    if rl.get("num_examples") != config["val_n"] or rl.get(
                        "missing_responses"
                    ):
                        raise RuntimeError(f"Incomplete evaluation: {path}")
                    row = {
                        "task": task,
                        "model": model,
                        "seed": seed,
                        "condition": condition,
                        "variant": name,
                    }
                    row.update({metric: rl.get(metric) for metric in METRICS})
                    rows.append(row)
    aggregate = []
    for task in tasks:
        for model in config["models"]:
            for condition in conditions:
                subset = [
                    r
                    for r in rows
                    if (r["task"], r["model"], r["condition"])
                    == (task, model, condition)
                ]
                if len(subset) != 3:
                    raise RuntimeError(
                        f"Expected three seeds for {task}/{model}/{condition}"
                    )
                out = {
                    "task": task,
                    "model": model,
                    "condition": condition,
                    "n_seeds": 3,
                }
                for metric in METRICS:
                    vals = [
                        float(r[metric]) for r in subset if r.get(metric) is not None
                    ]
                    out[f"{metric}_mean"] = statistics.fmean(vals) if vals else None
                    out[f"{metric}_sample_sd"] = (
                        statistics.stdev(vals) if len(vals) > 1 else None
                    )
                aggregate.append(out)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": config["experiment_name"],
        "tasks": list(tasks),
        "conditions": list(conditions),
        "expected_cells": len(tasks)
        * len(config["models"])
        * len(config["seeds"])
        * len(conditions),
        "per_seed_metrics": rows,
        "aggregate_metrics": aggregate,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename, data in (
        ("per_seed_metrics.csv", rows),
        ("aggregate_metrics.csv", aggregate),
    ):
        with (root / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts/easy_regime"))
    p.add_argument("--tasks", nargs="+", choices=TASKS)
    p.add_argument("--conditions", nargs="+")
    a = p.parse_args()
    print(
        json.dumps(
            {
                "cells": len(
                    summarize(
                        a.config,
                        a.artifacts_root,
                        tuple(a.tasks) if a.tasks else None,
                        tuple(a.conditions) if a.conditions else None,
                    )["per_seed_metrics"]
                )
            },
            indent=2,
        )
    )
