#!/usr/bin/env python3
"""Fail closed unless every Filler-only training and evaluation cell is complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.scaling_regime.protocols.filler.experiment import (
    _adapter_root,
    _condition_root,
    _task,
    _training_complete,
    load_config,
)


def audit(config_path: Path, selected_tasks: list[str] | None = None) -> dict:
    config = load_config(config_path)
    tasks = selected_tasks or list(config["tasks"])
    cells = []
    for task_name in tasks:
        task_spec = _task(config, task_name)
        variant = str(task_spec["variant"])
        expected_eval = int(task_spec["eval_examples_per_difficulty"])
        for model in config["models"]:
            for seed in config["seeds"]:
                adapter_root = _adapter_root(task_spec, model, int(seed))
                if not _training_complete(adapter_root):
                    raise RuntimeError(
                        f"Incomplete training cell: {task_name}/{model}/seed_{seed}"
                    )
                metadata = json.loads(
                    (adapter_root / "training_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                if metadata.get("supervision_mode") != "filler_only":
                    raise RuntimeError(f"Wrong supervision mode in {adapter_root}")
                if (
                    metadata.get("filler_token_counts")
                    != task_spec["filler_token_counts"]
                ):
                    raise RuntimeError(f"Wrong filler budgets in {adapter_root}")

                condition_root = _condition_root(task_spec, model, int(seed))
                responses_root = (
                    Path(task_spec["responses_root"])
                    / model
                    / f"seed_{seed}"
                    / "filler_only"
                )
                reports = []
                if task_spec["evaluation_kind"] == "parity_validation":
                    report_path = (
                        adapter_root
                        / "per_variant_eval"
                        / "ckpt_final"
                        / f"{variant}.json"
                    )
                    reports.append(report_path)
                    response_dirs = [
                        responses_root / "ckpt_final" / "finetuned" / variant
                    ]
                    expected_counts = [
                        expected_eval * len(task_spec["difficulty_values"])
                    ]
                else:
                    report_paths = [
                        condition_root
                        / "per_length_eval"
                        / "ckpt_final"
                        / f"length_{value}.json"
                        for value in task_spec["difficulty_values"]
                    ]
                    reports.extend(report_paths)
                    response_dirs = [
                        responses_root
                        / f"length_{value}"
                        / "ckpt_final"
                        / "trained"
                        / variant
                        for value in task_spec["difficulty_values"]
                    ]
                    expected_counts = [expected_eval] * len(response_dirs)
                for report_path in reports:
                    if not report_path.is_file():
                        raise RuntimeError(f"Missing evaluation report: {report_path}")
                total_responses = 0
                for response_dir, expected_count in zip(response_dirs, expected_counts):
                    count = len(list(response_dir.glob("*.json")))
                    if count != expected_count:
                        raise RuntimeError(
                            f"Expected {expected_count} responses in {response_dir}, found {count}."
                        )
                    total_responses += count
                cells.append(
                    {
                        "task": task_name,
                        "model": model,
                        "seed": int(seed),
                        "checkpoint": str(adapter_root / "ckpt_final"),
                        "evaluation_reports": len(reports),
                        "responses": total_responses,
                    }
                )
    return {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "complete": True,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=("multiplication", "knowledge", "s5", "parity"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.task)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"complete": True, "validated_cells": len(result["cells"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
