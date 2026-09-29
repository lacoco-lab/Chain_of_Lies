#!/usr/bin/env python3
"""Validate and aggregate all 18 completed fixed-Hard-checkpoint cells."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

TASKS = ("multiplication", "s5", "knowledge")
MODELS = ("qwen", "llama")
SEEDS = (0, 1, 2)
METHODS = ("vanilla", "public_only_cot", "piggyback", "steganography")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(root: Path) -> None:
    per_seed: list[dict[str, str]] = []
    for task in TASKS:
        for model in MODELS:
            for seed in SEEDS:
                per_seed.extend(
                    _read(
                        root / task / model / f"seed_{seed}" / "per_length_metrics.csv"
                    )
                )
    aggregates: list[dict[str, Any]] = []
    keys = sorted(
        {(r["task"], r["model"], r["method"], int(r["length"])) for r in per_seed}
    )
    for task, model, method, length in keys:
        subset = [
            r
            for r in per_seed
            if (r["task"], r["model"], r["method"], int(r["length"]))
            == (task, model, method, length)
        ]
        if sorted(int(r["seed"]) for r in subset) != list(SEEDS):
            raise RuntimeError(
                f"Incomplete seeds for {task}/{model}/{method}/L{length}"
            )
        row: dict[str, Any] = {
            "task": task,
            "model": model,
            "method": method,
            "length": length,
            "n_seeds": 3,
            "n_examples": sum(int(r["n"]) for r in subset),
        }
        for metric in (
            "public_exact_rate",
            "private_exact_rate",
            "task_success_rate",
            "avg_cot_words",
            "avg_generated_tokens_including_answer_block",
        ):
            values = [float(r[metric]) for r in subset]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_sample_sd"] = statistics.stdev(values)
        aggregates.append(row)
    _write(root / "all_per_seed_length_metrics.csv", per_seed)
    _write(root / "aggregate_length_metrics.csv", aggregates)
    print(f"Verified 18 cells; wrote {len(aggregates)} aggregate rows to {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/hard_checkpoint_length_sweep")
    )
    main(parser.parse_args().root)
