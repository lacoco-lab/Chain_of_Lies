#!/usr/bin/env python3
"""Strict completion audit and seed aggregates; run locally after transfer."""

import csv
import json
from pathlib import Path
import statistics
import sys

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
from experiments.scaling_regime.protocols.full_cot.experiment import SOURCE_CONFIG
from experiments.scaling_regime.protocols.filler.experiment import (
    load_config,
    _training_complete,
)


def main():
    config = load_config(SOURCE_CONFIG)
    root = Path("artifacts/full_cot_scaling_v1")
    errors, groups = [], {}
    for task, spec in config["tasks"].items():
        for model in config["models"]:
            for seed in config["seeds"]:
                condition = root / task / model / f"seed_{seed}" / "full_cot"
                adapter = condition / spec["variant"]
                if (
                    not _training_complete(adapter)
                    or not (adapter / "ckpt_final/adapter_model.safetensors").is_file()
                ):
                    errors.append(f"Incomplete adapter: {adapter}")
                if not (condition / "provenance.json").is_file():
                    errors.append(f"Missing provenance: {condition}")
                for length in spec["difficulty_values"]:
                    file = (
                        condition
                        / "per_length_eval/ckpt_final"
                        / f"length_{length}.json"
                    )
                    if not file.is_file():
                        errors.append(f"Missing report: {file}")
                        continue
                    metrics = json.loads(file.read_text())["metrics"]
                    expected = spec["eval_examples_per_difficulty"]
                    responses = (
                        Path("generated_data/full_cot_scaling_v1_eval_responses")
                        / task
                        / model
                        / f"seed_{seed}"
                        / f"length_{length}"
                    )
                    raw_files = list(responses.glob("*.json"))
                    if (
                        len(raw_files) != expected
                        or metrics.get("missing_responses")
                        or metrics.get("num_examples") != expected
                    ):
                        errors.append(f"Incomplete responses: {responses}")
                        continue
                    cap = spec["training"]["evaluation_max_new_tokens"]
                    exhausted = 0
                    for raw in raw_files:
                        response = json.loads(raw.read_text())
                        ids = response.get("generated_token_ids")
                        if not isinstance(ids, list) or not ids:
                            errors.append(f"Missing generated token IDs: {raw}")
                        elif len(ids) >= cap:
                            exhausted += 1
                    metrics["token_cap_hit_rate"] = exhausted / expected
                    groups.setdefault((task, model, str(length)), []).append(metrics)
    if errors:
        raise SystemExit("\n".join(errors))
    rows = []
    for (task, model, length), metrics in groups.items():
        if len(metrics) != 3:
            raise RuntimeError("Expected three seeds")
        row = {
            "task": task,
            "model": model,
            "condition": "full_cot",
            "length": length,
            "n_seeds": 3,
        }
        for key in [
            "private_exact_rate",
            "public_exact_rate",
            "task_success_rate",
            "format_rate",
            "token_cap_hit_rate",
        ]:
            values = [float(m[key]) for m in metrics]
            row[key + "_mean"] = statistics.mean(values)
            row[key + "_sample_sd"] = statistics.stdev(values)
        rows.append(row)
    with (root / "aggregate_per_length_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "PASS: 24 adapters and 192 per-length reports, raw responses and token IDs audited."
    )
    print(
        "Accuracy remains an empirical result; inspect token_cap_hit_rate and format_rate before interpretation."
    )


if __name__ == "__main__":
    main()
