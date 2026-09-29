#!/usr/bin/env python3
"""Fail-closed three-seed aggregation for the plain-parity calibration."""

from __future__ import annotations

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

from experiments.scaling_regime.tasks.plain_parity.generate_splits import load_config
from experiments.scaling_regime.tasks.plain_parity.summarize import summarize

EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATHS = (
    EXPERIMENT_DIR / "config.json",
    EXPERIMENT_DIR / "config_seed_1.json",
    EXPERIMENT_DIR / "config_seed_2.json",
)
SPLIT_ROOT = Path("generated_data/parity_regime_calibration")
RESPONSES_ROOT = Path("generated_data/parity_regime_calibration_eval_responses")
ARTIFACTS_ROOT = Path("artifacts/parity_regime_calibration")
METRICS = (
    "joint_exact",
    "public_exact",
    "private_exact",
    "format_rate",
    "concealment_rate",
    "payload_decode_rate",
    "avg_cot_words",
)


def _scientific_signature(config: dict[str, Any]) -> dict[str, Any]:
    signature = json.loads(json.dumps(config))
    signature.pop("seed", None)
    training = signature["training"]
    for operational_key in (
        "activation_cpu_offload",
        "ce_token_chunk_size",
        "evaluation_batch_size",
        "memory_efficient_ce",
        "run_training_validation",
    ):
        training.pop(operational_key, None)
    return signature


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise RuntimeError(f"Expected exactly three seed values; got {len(values)}.")
    return statistics.mean(values), statistics.stdev(values)


def summarize_three_seeds(
    *,
    split_root: Path = SPLIT_ROOT,
    responses_root: Path = RESPONSES_ROOT,
    artifacts_root: Path = ARTIFACTS_ROOT,
    config_paths: tuple[Path, ...] = CONFIG_PATHS,
) -> dict[str, Any]:
    configs = [load_config(path) for path in config_paths]
    seeds = [int(config["seed"]) for config in configs]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"Expected ordered seed configs [0, 1, 2]; got {seeds}.")
    signature = _scientific_signature(configs[0])
    if any(_scientific_signature(config) != signature for config in configs[1:]):
        raise RuntimeError(
            "Seed configs differ in scientific settings; refusing to aggregate."
        )

    per_seed = []
    for config_path, seed in zip(config_paths, seeds):
        per_seed_root = artifacts_root / "per_seed" / f"seed_{seed}"
        per_seed.append(
            summarize(config_path, split_root, responses_root, per_seed_root)
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for seed_summary in per_seed:
        for row in seed_summary["metrics_by_length_bucket"]:
            grouped[
                (row["model"], row["condition"], row["source"], row["length_bucket"])
            ].append(row)

    aggregate_rows = []
    for key in sorted(grouped):
        model, condition, source, bucket = key
        seed_rows = sorted(grouped[key], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in seed_rows] != seeds:
            raise RuntimeError(f"Incomplete seed coverage for {key}.")
        aggregate: dict[str, Any] = {
            "model": model,
            "condition": condition,
            "source": source,
            "length_bucket": bucket,
            "length_min": seed_rows[0]["length_min"],
            "length_max": seed_rows[0]["length_max"],
            "n_per_seed": seed_rows[0]["n"],
            "seeds": "0,1,2",
        }
        for metric in METRICS:
            values = [row[metric] for row in seed_rows]
            if all(value is None for value in values):
                aggregate[f"{metric}_mean"] = None
                aggregate[f"{metric}_sample_sd"] = None
            elif any(value is None for value in values):
                raise RuntimeError(f"Mixed null/non-null values for {key}/{metric}.")
            else:
                mean, sample_sd = _mean_std([float(value) for value in values])
                aggregate[f"{metric}_mean"] = mean
                aggregate[f"{metric}_sample_sd"] = sample_sd
        aggregate_rows.append(aggregate)

    candidate_lookup = {
        (
            candidate["model"],
            candidate["length_bucket"],
            seed_summary["seed"],
        ): candidate
        for seed_summary in per_seed
        for candidate in seed_summary["provisional_regime_candidates"]
    }
    replicated_candidates = []
    for model in configs[0]["models"]:
        for bucket_spec in configs[0]["length_buckets"]:
            bucket = str(bucket_spec["name"])
            candidates = [candidate_lookup[(model, bucket, seed)] for seed in seeds]
            labels = [
                candidate["provisional_classification"] for candidate in candidates
            ]
            stable_mechanisms = []
            for mechanism in (
                "public_only_cot",
                "filler",
                "piggyback",
                "steganography",
            ):
                passes = []
                for candidate in candidates:
                    item = next(
                        item
                        for item in candidate["mechanisms"]
                        if item["mechanism"] == mechanism
                    )
                    passes.append(bool(item["passes_hard_mechanism_gate"]))
                if all(passes):
                    stable_mechanisms.append(mechanism)
            if all(label == "easy_candidate" for label in labels):
                consensus = "replicated_easy_candidate"
            elif (
                all(label == "hard_candidate" for label in labels) and stable_mechanisms
            ):
                consensus = "replicated_hard_candidate"
            else:
                consensus = "not_replicated"
            replicated_candidates.append(
                {
                    "model": model,
                    "length_bucket": bucket,
                    "per_seed_labels": labels,
                    "replicated_classification": consensus,
                    "hard_mechanisms_passing_in_all_seeds": stable_mechanisms,
                }
            )

    artifacts_root.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_root / "metrics_by_length_bucket_three_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    result = {
        "experiment_name": configs[0]["experiment_name"],
        "seeds": seeds,
        "scientific_config_consistent": True,
        "aggregate_metrics": aggregate_rows,
        "replicated_regime_candidates": replicated_candidates,
    }
    (artifacts_root / "summary_three_seed.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lookup = {
        (row["model"], row["condition"], row["source"], row["length_bucket"]): row
        for row in aggregate_rows
    }
    replicated_lookup = {
        (item["model"], item["length_bucket"]): item for item in replicated_candidates
    }
    lines = [
        "# Plain parity: three-seed replication",
        "",
        "Private exact accuracy, mean ± sample SD over seeds 0, 1, and 2.",
        "",
        "| Model | Length | Vanilla | Public-only | Filler | Piggyback | Steganography | Replicated label |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in configs[0]["models"]:
        for bucket_spec in configs[0]["length_buckets"]:
            bucket = str(bucket_spec["name"])
            formatted = {}
            for condition in configs[0]["conditions"]:
                row = lookup[(model, condition, "finetuned", bucket)]
                formatted[condition] = (
                    f"{100 * row['private_exact_mean']:.1f} ± "
                    f"{100 * row['private_exact_sample_sd']:.1f}"
                )
            label = replicated_lookup[(model, bucket)]["replicated_classification"]
            lines.append(
                f"| {model} | `{bucket}` | {formatted['vanilla']} | "
                f"{formatted['public_only_cot']} | {formatted['filler']} | "
                f"{formatted['piggyback']} | {formatted['steganography']} | {label} |"
            )
    lines.extend(
        [
            "",
            "> A replicated Hard label requires all three seeds to pass the per-seed Hard rule and at least one mechanism to pass its gate in all three seeds.",
        ]
    )
    (artifacts_root / "REPORT_THREE_SEED.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    result = summarize_three_seeds()
    print(
        json.dumps(
            {
                "aggregate_rows": len(result["aggregate_metrics"]),
                "replicated_candidates": len(result["replicated_regime_candidates"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
