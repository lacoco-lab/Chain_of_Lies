#!/usr/bin/env python3
"""Audit and index the complete three-seed plain-parity experiment package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.scaling_regime.tasks.plain_parity.generate_splits import load_config
from experiments.scaling_regime.tasks.plain_parity.summarize_three_seeds import (
    _scientific_signature,
)
from experiments.scaling_regime.tasks.plain_parity.validate_splits import validate

EXPERIMENT_ROOT = Path("experiments/scaling_regime/tasks/plain_parity")
SPLIT_ROOT = Path("generated_data/parity_regime_calibration")
RESPONSES_ROOT = Path("generated_data/parity_regime_calibration_eval_responses")
ARTIFACTS_ROOT = Path("artifacts/parity_regime_calibration")
CONFIG_PATHS = (
    EXPERIMENT_ROOT / "config.json",
    EXPERIMENT_ROOT / "config_seed_1.json",
    EXPERIMENT_ROOT / "config_seed_2.json",
)
METRICS = (
    "joint_exact",
    "public_exact",
    "private_exact",
    "format_rate",
    "concealment_rate",
    "avg_cot_words",
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def audit() -> dict[str, Any]:
    configs = [load_config(path) for path in CONFIG_PATHS]
    seeds = [int(config["seed"]) for config in configs]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"Expected seed configs [0, 1, 2], got {seeds}.")
    signature = _scientific_signature(configs[0])
    if any(_scientific_signature(config) != signature for config in configs[1:]):
        raise RuntimeError("Seed configs differ in scientific settings.")

    validations = [validate(path, SPLIT_ROOT) for path in CONFIG_PATHS]
    if not all(item.get("valid") for item in validations):
        raise RuntimeError("At least one generated split failed validation.")

    cells = []
    adapter_files = []
    total_responses = 0
    for config in configs:
        seed = int(config["seed"])
        expected_train = int(config["train_per_bucket"]) * len(config["length_buckets"])
        expected_val = int(config["validation_per_bucket"]) * len(
            config["length_buckets"]
        )
        for model, model_id in config["models"].items():
            for condition, condition_spec in config["conditions"].items():
                variant = str(condition_spec["variant"])
                condition_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / condition
                variant_root = condition_root / variant
                checkpoint = variant_root / "ckpt_final"
                adapter = _require(checkpoint / "adapter_model.safetensors")
                adapter_config_path = _require(checkpoint / "adapter_config.json")
                metadata_path = _require(checkpoint / "training_metadata.json")
                _require(checkpoint / "train_history.json")
                _require(condition_root / "provenance/config.json")
                _require(condition_root / "provenance/split_manifest.json")
                validation_path = _require(
                    condition_root / "provenance/validation_report.json"
                )

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                adapter_config = json.loads(
                    adapter_config_path.read_text(encoding="utf-8")
                )
                validation_report = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                if (
                    metadata.get("base_model") != model_id
                    or int(metadata.get("steps", -1)) != expected_train
                    or int(metadata.get("train_examples_seen", -1)) != expected_train
                    or metadata.get("supervision_mode")
                    != condition_spec["supervision_mode"]
                ):
                    raise RuntimeError(
                        f"Invalid training metadata for seed={seed}/{model}/{condition}."
                    )
                if (
                    adapter_config.get("base_model_name_or_path") != model_id
                    or int(adapter_config.get("r", -1))
                    != int(config["training"]["lora_r"])
                    or int(adapter_config.get("lora_alpha", -1))
                    != int(config["training"]["lora_alpha"])
                    or not validation_report.get("valid")
                ):
                    raise RuntimeError(
                        f"Invalid adapter/provenance for seed={seed}/{model}/{condition}."
                    )

                prompt_dir = SPLIT_ROOT / f"seed_{seed}" / variant / "val_prompts"
                prompt_ids = {path.stem for path in prompt_dir.glob("*.json")}
                if len(prompt_ids) != expected_val:
                    raise RuntimeError(
                        f"Wrong validation prompt count for seed={seed}/{variant}."
                    )
                source_counts = {}
                for source in ("baseline", "finetuned"):
                    response_dir = (
                        RESPONSES_ROOT
                        / model
                        / f"seed_{seed}"
                        / condition
                        / "ckpt_final"
                        / source
                        / variant
                    )
                    response_ids = {path.stem for path in response_dir.glob("*.json")}
                    if response_ids != prompt_ids:
                        raise RuntimeError(
                            f"Response/prompt ID mismatch for seed={seed}/{model}/{condition}/{source}."
                        )
                    source_counts[source] = len(response_ids)
                    total_responses += len(response_ids)

                adapter_files.append(adapter)
                cells.append(
                    {
                        "seed": seed,
                        "model": model,
                        "condition": condition,
                        "variant": variant,
                        "train_examples_seen": expected_train,
                        "responses": source_counts,
                        "final_adapter": str(adapter),
                    }
                )

    per_seed_rows = []
    for seed in seeds:
        path = _require(
            ARTIFACTS_ROOT / f"per_seed/seed_{seed}/metrics_by_length_bucket.csv"
        )
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if len(rows) != 100 or any(
            int(row["seed"]) != seed or int(row["n"]) != 200 for row in rows
        ):
            raise RuntimeError(f"Invalid per-seed metrics for seed {seed}.")
        per_seed_rows.extend(rows)

    aggregate_path = _require(
        ARTIFACTS_ROOT / "metrics_by_length_bucket_three_seed.csv"
    )
    aggregate_rows = list(csv.DictReader(aggregate_path.open(encoding="utf-8")))
    if len(aggregate_rows) != 100:
        raise RuntimeError(f"Expected 100 aggregate rows; found {len(aggregate_rows)}.")
    lookup = {
        (
            row["model"],
            row["condition"],
            row["source"],
            row["length_bucket"],
            int(row["seed"]),
        ): row
        for row in per_seed_rows
    }
    for row in aggregate_rows:
        key = (row["model"], row["condition"], row["source"], row["length_bucket"])
        for metric in METRICS:
            values = [float(lookup[key + (seed,)][metric]) for seed in seeds]
            if not math.isclose(
                float(row[f"{metric}_mean"]), statistics.mean(values), abs_tol=1e-12
            ):
                raise RuntimeError(f"Aggregate mean mismatch for {key}/{metric}.")
            if not math.isclose(
                float(row[f"{metric}_sample_sd"]),
                statistics.stdev(values),
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"Aggregate SD mismatch for {key}/{metric}.")

    summary_path = _require(ARTIFACTS_ROOT / "summary_three_seed.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = summary.get("replicated_regime_candidates") or []
    if (
        summary.get("seeds") != seeds
        or not summary.get("scientific_config_consistent")
        or len(candidates) != 10
        or any(
            item.get("replicated_classification") != "replicated_hard_candidate"
            for item in candidates
        )
    ):
        raise RuntimeError("Three-seed summary is incomplete or inconsistent.")

    prompt_count = sum(
        1
        for path in SPLIT_ROOT.rglob("*.json")
        if any(part.endswith("_prompts") for part in path.parts)
    )
    if prompt_count != 99_000 or total_responses != 60_000 or len(cells) != 30:
        raise RuntimeError(
            f"Package count mismatch: prompts={prompt_count}, responses={total_responses}, cells={len(cells)}."
        )

    key_files = [
        *CONFIG_PATHS,
        SPLIT_ROOT / "split_manifest.json",
        SPLIT_ROOT / "split_manifest_seed_1.json",
        SPLIT_ROOT / "split_manifest_seed_2.json",
        ARTIFACTS_ROOT / "REPORT_THREE_SEED.md",
        aggregate_path,
        summary_path,
        *adapter_files,
    ]
    return {
        "schema_version": 1,
        "experiment_name": configs[0]["experiment_name"],
        "package_complete": True,
        "seeds": seeds,
        "models": list(configs[0]["models"]),
        "conditions": list(configs[0]["conditions"]),
        "canonical_roots": {
            "implementation": str(EXPERIMENT_ROOT),
            "generated_splits": str(SPLIT_ROOT),
            "raw_responses": str(RESPONSES_ROOT),
            "adapters_provenance_and_reports": str(ARTIFACTS_ROOT),
        },
        "counts": {
            "cells": len(cells),
            "final_adapters": len(adapter_files),
            "generated_prompt_records": prompt_count,
            "raw_response_records": total_responses,
            "per_seed_metric_rows": len(per_seed_rows),
            "aggregate_metric_rows": len(aggregate_rows),
            "replicated_regime_candidates": len(candidates),
        },
        "scientific_config_consistent": True,
        "all_splits_valid_and_disjoint": True,
        "all_response_ids_match_validation_prompts": True,
        "all_candidates_replicated_hard": True,
        "split_validations": validations,
        "cells": cells,
        "key_file_checksums": [_file_record(_require(path)) for path in key_files],
    }


def main() -> None:
    result = audit()
    output = ARTIFACTS_ROOT / "PACKAGE_MANIFEST.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
