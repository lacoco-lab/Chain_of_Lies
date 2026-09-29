#!/usr/bin/env python3
"""Audit the final mixed-source Easy package before archival or deletion."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.generate_splits import (
    TASKS,
    load_config,
)

CONFIG_PATH = (
    REPO_ROOT
    / "experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/config.json"
)
CANONICAL_ARTIFACTS = REPO_ROOT / "artifacts/easy_regime"
CANONICAL_RESPONSES = REPO_ROOT / "generated_data/easy_regime_eval_responses"
OUTPUT_PATH = CANONICAL_ARTIFACTS / "final_package_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root(config: dict[str, Any], task: str, model: str) -> Path:
    override = config.get("result_overrides", {}).get(f"{task}/{model}")
    return (
        REPO_ROOT / override["data_root"]
        if override
        else REPO_ROOT / "generated_data/easy_regime"
    )


def main() -> None:
    config = load_config(CONFIG_PATH)
    cells = []
    for task in TASKS:
        for model in config["models"]:
            for seed in config["seeds"]:
                for condition, spec in config["conditions"].items():
                    variant = spec["variant"].format(task=task)
                    prompts = (
                        _data_root(config, task, model)
                        / task
                        / f"seed_{seed}"
                        / variant
                        / "val_prompts"
                    )
                    cell = (
                        CANONICAL_ARTIFACTS / task / model / f"seed_{seed}" / condition
                    )
                    report_path = (
                        cell / "per_variant_eval/ckpt_final" / f"{variant}.json"
                    )
                    metadata_path = cell / variant / "ckpt_final/training_metadata.json"
                    responses = (
                        CANONICAL_RESPONSES
                        / task
                        / model
                        / f"seed_{seed}"
                        / condition
                        / "ckpt_final/trained"
                        / variant
                    )
                    if len(list(prompts.glob("*.json"))) != 1_000:
                        raise RuntimeError(f"Incomplete prompts: {prompts}")
                    if len(list(responses.glob("*.json"))) != 1_000:
                        raise RuntimeError(f"Incomplete responses: {responses}")
                    if not report_path.exists() or not metadata_path.exists():
                        raise RuntimeError(f"Missing artifact in {cell}")
                    report = json.loads(report_path.read_text(encoding="utf-8"))[
                        "reports"
                    ][0]
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if (
                        report["rl"]["num_examples"] != 1_000
                        or report["rl"]["missing_responses"]
                    ):
                        raise RuntimeError(f"Incomplete evaluation: {report_path}")
                    if (
                        metadata.get("steps") != 10_000
                        or metadata.get("train_examples_seen") != 10_000
                        or metadata.get("target_train_coverage") != 1.0
                    ):
                        raise RuntimeError(f"Incomplete training: {metadata_path}")
                    adapter = cell / variant / "ckpt_final/adapter_model.safetensors"
                    cells.append(
                        {
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "variant": variant,
                            "prompts": 1_000,
                            "trained_responses": 1_000,
                            "evaluation_complete": True,
                            "training_metadata_complete": True,
                            "final_adapter_present": adapter.exists(),
                        }
                    )
    summaries = [
        CANONICAL_ARTIFACTS / "summary.json",
        CANONICAL_ARTIFACTS / "per_seed_metrics.csv",
        CANONICAL_ARTIFACTS / "aggregate_metrics.csv",
        CANONICAL_ARTIFACTS / "analysis/analysis_manifest.json",
        REPO_ROOT / "docs/legacy_regimes/EASY_REGIME_V2_REPORT.md",
        REPO_ROOT / "docs/FIGURE_DATA_PRESERVATION.md",
    ]
    missing = [
        str(path.relative_to(REPO_ROOT)) for path in summaries if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"Missing final summaries: {missing}")
    adapters = sum(int(cell["final_adapter_present"]) for cell in cells)
    manifest = {
        "schema_version": 1,
        "valid_for_paper": True,
        "exact_checkpoint_archive_complete": adapters == len(cells),
        "cells": len(cells),
        "validation_prompts": sum(cell["prompts"] for cell in cells),
        "trained_responses": sum(cell["trained_responses"] for cell in cells),
        "complete_evaluation_reports": len(cells),
        "complete_training_metadata": len(cells),
        "final_adapters_present": adapters,
        "final_adapters_expected": len(cells),
        "summary_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in summaries
        },
        "result_overrides": config.get("result_overrides", {}),
        "cell_inventory": cells,
    }
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "cell_inventory"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
