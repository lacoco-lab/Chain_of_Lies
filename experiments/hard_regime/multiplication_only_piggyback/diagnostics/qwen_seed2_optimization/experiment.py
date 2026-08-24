#!/usr/bin/env python3
"""Retrain fixed Qwen seed-2 data under additional optimization seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXPERIMENT_DIR.parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.hard_regime.multiplication_only_piggyback.generate_splits import load_config as load_parent_config
from experiments.hard_regime.multiplication_only_piggyback.validate_splits import validate

DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
PARENT_CONFIG = PARENT_DIR / "config.json"
DATA_ROOT = Path("generated_data/hard_regime/multiplication_only_piggyback")
ORIGINAL_RESPONSES = Path(
    "generated_data/hard_regime/multiplication_only_piggyback_eval_responses/"
    "qwen/seed_2/public_cot/ckpt_final/baseline/arith_piggyback"
)
ARTIFACTS_ROOT = Path(
    "artifacts/hard_regime/multiplication_only_piggyback_diagnostics/qwen_seed2_optimization"
)
RESPONSES_ROOT = Path(
    "generated_data/hard_regime/multiplication_only_piggyback_diagnostics/"
    "qwen_seed2_optimization_eval_responses"
)


def load_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    parent_bytes = PARENT_CONFIG.read_bytes()
    observed_sha = hashlib.sha256(parent_bytes).hexdigest()
    if observed_sha != diagnostic.get("parent_config_sha256"):
        raise RuntimeError(
            f"Parent config drifted: expected {diagnostic.get('parent_config_sha256')}, got {observed_sha}."
        )
    parent = load_parent_config(PARENT_CONFIG)
    required = {
        "schema_version": 1,
        "model_key": "qwen",
        "data_seed": 2,
        "original_optimization_seed": 2,
        "additional_optimization_seeds": [3, 4, 5],
        "variant": "arith_piggyback",
        "mode": "public_cot",
    }
    mismatches = {key: (diagnostic.get(key), value) for key, value in required.items() if diagnostic.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Diagnostic config changed: {mismatches}")
    return diagnostic, parent


def _run(arguments: list[str]) -> None:
    print("[Qwen seed-2 optimization diagnostic]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _write_provenance(
    diagnostic: dict[str, Any], parent: dict[str, Any], optimization_seed: int
) -> None:
    data_seed = int(diagnostic["data_seed"])
    root = ARTIFACTS_ROOT / f"opt_seed_{optimization_seed}" / "provenance"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEFAULT_CONFIG, root / "diagnostic_config.json")
    shutil.copy2(PARENT_CONFIG, root / "parent_config.json")
    shutil.copy2(DATA_ROOT / "split_manifest.json", root / "global_split_manifest.json")
    shutil.copy2(DATA_ROOT / f"seed_{data_seed}" / "split_manifest.json", root / "seed_manifest.json")
    shutil.copy2(
        DATA_ROOT / f"seed_{data_seed}" / diagnostic["variant"] / "split_manifest.json",
        root / "variant_manifest.json",
    )
    validation = validate(PARENT_CONFIG, DATA_ROOT, data_seed)
    (root / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "diagnostic_cell.json").write_text(
        json.dumps(
            {
                "diagnostic": diagnostic["diagnostic_name"],
                "model_key": diagnostic["model_key"],
                "model_id": parent["models"][diagnostic["model_key"]],
                "data_seed": data_seed,
                "optimization_seed": optimization_seed,
                "variant": diagnostic["variant"],
                "mode": diagnostic["mode"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def train(
    diagnostic: dict[str, Any], parent: dict[str, Any], optimization_seed: int
) -> None:
    section = parent["piggyback"]
    data_seed = int(diagnostic["data_seed"])
    output_root = ARTIFACTS_ROOT / f"opt_seed_{optimization_seed}" / diagnostic["mode"]
    steps = (
        (int(parent["train_n"]) + int(section["batch_size"]) - 1)
        // int(section["batch_size"])
        * int(section["epochs"])
    )
    _run([
        sys.executable, "scripts/run_ce_only.py",
        "--variant", str(diagnostic["variant"]),
        "--model", str(parent["models"][diagnostic["model_key"]]),
        "--output-root", str(output_root),
        "--split-root", str(DATA_ROOT / f"seed_{data_seed}"),
        "--epochs", str(section["epochs"]),
        "--batch-size", str(section["batch_size"]),
        "--learning-rate", str(section["learning_rate"]),
        "--lora-r", str(section["lora_r"]),
        "--lora-alpha", str(section["lora_alpha"]),
        "--lora-dropout", str(section["lora_dropout"]),
        "--max-new-tokens", str(section["training_max_new_tokens"]),
        "--save-every", str(steps),
        "--eval-every", str(steps),
        "--validation-sample-size", str(section["validation_sample_size"]),
        "--validation-batch-size", str(section["validation_batch_size"]),
        "--expected-train-prompts", str(parent["train_n"]),
        "--expected-val-prompts", str(parent["val_n"]),
        "--supervision-mode", str(diagnostic["mode"]),
        "--seed", str(optimization_seed),
        "--deterministic-training",
    ])


def evaluate(
    diagnostic: dict[str, Any], parent: dict[str, Any], optimization_seed: int
) -> None:
    section = parent["piggyback"]
    data_seed = int(diagnostic["data_seed"])
    artifact_root = ARTIFACTS_ROOT / f"opt_seed_{optimization_seed}" / diagnostic["mode"]
    response_root = RESPONSES_ROOT / f"opt_seed_{optimization_seed}"
    baseline_destination = response_root / parent["checkpoint"] / "baseline" / diagnostic["variant"]
    if not ORIGINAL_RESPONSES.is_dir():
        raise RuntimeError(f"Missing reusable original baseline responses: {ORIGINAL_RESPONSES}")
    shutil.copytree(ORIGINAL_RESPONSES, baseline_destination, dirs_exist_ok=True)
    json_out = artifact_root / "per_variant_eval" / parent["checkpoint"] / f"{diagnostic['variant']}.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, "scripts/evaluate_ce_model.py",
        "--artifacts-root", str(artifact_root),
        "--variant", str(diagnostic["variant"]),
        "--prompts-dir", str(DATA_ROOT / f"seed_{data_seed}" / diagnostic["variant"] / "val_prompts"),
        "--eval-responses-root", str(response_root),
        "--model-responses-subdir", "trained",
        "--checkpoint", str(parent["checkpoint"]),
        "--max-new-tokens", str(section["evaluation_max_new_tokens"]),
        "--temperature", "0.0",
        "--greedy",
        "--inference-batch-size", str(section["validation_batch_size"]),
        "--json-out", str(json_out),
    ])


def run_cell(config_path: Path, optimization_seed: int) -> None:
    diagnostic, parent = load_config(config_path)
    if optimization_seed not in diagnostic["additional_optimization_seeds"]:
        raise ValueError(f"Unconfigured optimization seed: {optimization_seed}")
    _write_provenance(diagnostic, parent, optimization_seed)
    train(diagnostic, parent, optimization_seed)
    evaluate(diagnostic, parent, optimization_seed)


def summarize(config_path: Path) -> None:
    _run([
        sys.executable,
        str(EXPERIMENT_DIR / "summarize.py"),
        "--config", str(config_path),
        "--diagnostic-artifacts-root", str(ARTIFACTS_ROOT),
        "--original-artifacts-root", "artifacts/hard_regime/multiplication_only_piggyback",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cell", "summarize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--optimization-seed", type=int)
    args = parser.parse_args()
    if args.action == "cell":
        run_cell(args.config, int(args.optimization_seed))
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
