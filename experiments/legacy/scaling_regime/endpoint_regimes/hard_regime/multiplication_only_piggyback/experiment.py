#!/usr/bin/env python3
"""Run multiplication-only piggyback generation, training, evaluation, and summary."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.multiplication_only_piggyback.generate_splits import (
    load_config,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.multiplication_only_piggyback.validate_splits import (
    validate,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
DATA_ROOT = Path("generated_data/hard_regime/multiplication_only_piggyback")
RESPONSES_ROOT = Path(
    "generated_data/hard_regime/multiplication_only_piggyback_eval_responses"
)
ARTIFACTS_ROOT = Path("artifacts/hard_regime/multiplication_only_piggyback")


def _run(arguments: list[str]) -> None:
    print("[Multiplication-only piggyback]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def generate(config_path: Path) -> None:
    _run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "generate_splits.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(DATA_ROOT),
        ]
    )
    _run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "validate_splits.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(DATA_ROOT),
        ]
    )


def _write_provenance(
    config: dict[str, Any], config_path: Path, model: str, seed: int
) -> None:
    root = ARTIFACTS_ROOT / model / f"seed_{seed}" / "provenance"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, root / "config.json")
    shutil.copy2(DATA_ROOT / "split_manifest.json", root / "global_split_manifest.json")
    shutil.copy2(
        DATA_ROOT / f"seed_{seed}" / "split_manifest.json", root / "seed_manifest.json"
    )
    for variant in config["piggyback"]["variants"]:
        shutil.copy2(
            DATA_ROOT / f"seed_{seed}" / variant / "split_manifest.json",
            root / f"{variant}_manifest.json",
        )
    report = validate(config_path, DATA_ROOT, seed)
    (root / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def train(config: dict[str, Any], model: str, seed: int, mode: str) -> None:
    section = config["piggyback"]
    if (
        model not in config["models"]
        or seed not in config["seeds"]
        or mode not in section["modes"]
    ):
        raise ValueError(f"Invalid training cell: {model}/seed_{seed}/{mode}")
    output_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / mode
    split_root = DATA_ROOT / f"seed_{seed}"
    steps = (
        (int(config["train_n"]) + int(section["batch_size"]) - 1)
        // int(section["batch_size"])
        * int(section["epochs"])
    )
    for variant in section["variants"]:
        _run(
            [
                sys.executable,
                "scripts/shared/training/run_ce.py",
                "--variant",
                str(variant),
                "--model",
                str(config["models"][model]),
                "--output-root",
                str(output_root),
                "--split-root",
                str(split_root),
                "--epochs",
                str(section["epochs"]),
                "--batch-size",
                str(section["batch_size"]),
                "--learning-rate",
                str(section["learning_rate"]),
                "--lora-r",
                str(section["lora_r"]),
                "--lora-alpha",
                str(section["lora_alpha"]),
                "--lora-dropout",
                str(section["lora_dropout"]),
                "--max-new-tokens",
                str(section["training_max_new_tokens"]),
                "--save-every",
                str(steps),
                "--eval-every",
                str(steps),
                "--validation-sample-size",
                str(section["validation_sample_size"]),
                "--validation-batch-size",
                str(section["validation_batch_size"]),
                "--expected-train-prompts",
                str(config["train_n"]),
                "--expected-val-prompts",
                str(config["val_n"]),
                "--supervision-mode",
                str(mode),
                "--seed",
                str(seed),
                "--deterministic-training",
            ]
        )


def evaluate(config: dict[str, Any], model: str, seed: int, mode: str) -> None:
    section = config["piggyback"]
    if (
        model not in config["models"]
        or seed not in config["seeds"]
        or mode not in section["modes"]
    ):
        raise ValueError(f"Invalid evaluation cell: {model}/seed_{seed}/{mode}")
    artifact_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / mode
    response_root = RESPONSES_ROOT / model / f"seed_{seed}" / mode
    for variant in section["variants"]:
        json_out = (
            artifact_root
            / "per_variant_eval"
            / config["checkpoint"]
            / f"{variant}.json"
        )
        json_out.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "scripts/shared/evaluation/evaluate_ce.py",
                "--artifacts-root",
                str(artifact_root),
                "--variant",
                str(variant),
                "--prompts-dir",
                str(DATA_ROOT / f"seed_{seed}" / variant / "val_prompts"),
                "--eval-responses-root",
                str(response_root),
                "--model-responses-subdir",
                "trained",
                "--checkpoint",
                str(config["checkpoint"]),
                "--max-new-tokens",
                str(section["evaluation_max_new_tokens"]),
                "--temperature",
                "0.0",
                "--greedy",
                "--inference-batch-size",
                str(section["validation_batch_size"]),
                "--json-out",
                str(json_out),
            ]
        )


def run_cell(config: dict[str, Any], config_path: Path, model: str, seed: int) -> None:
    if model not in config["models"] or seed not in config["seeds"]:
        raise ValueError(f"Invalid cell: {model}/seed_{seed}")
    _write_provenance(config, config_path, model, seed)
    for mode in config["piggyback"]["modes"]:
        train(config, model, seed, str(mode))
    for mode in config["piggyback"]["modes"]:
        evaluate(config, model, seed, str(mode))


def summarize(config_path: Path) -> None:
    _run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "summarize.py"),
            "--config",
            str(config_path),
            "--artifacts-root",
            str(ARTIFACTS_ROOT),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("generate", "validate", "train", "eval", "cell", "summarize")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("qwen", "llama"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "generate":
        generate(args.config)
    elif args.action == "validate":
        validate(args.config, DATA_ROOT, args.seed)
    elif args.action == "cell":
        run_cell(config, args.config, str(args.model), int(args.seed))
    elif args.action == "train":
        train(config, str(args.model), int(args.seed), str(args.mode))
    elif args.action == "eval":
        evaluate(config, str(args.model), int(args.seed), str(args.mode))
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
