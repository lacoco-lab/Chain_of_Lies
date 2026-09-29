"""Shared execution harness for finalized S5 and knowledge Hard-regime suites."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(arguments: list[str]) -> None:
    print("[Hard task]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _layout(task: str) -> tuple[Path, Path, Path, Path]:
    experiment_dir = (
        Path("experiments/legacy/scaling_regime/endpoint_regimes/hard_regime") / task
    )
    return (
        experiment_dir,
        Path("generated_data/hard_regime") / task,
        Path("generated_data/hard_regime") / f"{task}_eval_responses",
        Path("artifacts/hard_regime") / task,
    )


def load_config(task: str, config_path: Path) -> dict[str, Any]:
    module = importlib.import_module(
        f"experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.{task}.generate_splits"
    )
    return module.load_config(config_path)


def generate(task: str, config_path: Path) -> None:
    experiment_dir, data_root, _, _ = _layout(task)
    _run(
        [
            sys.executable,
            str(experiment_dir / "generate_splits.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(data_root),
        ]
    )
    _run(
        [
            sys.executable,
            str(experiment_dir / "validate_splits.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
        ]
    )


def _variants(config: dict[str, Any], family: str) -> list[str]:
    return [
        str(value)
        for value in config[family].get("variants", [config[family].get("variant")])
    ]


def _validate(
    task: str, config_path: Path, data_root: Path, seed: int
) -> dict[str, Any]:
    module = importlib.import_module(
        f"experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.{task}.validate_splits"
    )
    return module.validate(config_path, data_root, seed)


def _provenance(
    task: str, config: dict[str, Any], config_path: Path, model: str, seed: int
) -> None:
    _, data_root, _, artifacts_root = _layout(task)
    root = artifacts_root / model / f"seed_{seed}" / "provenance"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, root / "config.json")
    shutil.copy2(data_root / "split_manifest.json", root / "global_split_manifest.json")
    shutil.copy2(
        data_root / f"seed_{seed}" / "split_manifest.json", root / "seed_manifest.json"
    )
    report = _validate(task, config_path, data_root, seed)
    (root / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _tokenizer_check(task: str, config: dict[str, Any], model: str, seed: int) -> None:
    _, data_root, _, _ = _layout(task)
    section = config["steganography"]
    variant = _variants(config, "steganography")[0]
    _run(
        [
            sys.executable,
            "scripts/shared/validation/check_hard_task_steganography_tokenizer.py",
            "--model",
            str(config["models"][model]),
            "--prompts-dir",
            str(data_root / f"seed_{seed}" / variant / "train_prompts"),
            "--sample-size",
            str(section["tokenizer_check_sample_size"]),
            "--max-target-tokens",
            str(section["training_max_new_tokens"]),
        ]
    )


def train(
    task: str, config: dict[str, Any], model: str, seed: int, family: str, mode: str
) -> None:
    _, data_root, _, artifacts_root = _layout(task)
    section = config[family]
    output_root = artifacts_root / model / f"seed_{seed}" / family / mode
    steps = (
        (int(config["train_n"]) + int(section["batch_size"]) - 1)
        // int(section["batch_size"])
    ) * int(section["epochs"])
    for variant in _variants(config, family):
        variant_root = data_root / f"seed_{seed}" / variant
        _run(
            [
                sys.executable,
                "scripts/shared/training/run_ce.py",
                "--variant",
                variant,
                "--model",
                str(config["models"][model]),
                "--output-root",
                str(output_root),
                "--train-prompts-dir",
                str(variant_root / "train_prompts"),
                "--val-prompts-dir",
                str(variant_root / "val_prompts"),
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
                mode,
                "--seed",
                str(seed),
                "--deterministic-training",
            ]
        )


def evaluate(
    task: str, config: dict[str, Any], model: str, seed: int, family: str, mode: str
) -> None:
    _, data_root, responses_root, artifacts_root = _layout(task)
    section = config[family]
    artifact_root = artifacts_root / model / f"seed_{seed}" / family / mode
    for variant in _variants(config, family):
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
                variant,
                "--prompts-dir",
                str(data_root / f"seed_{seed}" / variant / "val_prompts"),
                "--eval-responses-root",
                str(responses_root / model / f"seed_{seed}" / family / mode),
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


def run_cell(task: str, config_path: Path, model: str, seed: int) -> None:
    config = load_config(task, config_path)
    if model not in config["models"] or seed not in config["seeds"]:
        raise ValueError(f"Invalid cell {task}/{model}/seed_{seed}.")
    _provenance(task, config, config_path, model, seed)
    _tokenizer_check(task, config, model, seed)
    for family in ("piggyback", "steganography"):
        for mode in config[family]["modes"]:
            train(task, config, model, seed, family, str(mode))
    for family in ("piggyback", "steganography"):
        for mode in config[family]["modes"]:
            evaluate(task, config, model, seed, family, str(mode))
