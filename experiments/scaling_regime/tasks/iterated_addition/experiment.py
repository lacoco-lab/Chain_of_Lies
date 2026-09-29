#!/usr/bin/env python3
"""Generate, train, and evaluate one balanced knowledge-length cell."""

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

from experiments.scaling_regime.tasks.iterated_addition.generate_splits import (
    generate,
    load_config,
    roots,
)
from experiments.scaling_regime.tasks.iterated_addition.validate_splits import validate


def _run(arguments: list[str]) -> None:
    print("[Knowledge length]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _condition_root(
    artifacts_root: Path, model: str, seed: int, condition: str
) -> Path:
    return artifacts_root / model / f"seed_{seed}" / condition


def _training_complete(adapter_root: Path) -> bool:
    metadata_path, history_path = (
        adapter_root / "training_metadata.json",
        adapter_root / "train_history.json",
    )
    if (
        not metadata_path.is_file()
        or not history_path.is_file()
        or not (adapter_root / "ckpt_final" / "adapter_config.json").is_file()
    ):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return (
        bool(history)
        and int(history[-1].get("step", 0)) >= int(metadata.get("steps", 0))
        and int(metadata.get("train_examples_seen", 0))
        >= int(metadata.get("target_train_examples_seen", 0))
    )


def _provenance(
    config_path: Path, config: dict[str, Any], model: str, seed: int, condition: str
) -> None:
    data_root, artifacts_root, _ = roots(config)
    destination = _condition_root(artifacts_root, model, seed, condition) / "provenance"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination / "config.json")
    shutil.copy2(data_root / "split_manifest.json", destination / "split_manifest.json")
    report = validate(config_path, data_root, seed)
    (destination / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def train(
    config_path: Path, config: dict[str, Any], model: str, seed: int, condition: str
) -> None:
    data_root, artifacts_root, _ = roots(config)
    condition_spec = config["conditions"][condition]
    variant = str(condition_spec["variant"])
    condition_root = _condition_root(artifacts_root, model, seed, condition)
    adapter_root = condition_root / variant
    if _training_complete(adapter_root):
        print(
            f"[Knowledge length] complete training already present; skipping {model}/seed_{seed}/{condition}",
            flush=True,
        )
        return
    if adapter_root.exists() and any(adapter_root.iterdir()):
        raise RuntimeError(
            f"Partial training artifacts exist at {adapter_root}. Move that directory aside before restarting; "
            "the trainer does not provide exact optimizer-state resume."
        )
    training = config["training"]
    train_total = int(config["train_examples_per_length"]) * len(config["lengths"])
    eval_total = int(config["eval_examples_per_length"]) * len(config["lengths"])
    batch_size, epochs = int(training["batch_size"]), int(training["epochs"])
    steps = ((train_total + batch_size - 1) // batch_size) * epochs
    if condition == "steganography":
        _run(
            [
                sys.executable,
                "experiments/shared/check_tokenizer.py",
                "--model",
                str(config["models"][model]),
                "--prompts-dir",
                str(data_root / f"seed_{seed}" / variant),
                "--sample-size",
                str(training["tokenizer_check_sample_size"]),
                "--max-target-tokens",
                str(training["training_max_new_tokens"]),
            ]
        )
    arguments = [
        sys.executable,
        "scripts/shared/training/run_ce.py",
        "--variant",
        variant,
        "--model",
        str(config["models"][model]),
        "--output-root",
        str(condition_root),
        "--train-prompts-dir",
        str(data_root / f"seed_{seed}" / variant),
        "--val-prompts-dir",
        str(data_root / "shared_eval" / variant),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(training["learning_rate"]),
        "--lora-r",
        str(training["lora_r"]),
        "--lora-alpha",
        str(training["lora_alpha"]),
        "--lora-dropout",
        str(training["lora_dropout"]),
        "--max-new-tokens",
        str(training["training_max_new_tokens"]),
        "--save-every",
        str(steps),
        "--eval-every",
        str(steps),
        "--validation-sample-size",
        str(training["validation_sample_size"]),
        "--validation-batch-size",
        str(training["validation_batch_size"]),
        "--expected-train-prompts",
        str(train_total),
        "--expected-val-prompts",
        str(eval_total),
        "--supervision-mode",
        str(condition_spec["supervision_mode"]),
        "--seed",
        str(seed),
        "--deterministic-training",
    ]
    if condition_spec.get("filler_token_count"):
        arguments.extend(
            ["--filler-token-count", str(condition_spec["filler_token_count"])]
        )
    _run(arguments)
    if not _training_complete(adapter_root):
        raise RuntimeError(
            f"Training returned without a complete final checkpoint: {adapter_root}"
        )


def evaluate(config: dict[str, Any], model: str, seed: int, condition: str) -> None:
    data_root, artifacts_root, responses_root = roots(config)
    condition_spec = config["conditions"][condition]
    variant = str(condition_spec["variant"])
    condition_root = _condition_root(artifacts_root, model, seed, condition)
    adapter_root = condition_root / variant
    training = config["training"]
    for length in config["lengths"]:
        report_path = (
            condition_root
            / "per_length_eval"
            / config["checkpoint"]
            / f"length_{length}.json"
        )
        _run(
            [
                sys.executable,
                "experiments/scaling_regime/tasks/iterated_addition/evaluate_checkpoint.py",
                "--adapter-root",
                str(adapter_root),
                "--variant",
                variant,
                "--prompts-dir",
                str(data_root / "shared_eval" / f"length_{length}" / variant),
                "--responses-root",
                str(
                    responses_root
                    / model
                    / f"seed_{seed}"
                    / condition
                    / f"length_{length}"
                ),
                "--report-path",
                str(report_path),
                "--checkpoint",
                str(config["checkpoint"]),
                "--max-new-tokens",
                str(training["evaluation_max_new_tokens"]),
                "--batch-size",
                str(training["validation_batch_size"]),
                "--model",
                model,
                "--seed",
                str(seed),
                "--condition",
                condition,
                "--length",
                str(length),
            ]
        )


def run_cell(
    config_path: Path, model: str, seed: int, condition: str, action: str
) -> None:
    config = load_config(config_path)
    if (
        model not in config["models"]
        or seed not in config["seeds"]
        or condition not in config["conditions"]
    ):
        raise ValueError(f"Invalid cell: {model}/seed_{seed}/{condition}")
    _provenance(config_path, config, model, seed, condition)
    if action in {"train", "cell"}:
        train(config_path, config, model, seed, condition)
    if action in {"evaluate", "cell"}:
        evaluate(config, model, seed, condition)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("generate", "train", "evaluate", "cell"))
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    p.add_argument("--model", choices=("qwen", "llama"))
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--condition",
        choices=("vanilla", "filler", "public_only_cot", "piggyback", "steganography"),
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.action == "generate":
        data_root = roots(cfg)[0]
        generate(a.config, data_root, overwrite=a.overwrite)
        validate(a.config, data_root)
    else:
        if a.model is None or a.seed is None or a.condition is None:
            p.error(
                "--model, --seed, and --condition are required for train/evaluate/cell"
            )
        run_cell(a.config, a.model, a.seed, a.condition, a.action)
