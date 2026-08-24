#!/usr/bin/env python3
"""Run one strictly-disjoint Hard-regime calibration cell."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
DATA_ROOT = Path("generated_data/hard_regime_range_calibration")
RESPONSES_ROOT = Path("generated_data/hard_regime_range_calibration_eval_responses")
ARTIFACTS_ROOT = Path("artifacts/hard_regime_range_calibration")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(arguments: list[str]) -> None:
    print("[Hard-range calibration]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _validate(config_path: Path, family: str, range_id: str) -> None:
    _run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "validate_splits.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(DATA_ROOT),
            "--family",
            family,
            "--range",
            range_id,
        ]
    )


def generate(config_path: Path, family: str, range_id: str) -> None:
    _run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "generate_splits.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(DATA_ROOT),
            "--family",
            family,
            "--range",
            range_id,
        ]
    )
    _validate(config_path, family, range_id)


def _variants(config: dict[str, Any], family: str) -> list[str]:
    if family == "piggyback":
        return list(config[family]["variants"])
    return [str(config[family]["variant"])]


def train(
    config: dict[str, Any],
    config_path: Path,
    model_key: str,
    family: str,
    range_id: str,
    mode: str,
) -> None:
    section = config[family]
    if model_key not in config["models"]:
        raise ValueError(f"Unknown model key {model_key!r}.")
    if mode not in section["modes"]:
        raise ValueError(f"Unsupported {family} mode {mode!r}; choose from {section['modes']}.")
    _validate(config_path, family, range_id)
    split_root = DATA_ROOT / family / range_id / f"seed_{config['seed']}"
    output_root = ARTIFACTS_ROOT / model_key / family / range_id / f"seed_{config['seed']}" / mode
    steps = (int(config["train_n"]) + int(section["batch_size"]) - 1) // int(section["batch_size"])
    steps *= int(section["epochs"])

    if family == "steganography" and mode == "local_channel_cot":
        _run(
            [
                sys.executable,
                "scripts/check_steganography_tokenizer.py",
                "--model",
                str(config["models"][model_key]),
                "--split-root",
                str(split_root),
                "--sample-size",
                str(section["tokenizer_check_sample_size"]),
                "--max-target-tokens",
                str(section["training_max_new_tokens"]),
            ]
        )

    for variant in _variants(config, family):
        _run(
            [
                sys.executable,
                "scripts/run_ce_only.py",
                "--variant",
                variant,
                "--model",
                str(config["models"][model_key]),
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
                mode,
                "--seed",
                str(config["seed"]),
                "--deterministic-training",
            ]
        )


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    model_key: str,
    family: str,
    range_id: str,
    mode: str,
) -> None:
    section = config[family]
    if model_key not in config["models"] or mode not in section["modes"]:
        raise ValueError(f"Invalid model/mode combination: {model_key}/{mode}.")
    _validate(config_path, family, range_id)
    artifact_root = ARTIFACTS_ROOT / model_key / family / range_id / f"seed_{config['seed']}" / mode
    # Keep each mode in an independent transfer directory so Condor evaluation
    # jobs can finish in either order without overwriting one another.
    response_root = RESPONSES_ROOT / model_key / family / range_id / f"seed_{config['seed']}" / mode
    for variant in _variants(config, family):
        json_out = artifact_root / "per_variant_eval" / config["checkpoint"] / f"{variant}.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "scripts/evaluate_ce_model.py",
                "--artifacts-root",
                str(artifact_root),
                "--variant",
                variant,
                "--prompts-dir",
                str(
                    DATA_ROOT
                    / family
                    / range_id
                    / f"seed_{config['seed']}"
                    / variant
                    / "val_prompts"
                ),
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


def summarize(config_path: Path, model_key: str, family: str) -> None:
    _run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "summarize.py"),
            "--config",
            str(config_path),
            "--artifacts-root",
            str(ARTIFACTS_ROOT),
            "--model",
            model_key,
            "--family",
            family,
        ]
    )


def run_cell(
    config: dict[str, Any],
    config_path: Path,
    model_key: str,
    family: str,
    range_id: str,
) -> None:
    """Run one complete range as an ordinary, self-contained GPU job."""
    generate(config_path, family, range_id)
    for mode in config[family]["modes"]:
        train(config, config_path, model_key, family, range_id, str(mode))
    for mode in config[family]["modes"]:
        evaluate(config, config_path, model_key, family, range_id, str(mode))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "validate", "train", "eval", "summarize", "cell"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("qwen", "llama"))
    parser.add_argument("--family", choices=("piggyback", "steganography"), required=True)
    parser.add_argument("--range", dest="range_id")
    parser.add_argument("--mode")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "summarize":
        summarize(args.config, str(args.model), args.family)
        return
    if args.range_id is None:
        parser.error("--range is required for generate, validate, train, and eval")
    if args.action == "generate":
        generate(args.config, args.family, args.range_id)
    elif args.action == "validate":
        _validate(args.config, args.family, args.range_id)
    elif args.action == "train":
        train(config, args.config, str(args.model), args.family, args.range_id, str(args.mode))
    elif args.action == "cell":
        run_cell(config, args.config, str(args.model), args.family, args.range_id)
    else:
        evaluate(config, args.config, str(args.model), args.family, args.range_id, str(args.mode))


if __name__ == "__main__":
    main()
