#!/usr/bin/env python3
"""Run the final Hard-regime invisible-steganography experiment."""

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

from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.confirmatory.validate_splits import (
    validate,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
DATA_ROOT = Path("generated_data/hard_regime/confirmatory")
RESPONSES_ROOT = Path("generated_data/hard_regime/confirmatory_eval_responses")
ARTIFACTS_ROOT = Path("artifacts/hard_regime/confirmatory")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(arguments: list[str]) -> None:
    print("[Hard confirmatory]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _variants(config: dict[str, Any], family: str) -> list[str]:
    return [str(config[family]["variant"])]


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
    provenance_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, provenance_root / "config.json")
    shutil.copy2(
        DATA_ROOT / "split_manifest.json",
        provenance_root / "global_split_manifest.json",
    )
    validation = validate(config_path, DATA_ROOT, seed)
    (provenance_root / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    for family in ("steganography",):
        seed_root = DATA_ROOT / family / f"seed_{seed}"
        shutil.copy2(
            seed_root / "split_manifest.json",
            provenance_root / f"{family}_seed_manifest.json",
        )
        for variant in _variants(config, family):
            shutil.copy2(
                seed_root / variant / "split_manifest.json",
                provenance_root / f"{family}_{variant}_manifest.json",
            )


def _check_steganography_tokenizer(
    config: dict[str, Any], model: str, seed: int
) -> None:
    section = config["steganography"]
    _run(
        [
            sys.executable,
            "scripts/shared/validation/check_steganography_tokenizer.py",
            "--model",
            str(config["models"][model]),
            "--split-root",
            str(DATA_ROOT / "steganography" / f"seed_{seed}"),
            "--sample-size",
            str(section["tokenizer_check_sample_size"]),
            "--max-target-tokens",
            str(section["training_max_new_tokens"]),
        ]
    )


def train(
    config: dict[str, Any], model: str, family: str, seed: int, mode: str
) -> None:
    section = config[family]
    if (
        model not in config["models"]
        or seed not in config["seeds"]
        or mode not in section["modes"]
    ):
        raise ValueError(f"Invalid training cell: {model}/{family}/seed_{seed}/{mode}")
    output_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / family / mode
    split_root = DATA_ROOT / family / f"seed_{seed}"
    steps = (int(config["train_n"]) + int(section["batch_size"]) - 1) // int(
        section["batch_size"]
    )
    steps *= int(section["epochs"])
    for variant in _variants(config, family):
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
                str(seed),
                "--deterministic-training",
            ]
        )


def evaluate(
    config: dict[str, Any], model: str, family: str, seed: int, mode: str
) -> None:
    section = config[family]
    artifact_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / family / mode
    response_root = RESPONSES_ROOT / model / f"seed_{seed}" / family / mode
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
                str(DATA_ROOT / family / f"seed_{seed}" / variant / "val_prompts"),
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
        raise ValueError(f"Invalid confirmatory cell: {model}/seed_{seed}")
    _write_provenance(config, config_path, model, seed)
    _check_steganography_tokenizer(config, model, seed)
    for family in ("steganography",):
        for mode in config[family]["modes"]:
            train(config, model, family, seed, str(mode))
    for family in ("steganography",):
        for mode in config[family]["modes"]:
            evaluate(config, model, family, seed, str(mode))


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
    parser.add_argument("--family", choices=("steganography",))
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
        train(config, str(args.model), str(args.family), int(args.seed), str(args.mode))
    elif args.action == "eval":
        evaluate(
            config, str(args.model), str(args.family), int(args.seed), str(args.mode)
        )
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
