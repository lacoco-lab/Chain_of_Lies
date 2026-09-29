#!/usr/bin/env python3
"""Generate data or run one model/condition cell of the parity calibration."""

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

from experiments.scaling_regime.tasks.plain_parity.generate_splits import (
    generate,
    load_config,
)
from experiments.scaling_regime.tasks.plain_parity.validate_splits import validate

DATA_ROOT = Path("generated_data/parity_regime_calibration")
RESPONSES_ROOT = Path("generated_data/parity_regime_calibration_eval_responses")
ARTIFACTS_ROOT = Path("artifacts/parity_regime_calibration")


def _run(arguments: list[str]) -> None:
    print("[Parity calibration]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def run_cell(config_path: Path, model: str, condition: str) -> None:
    config = load_config(config_path)
    if model not in config["models"] or condition not in config["conditions"]:
        raise ValueError(f"Invalid parity cell: {model}/{condition}.")
    seed = int(config["seed"])
    report = validate(config_path, DATA_ROOT)
    provenance = ARTIFACTS_ROOT / model / f"seed_{seed}" / condition / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, provenance / "config.json")
    seed_manifest = DATA_ROOT / f"split_manifest_seed_{seed}.json"
    if not seed_manifest.exists() and seed == 0:
        seed_manifest = DATA_ROOT / "split_manifest.json"
    shutil.copy2(seed_manifest, provenance / "split_manifest.json")
    (provenance / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    condition_spec: dict[str, Any] = config["conditions"][condition]
    variant = str(condition_spec["variant"])
    training = config["training"]
    train_n = int(config["train_per_bucket"]) * len(config["length_buckets"])
    val_n = int(config["validation_per_bucket"]) * len(config["length_buckets"])
    output_root = ARTIFACTS_ROOT / model / f"seed_{seed}" / condition
    prompts_root = DATA_ROOT / f"seed_{seed}" / variant
    # Condor declares both cell roots as transfer outputs. Materialize them
    # before training so a CUDA/runtime failure remains the real exit error
    # instead of being masked by a secondary missing-output hold.
    output_root.mkdir(parents=True, exist_ok=True)
    (RESPONSES_ROOT / model / f"seed_{seed}" / condition).mkdir(
        parents=True, exist_ok=True
    )

    if condition == "steganography":
        _run(
            [
                sys.executable,
                "experiments/shared/check_tokenizer.py",
                "--model",
                config["models"][model],
                "--prompts-dir",
                str(prompts_root / "train_prompts"),
                "--sample-size",
                str(training["tokenizer_check_sample_size"]),
                "--max-target-tokens",
                str(training["training_max_new_tokens"]),
            ]
        )

    run_training_validation = bool(training.get("run_training_validation", True))
    train_args = [
        sys.executable,
        "scripts/shared/training/run_ce.py",
        "--variant",
        variant,
        "--model",
        config["models"][model],
        "--output-root",
        str(output_root),
        "--train-prompts-dir",
        str(prompts_root / "train_prompts"),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
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
        str(train_n),
        "--eval-every",
        str(train_n),
        "--validation-batch-size",
        str(training["validation_batch_size"]),
        "--expected-train-prompts",
        str(train_n),
        "--supervision-mode",
        str(condition_spec["supervision_mode"]),
        "--seed",
        str(seed),
        "--deterministic-training",
    ]
    if run_training_validation:
        train_args.extend(
            [
                "--val-prompts-dir",
                str(prompts_root / "val_prompts"),
                "--validation-sample-size",
                str(training["validation_sample_size"]),
                "--expected-val-prompts",
                str(val_n),
            ]
        )
    if training.get("memory_efficient_ce", False):
        train_args.extend(
            [
                "--memory-efficient-ce",
                "--ce-token-chunk-size",
                str(training["ce_token_chunk_size"]),
            ]
        )
    if training.get("activation_cpu_offload", False):
        train_args.append("--activation-cpu-offload")
    if condition_spec.get("filler_token_count"):
        train_args.extend(
            ["--filler-token-count", str(condition_spec["filler_token_count"])]
        )
    _run(train_args)

    report_path = (
        output_root / variant / "per_variant_eval" / "ckpt_final" / f"{variant}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/shared/evaluation/evaluate_ce.py",
            "--artifacts-root",
            str(output_root),
            "--variant",
            variant,
            "--prompts-dir",
            str(prompts_root / "val_prompts"),
            "--eval-responses-root",
            str(RESPONSES_ROOT / model / f"seed_{seed}" / condition),
            "--model-responses-subdir",
            "finetuned",
            "--checkpoint",
            "ckpt_final",
            "--max-new-tokens",
            str(training["evaluation_max_new_tokens"]),
            "--temperature",
            "0.0",
            "--greedy",
            "--inference-batch-size",
            str(
                training.get("evaluation_batch_size", training["validation_batch_size"])
            ),
            "--json-out",
            str(report_path),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "cell"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--model", choices=("qwen", "llama"))
    parser.add_argument(
        "--condition",
        choices=("vanilla", "public_only_cot", "filler", "piggyback", "steganography"),
    )
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.config, DATA_ROOT)
        print(json.dumps(validate(args.config, DATA_ROOT), indent=2))
    else:
        if args.model is None or args.condition is None:
            parser.error("cell requires --model and --condition")
        run_cell(args.config, args.model, args.condition)


if __name__ == "__main__":
    main()
