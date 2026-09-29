#!/usr/bin/env python3
"""Generate or run one task/model/seed cell of matched Easy v2."""

from __future__ import annotations
import argparse, json, shutil, subprocess, sys
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
    active_tasks,
    generate,
    load_config,
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.validate_splits import (
    validate,
)

DEFAULT_DATA_ROOT = Path("generated_data/easy_regime")
DEFAULT_ARTIFACTS_ROOT = Path("artifacts/easy_regime")
DEFAULT_RESPONSES_ROOT = Path("generated_data/easy_regime_eval_responses")


def _run(args: list[str]) -> None:
    print("[Easy]", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _name(task: str, template: str) -> str:
    return template.format(task=task)


def _roots(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = config.get("paths", {})
    return (
        Path(paths.get("data_root", DEFAULT_DATA_ROOT)),
        Path(paths.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT)),
        Path(paths.get("responses_root", DEFAULT_RESPONSES_ROOT)),
    )


def run_cell(
    config_path: Path,
    task: str,
    model: str,
    seed: int,
    conditions: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    if (
        task not in active_tasks(config)
        or model not in config["models"]
        or seed not in config["seeds"]
    ):
        raise ValueError(f"Invalid Easy cell: {task}/{model}/seed_{seed}")
    selected = conditions or list(config["conditions"])
    unknown = set(selected) - set(config["conditions"])
    if not selected or unknown:
        raise ValueError(
            f"Invalid conditions: {sorted(unknown) if unknown else selected}"
        )
    data_root, artifacts_root, responses_root = _roots(config)
    report = validate(config_path, data_root, seed)
    provenance = artifacts_root / task / model / f"seed_{seed}" / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, provenance / "config.json")
    shutil.copy2(
        data_root / "split_manifest.json", provenance / "global_split_manifest.json"
    )
    shutil.copy2(
        data_root / task / f"seed_{seed}" / "split_manifest.json",
        provenance / "seed_manifest.json",
    )
    (provenance / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    training = config["training"]
    steg_variant = _name(task, config["conditions"]["steganography"]["variant"])
    if "steganography" in selected:
        _run(
            [
                sys.executable,
                "experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/check_tokenizer.py",
                "--model",
                config["models"][model],
                "--prompts-dir",
                str(data_root / task / f"seed_{seed}" / steg_variant / "train_prompts"),
                "--sample-size",
                str(training["tokenizer_check_sample_size"]),
                "--max-target-tokens",
                str(training["training_max_new_tokens"]),
            ]
        )
    steps = (
        (int(config["train_n"]) + int(training["batch_size"]) - 1)
        // int(training["batch_size"])
    ) * int(training["epochs"])
    for condition in selected:
        spec = config["conditions"][condition]
        name = _name(task, spec["variant"])
        root = artifacts_root / task / model / f"seed_{seed}" / condition
        args = [
            sys.executable,
            "scripts/shared/training/run_ce.py",
            "--variant",
            name,
            "--model",
            config["models"][model],
            "--output-root",
            str(root),
            "--train-prompts-dir",
            str(data_root / task / f"seed_{seed}" / name / "train_prompts"),
            "--val-prompts-dir",
            str(data_root / task / f"seed_{seed}" / name / "val_prompts"),
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
            str(steps),
            "--eval-every",
            str(steps),
            "--validation-sample-size",
            str(training["validation_sample_size"]),
            "--validation-batch-size",
            str(training["validation_batch_size"]),
            "--expected-train-prompts",
            str(config["train_n"]),
            "--expected-val-prompts",
            str(config["val_n"]),
            "--supervision-mode",
            spec["supervision_mode"],
            "--seed",
            str(seed),
            "--deterministic-training",
        ]
        if spec.get("filler_token_count"):
            args.extend(["--filler-token-count", str(spec["filler_token_count"])])
        _run(args)
    for condition in selected:
        spec = config["conditions"][condition]
        name = _name(task, spec["variant"])
        root = artifacts_root / task / model / f"seed_{seed}" / condition
        report_path = root / "per_variant_eval" / config["checkpoint"] / f"{name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "scripts/shared/evaluation/evaluate_ce.py",
                "--artifacts-root",
                str(root),
                "--variant",
                name,
                "--prompts-dir",
                str(data_root / task / f"seed_{seed}" / name / "val_prompts"),
                "--eval-responses-root",
                str(responses_root / task / model / f"seed_{seed}" / condition),
                "--model-responses-subdir",
                "trained",
                "--checkpoint",
                config["checkpoint"],
                "--max-new-tokens",
                str(training["evaluation_max_new_tokens"]),
                "--temperature",
                "0.0",
                "--greedy",
                "--inference-batch-size",
                str(training["validation_batch_size"]),
                "--json-out",
                str(report_path),
            ]
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("generate", "cell"))
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--model", choices=("qwen", "llama"))
    p.add_argument("--seed", type=int)
    p.add_argument("--conditions", nargs="+")
    a = p.parse_args()
    config = load_config(a.config)
    data_root, _, _ = _roots(config)
    if a.action == "generate":
        generate(a.config, data_root)
        validate(a.config, data_root)
    else:
        run_cell(a.config, str(a.task), str(a.model), int(a.seed), a.conditions)
