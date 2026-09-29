#!/usr/bin/env python3
"""Train and evaluate isolated Filler-only scaling cells on finalized splits."""

from __future__ import annotations

import argparse
import hashlib
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


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(arguments: list[str]) -> None:
    print("[Filler-only]", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _task(config: dict[str, Any], task: str) -> dict[str, Any]:
    if task not in config["tasks"]:
        raise ValueError(
            f"Unknown task {task!r}; expected one of {sorted(config['tasks'])}."
        )
    return config["tasks"][task]


def _training(task_spec: dict[str, Any], seed: int) -> dict[str, Any]:
    result = dict(task_spec["training"])
    result.update(task_spec.get("seed_training_overrides", {}).get(str(seed), {}))
    return result


def _directories(task_spec: dict[str, Any], seed: int) -> tuple[Path, Path]:
    data_root = Path(task_spec["data_root"])
    variant = str(task_spec["variant"])
    if task_spec["evaluation_kind"] == "parity_validation":
        base = data_root / f"seed_{seed}" / variant
        return base / "train_prompts", base / "val_prompts"
    return data_root / f"seed_{seed}" / variant, data_root / "shared_eval" / variant


def _source_manifest(task_spec: dict[str, Any], seed: int) -> Path:
    pattern = task_spec.get("source_manifest_pattern")
    if pattern:
        candidate = Path(str(pattern).format(seed=seed))
        if not candidate.exists() and seed == 0:
            candidate = Path(task_spec["data_root"]) / "split_manifest.json"
        return candidate
    return Path(task_spec["source_manifest"])


def _source_config(task_spec: dict[str, Any], seed: int) -> Path:
    by_seed = task_spec.get("source_configs_by_seed", {})
    return Path(by_seed.get(str(seed), task_spec.get("source_config", "")))


def _condition_root(task_spec: dict[str, Any], model: str, seed: int) -> Path:
    return Path(task_spec["artifacts_root"]) / model / f"seed_{seed}" / "filler_only"


def _adapter_root(task_spec: dict[str, Any], model: str, seed: int) -> Path:
    return _condition_root(task_spec, model, seed) / str(task_spec["variant"])


def _training_complete(adapter_root: Path) -> bool:
    metadata_path = adapter_root / "training_metadata.json"
    history_path = adapter_root / "train_history.json"
    checkpoint = adapter_root / "ckpt_final" / "adapter_config.json"
    if (
        not metadata_path.is_file()
        or not history_path.is_file()
        or not checkpoint.is_file()
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cell_inputs(
    task_spec: dict[str, Any], seed: int
) -> tuple[Path, Path, Path]:
    train_dir, eval_dir = _directories(task_spec, seed)
    manifest = _source_manifest(task_spec, seed)
    expected_train = int(task_spec["train_examples_per_difficulty"]) * len(
        task_spec["difficulty_values"]
    )
    expected_eval = int(task_spec["eval_examples_per_difficulty"]) * len(
        task_spec["difficulty_values"]
    )
    actual_train = len(list(train_dir.glob("*.json")))
    actual_eval = len(list(eval_dir.glob("*.json")))
    if actual_train != expected_train:
        raise RuntimeError(
            f"Expected {expected_train} unchanged train records in {train_dir}, found {actual_train}."
        )
    if actual_eval != expected_eval:
        raise RuntimeError(
            f"Expected {expected_eval} unchanged eval records in {eval_dir}, found {actual_eval}."
        )
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing source split manifest: {manifest}")
    return train_dir, eval_dir, manifest


def _write_provenance(
    config_path: Path,
    config: dict[str, Any],
    task: str,
    model: str,
    seed: int,
    train_dir: Path,
    eval_dir: Path,
    manifest: Path,
) -> None:
    task_spec = _task(config, task)
    destination = _condition_root(task_spec, model, seed) / "provenance"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination / "config.json")
    shutil.copy2(manifest, destination / "source_split_manifest.json")
    source_config = _source_config(task_spec, seed)
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing source experiment config: {source_config}")
    shutil.copy2(source_config, destination / "source_experiment_config.json")
    cell = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "task": task,
        "model": model,
        "seed": seed,
        "condition": "filler_only",
        "supervision_mode": "filler_only",
        "variant": task_spec["variant"],
        "train_prompts_dir": str(train_dir),
        "eval_prompts_dir": str(eval_dir),
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "source_experiment_config": str(source_config),
        "source_experiment_config_sha256": _sha256(source_config),
        "difficulty_field": task_spec["difficulty_field"],
        "filler_token_counts": task_spec["filler_token_counts"],
        "filler_token_text": config["filler_token_text"],
        "training": _training(task_spec, seed),
    }
    (destination / "cell_manifest.json").write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )


def train(
    config_path: Path, config: dict[str, Any], task: str, model: str, seed: int
) -> None:
    task_spec = _task(config, task)
    train_dir, eval_dir, manifest = _validate_cell_inputs(task_spec, seed)
    _write_provenance(
        config_path, config, task, model, seed, train_dir, eval_dir, manifest
    )
    adapter_root = _adapter_root(task_spec, model, seed)
    if _training_complete(adapter_root):
        print(
            f"[Filler-only] complete training already present; skipping {task}/{model}/seed_{seed}",
            flush=True,
        )
        return
    if adapter_root.exists() and any(adapter_root.iterdir()):
        raise RuntimeError(
            f"Partial training artifacts exist at {adapter_root}; move them aside before retrying."
        )

    training = _training(task_spec, seed)
    train_total = int(task_spec["train_examples_per_difficulty"]) * len(
        task_spec["difficulty_values"]
    )
    eval_total = int(task_spec["eval_examples_per_difficulty"]) * len(
        task_spec["difficulty_values"]
    )
    steps = (
        (train_total + int(training["batch_size"]) - 1) // int(training["batch_size"])
    ) * int(training["epochs"])
    arguments = [
        sys.executable,
        "scripts/shared/training/run_ce.py",
        "--variant",
        str(task_spec["variant"]),
        "--model",
        str(config["models"][model]),
        "--output-root",
        str(_condition_root(task_spec, model, seed)),
        "--train-prompts-dir",
        str(train_dir),
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
        "--validation-batch-size",
        str(training["validation_batch_size"]),
        "--expected-train-prompts",
        str(train_total),
        "--supervision-mode",
        "filler_only",
        "--filler-token-count-field",
        str(task_spec["difficulty_field"]),
        "--filler-token-counts-json",
        json.dumps(task_spec["filler_token_counts"], separators=(",", ":")),
        "--seed",
        str(seed),
        "--deterministic-training",
    ]
    if bool(training.get("run_training_validation", True)):
        arguments.extend(
            [
                "--val-prompts-dir",
                str(eval_dir),
                "--validation-sample-size",
                str(training["validation_sample_size"]),
                "--expected-val-prompts",
                str(eval_total),
            ]
        )
    if bool(training.get("memory_efficient_ce", False)):
        arguments.extend(
            [
                "--memory-efficient-ce",
                "--ce-token-chunk-size",
                str(training["ce_token_chunk_size"]),
            ]
        )
    if bool(training.get("activation_cpu_offload", False)):
        arguments.append("--activation-cpu-offload")
    _run(arguments)
    if not _training_complete(adapter_root):
        raise RuntimeError(
            f"Training returned without a complete final checkpoint: {adapter_root}"
        )


def evaluate(config: dict[str, Any], task: str, model: str, seed: int) -> None:
    task_spec = _task(config, task)
    _, eval_dir, _ = _validate_cell_inputs(task_spec, seed)
    adapter_root = _adapter_root(task_spec, model, seed)
    if not _training_complete(adapter_root):
        raise RuntimeError(
            f"Cannot evaluate incomplete training artifacts: {adapter_root}"
        )
    condition_root = _condition_root(task_spec, model, seed)
    responses_root = (
        Path(task_spec["responses_root"]) / model / f"seed_{seed}" / "filler_only"
    )
    training = _training(task_spec, seed)
    variant = str(task_spec["variant"])

    if task_spec["evaluation_kind"] == "parity_validation":
        report_path = (
            condition_root
            / variant
            / "per_variant_eval"
            / "ckpt_final"
            / f"{variant}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "scripts/shared/evaluation/evaluate_ce.py",
                "--artifacts-root",
                str(condition_root),
                "--variant",
                variant,
                "--prompts-dir",
                str(eval_dir),
                "--eval-responses-root",
                str(responses_root),
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
                    training.get(
                        "evaluation_batch_size", training["validation_batch_size"]
                    )
                ),
                "--json-out",
                str(report_path),
            ]
        )
        return

    for difficulty in task_spec["difficulty_values"]:
        _run(
            [
                sys.executable,
                str(task_spec["evaluation_script"]),
                "--adapter-root",
                str(adapter_root),
                "--variant",
                variant,
                "--prompts-dir",
                str(
                    Path(task_spec["data_root"])
                    / "shared_eval"
                    / f"length_{difficulty}"
                    / variant
                ),
                "--responses-root",
                str(responses_root / f"length_{difficulty}"),
                "--report-path",
                str(
                    condition_root
                    / "per_length_eval"
                    / "ckpt_final"
                    / f"length_{difficulty}.json"
                ),
                "--checkpoint",
                "ckpt_final",
                "--max-new-tokens",
                str(training["evaluation_max_new_tokens"]),
                "--batch-size",
                str(training["validation_batch_size"]),
                "--model",
                model,
                "--seed",
                str(seed),
                "--condition",
                "filler_only",
                "--length",
                str(difficulty),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "cell"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--task", choices=("multiplication", "knowledge", "s5", "parity"), required=True
    )
    parser.add_argument("--model", choices=("qwen", "llama"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action in {"train", "cell"}:
        train(args.config, config, args.task, args.model, args.seed)
    if args.action in {"evaluate", "cell"}:
        evaluate(config, args.task, args.model, args.seed)


if __name__ == "__main__":
    main()
