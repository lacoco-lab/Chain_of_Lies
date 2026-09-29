#!/usr/bin/env python3
"""Train a resumable curriculum track and evaluate every completed stage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> None:
    print("[curriculum]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _evaluate(
    *,
    experiment_dir: Path,
    data_root: Path,
    responses_root: Path,
    checkpoint: Path,
    task: str,
    max_new_tokens: int,
    batch_size: int,
) -> None:
    _run(
        [
            sys.executable,
            str(experiment_dir / "run_inference.py"),
            "--prompts-dir",
            str(data_root / "validation" / task),
            "--responses-dir",
            str(responses_root / task),
            "--model",
            str(checkpoint),
            "--max-new-tokens",
            str(max_new_tokens),
            "--batch-size",
            str(batch_size),
        ]
    )


def run_track(
    *,
    config_path: Path,
    track: str,
    data_root: Path,
    artifacts_root: Path,
    responses_root: Path,
    initial_adapter: Path | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_dir = config_path.parent
    repo_root = experiment_dir.parents[1]
    stages_key = "update_stages" if track == "update" else "goldreich_stages"
    stages = config[stages_key]
    if track == "goldreich" and initial_adapter is None:
        raise ValueError(
            "The Goldreich track must start from the completed update adapter."
        )
    if (
        initial_adapter is not None
        and not (initial_adapter / "adapter_config.json").exists()
    ):
        raise FileNotFoundError(f"Missing initial adapter: {initial_adapter}")

    track_artifacts = artifacts_root / track
    track_responses = responses_root / track
    entries: list[dict[str, Any]] = []
    current_adapter = initial_adapter
    for index, stage in enumerate(stages, start=1):
        adapter_used_for_initialization = current_adapter
        stage_name = str(stage["name"])
        stage_label = f"{index:02d}_{stage_name}"
        stage_parent = track_artifacts / "stages" / stage_label
        train_dir = data_root / "curriculum" / track / stage_name
        command = [
            sys.executable,
            str(repo_root / "scripts/shared/training/run_ce.py"),
            "--variant",
            stage_name,
            "--model",
            str(config["model"]),
            "--output-root",
            str(stage_parent),
            "--train-prompts-dir",
            str(train_dir),
            "--epochs",
            str(stage["epochs"]),
            "--batch-size",
            str(config["batch_size"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--lora-r",
            str(config["lora_r"]),
            "--lora-alpha",
            str(config["lora_alpha"]),
            "--lora-dropout",
            str(config["lora_dropout"]),
            "--max-new-tokens",
            str(config["max_new_tokens"]),
            "--save-every",
            str(config["curriculum_train_n"]),
            "--expected-train-prompts",
            str(config["curriculum_train_n"]),
            "--supervision-mode",
            "record_target",
            "--seed",
            str(config["seed"]),
            "--save-each-epoch",
            "--deterministic-training",
        ]
        if current_adapter is not None:
            command.extend(["--initial-adapter-path", str(current_adapter)])
        _run(command)
        if (
            adapter_used_for_initialization is not None
            and track_artifacts in adapter_used_for_initialization.parents
        ):
            old_weights = adapter_used_for_initialization / "adapter_model.safetensors"
            if old_weights.exists():
                old_weights.unlink()
        stage_output = stage_parent / stage_name
        evaluation_task = str(stage["evaluation_task"])
        for epoch in range(1, int(stage["epochs"]) + 1):
            epoch_checkpoint = stage_output / f"ckpt_epoch_{epoch}"
            response_dir = track_responses / "stages" / stage_label / f"epoch_{epoch}"
            _evaluate(
                experiment_dir=experiment_dir,
                data_root=data_root,
                responses_root=response_dir,
                checkpoint=epoch_checkpoint,
                task=evaluation_task,
                max_new_tokens=int(config["max_new_tokens"]),
                batch_size=int(config["inference_batch_size"]),
            )
            entries.append(
                {
                    "stage_index": index,
                    "stage": stage_name,
                    "epoch": epoch,
                    "stage_epochs": int(stage["epochs"]),
                    "training_directory": str(train_dir),
                    "checkpoint": str(epoch_checkpoint),
                    "evaluation_task": evaluation_task,
                    "responses": str(response_dir / evaluation_task),
                }
            )
            # The scored responses and metadata are sufficient for analysis. Keep only the final
            # stage adapter used for continuation, avoiding tens of gigabytes of duplicate LoRA
            # weights in Condor output transfer.
            adapter_weights = epoch_checkpoint / "adapter_model.safetensors"
            if adapter_weights.exists():
                adapter_weights.unlink()
        current_adapter = stage_output / "ckpt_final"

    final_adapter = track_artifacts / "final_adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(current_adapter, final_adapter)
    last_stage_weights = current_adapter / "adapter_model.safetensors"
    if last_stage_weights.exists():
        last_stage_weights.unlink()

    if track == "update":
        final_tasks = [
            "local_update",
            *[f"supplied_{length}" for length in config["lengths"]],
        ]
    else:
        final_tasks = [
            "local_update",
            "predicate_local",
            f"supplied_{config['max_mask_bits']}",
            *[f"mask_{length}" for length in config["lengths"]],
            *[f"joint_{length}" for length in config["lengths"]],
        ]
    for task in final_tasks:
        _evaluate(
            experiment_dir=experiment_dir,
            data_root=data_root,
            responses_root=track_responses / "final",
            checkpoint=final_adapter,
            task=task,
            max_new_tokens=int(config["max_new_tokens"]),
            batch_size=int(config["inference_batch_size"]),
        )

    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "track": track,
        "seed": config["seed"],
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "final_adapter": str(final_adapter),
        "stages": entries,
        "final_evaluation_tasks": final_tasks,
    }
    track_artifacts.mkdir(parents=True, exist_ok=True)
    (track_artifacts / "track_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--track", choices=("update", "goldreich"), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_curriculum/seed_0"),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_curriculum"),
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_curriculum_responses/seed_0"),
    )
    parser.add_argument("--initial-adapter", type=Path, default=None)
    args = parser.parse_args()
    run_track(
        config_path=args.config,
        track=args.track,
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        responses_root=args.responses_root,
        initial_adapter=args.initial_adapter,
    )


if __name__ == "__main__":
    main()
