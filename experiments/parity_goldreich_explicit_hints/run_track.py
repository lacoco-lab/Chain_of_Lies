#!/usr/bin/env python3
"""Continue one LoRA adapter through an explicit-hint curriculum track."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> None:
    print("[explicit-hints]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _evaluate(
    experiment_dir: Path, data_root: Path, responses_root: Path,
    checkpoint: Path, task: str, max_new_tokens: int, batch_size: int,
) -> None:
    _run([
        sys.executable, str(experiment_dir / "run_inference.py"),
        "--prompts-dir", str(data_root / "validation" / task),
        "--responses-dir", str(responses_root / task),
        "--model", str(checkpoint), "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
    ])


def run_track(
    config_path: Path, track: str, data_root: Path, artifacts_root: Path,
    responses_root: Path, initial_adapter: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_dir, repo_root = config_path.parent, config_path.parent.parents[1]
    stages = config[f"{track}_stages"]
    if not (initial_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing initial adapter: {initial_adapter}")
    track_artifacts, track_responses = artifacts_root / track, responses_root / track
    current_adapter = initial_adapter
    entries: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        previous_adapter = current_adapter
        name, label = str(stage["name"]), f"{index:02d}_{stage['name']}"
        stage_parent = track_artifacts / "stages" / label
        train_dir = data_root / "curriculum" / track / name
        command = [
            sys.executable, str(repo_root / "scripts/run_ce_only.py"),
            "--variant", name, "--model", str(config["model"]),
            "--output-root", str(stage_parent), "--train-prompts-dir", str(train_dir),
            "--epochs", str(stage["epochs"]), "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]), "--lora-r", str(config["lora_r"]),
            "--lora-alpha", str(config["lora_alpha"]), "--lora-dropout", str(config["lora_dropout"]),
            "--max-new-tokens", str(config["max_new_tokens"]),
            "--save-every", str(config["curriculum_train_n"]),
            "--expected-train-prompts", str(config["curriculum_train_n"]),
            "--supervision-mode", "record_target", "--seed", str(config["seed"]),
            "--initial-adapter-path", str(current_adapter),
            "--save-each-epoch", "--deterministic-training",
        ]
        _run(command)
        if track_artifacts in previous_adapter.parents:
            old_weights = previous_adapter / "adapter_model.safetensors"
            if old_weights.exists():
                old_weights.unlink()
        stage_output, task = stage_parent / name, str(stage["evaluation_task"])
        for epoch in range(1, int(stage["epochs"]) + 1):
            checkpoint = stage_output / f"ckpt_epoch_{epoch}"
            response_dir = track_responses / "stages" / label / f"epoch_{epoch}"
            _evaluate(experiment_dir, data_root, response_dir, checkpoint, task,
                      int(config["max_new_tokens"]), int(config["inference_batch_size"]))
            entries.append({
                "stage_index": index, "stage": name, "epoch": epoch,
                "stage_epochs": int(stage["epochs"]), "evaluation_task": task,
                "responses": str(response_dir / task),
            })
            weights = checkpoint / "adapter_model.safetensors"
            if weights.exists():
                weights.unlink()
        current_adapter = stage_output / "ckpt_final"
    final_adapter = track_artifacts / "final_adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(current_adapter, final_adapter)
    last_weights = current_adapter / "adapter_model.safetensors"
    if last_weights.exists():
        last_weights.unlink()
    lengths = list(map(int, config["core_lengths"]))
    if track == "extension":
        lengths += list(map(int, config["extension_lengths"]))
    final_tasks = ["predicate_local", "hinted_mask_local", "hinted_start_local", "hinted_update_local"] + [
        f"hinted_joint_{length}" for length in lengths
    ]
    for task in final_tasks:
        _evaluate(experiment_dir, data_root, track_responses / "final", final_adapter, task,
                  int(config["max_new_tokens"]), int(config["inference_batch_size"]))
    manifest = {
        "schema_version": 1, "experiment_name": config["experiment_name"], "track": track,
        "seed": config["seed"], "initial_adapter": str(initial_adapter),
        "final_adapter": str(final_adapter), "stages": entries,
        "final_evaluation_tasks": final_tasks,
    }
    track_artifacts.mkdir(parents=True, exist_ok=True)
    (track_artifacts / "track_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--track", choices=("core", "extension"), required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path("generated_data/parity_goldreich_explicit_hints/seed_0"))
    parser.add_argument("--artifacts-root", type=Path,
                        default=Path("artifacts/ce_parity_goldreich_explicit_hints"))
    parser.add_argument("--responses-root", type=Path,
                        default=Path("generated_data/parity_goldreich_explicit_hints_responses/seed_0"))
    parser.add_argument("--initial-adapter", type=Path, required=True)
    args = parser.parse_args()
    run_track(args.config, args.track, args.data_root, args.artifacts_root,
              args.responses_root, args.initial_adapter)


if __name__ == "__main__":
    main()
