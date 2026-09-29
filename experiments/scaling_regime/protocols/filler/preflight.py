#!/usr/bin/env python3
"""Verify unchanged inputs and Filler-only capacity before cluster submission."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.scaling_regime.protocols.filler.experiment import (  # noqa: E402
    _directories,
    _source_manifest,
    _source_config,
    _task,
    _validate_cell_inputs,
    load_config,
)


def _private_as_public(record: dict[str, Any]) -> dict[str, Any]:
    swapped = copy.deepcopy(record)
    spec = swapped.setdefault("spec", {})
    for stem in (
        "question",
        "answer",
        "operands",
        "facts",
        "fact",
        "instruction_sequence",
        "bits",
        "initial_state",
    ):
        private_key = f"private_{stem}"
        if private_key in spec:
            spec[f"public_{stem}"] = copy.deepcopy(spec[private_key])
    return swapped


def _records(task_spec: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for seed in seeds:
        train_dir, eval_dir, _ = _validate_cell_inputs(task_spec, seed)
        paths.update(train_dir.glob("*.json"))
        paths.update(eval_dir.glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def _data_report(
    config: dict[str, Any],
    task_name: str,
    selected_seeds: list[int] | None = None,
    *,
    validate_targets: bool = True,
) -> dict[str, Any]:
    if validate_targets:
        from chain_of_lies.training.shared.trainer_utils import (
            FILLER_TOKEN_MARKER,
            _canonical_public_cot_suffix,
        )

    task_spec = _task(config, task_name)
    seeds = selected_seeds or [int(seed) for seed in config["seeds"]]
    cells = []
    for seed in seeds:
        train_dir, eval_dir, manifest = _validate_cell_inputs(task_spec, seed)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        if not bool(
            manifest_data.get(
                "private_questions_matched_across_all_five_conditions", True
            )
        ):
            raise RuntimeError(
                f"Source manifest does not certify matched private questions: {manifest}"
            )
        source_config_path = _source_config(task_spec, seed)
        if not source_config_path.is_file():
            raise FileNotFoundError(
                f"Missing original experiment config: {source_config_path}"
            )
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        if source_config.get("models") != config.get("models"):
            raise RuntimeError(
                f"Model IDs drift from the source experiment: {source_config_path}"
            )
        source_training = source_config["training"]
        current_training = dict(task_spec["training"])
        current_training.update(
            task_spec.get("seed_training_overrides", {}).get(str(seed), {})
        )
        defaults = {
            "memory_efficient_ce": False,
            "activation_cpu_offload": False,
            "run_training_validation": True,
            "evaluation_batch_size": source_training.get("validation_batch_size", 1),
            "ce_token_chunk_size": 16,
        }
        scientific_keys = (
            "epochs",
            "batch_size",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "training_max_new_tokens",
            "evaluation_max_new_tokens",
            "validation_sample_size",
            "validation_batch_size",
            "memory_efficient_ce",
            "ce_token_chunk_size",
            "activation_cpu_offload",
            "run_training_validation",
            "evaluation_batch_size",
        )
        for key in scientific_keys:
            source_value = source_training.get(key, defaults.get(key))
            current_value = current_training.get(key, defaults.get(key))
            if source_value != current_value:
                raise RuntimeError(
                    f"Scientific/operational setting drift for {task_name}/seed_{seed}/{key}: "
                    f"source={source_value!r}, filler_only={current_value!r}."
                )
        cells.append(
            {
                "seed": seed,
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "manifest": str(manifest),
                "train_count": len(list(train_dir.glob("*.json"))),
                "eval_count": len(list(eval_dir.glob("*.json"))),
                "source_config": str(source_config_path),
            }
        )
    records = _records(task_spec, seeds)
    expected_keys = {str(value) for value in task_spec["difficulty_values"]}
    actual_keys: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        spec = record.get("spec") or {}
        key = str(spec.get(task_spec["difficulty_field"]))
        actual_keys.add(key)
        counts[key] += 1
        if validate_targets:
            trial = copy.deepcopy(record)
            trial["spec"]["filler_token_count"] = int(
                task_spec["filler_token_counts"][key]
            )
            suffix = _canonical_public_cot_suffix(trial, supervision_mode="filler_only")
            if suffix is None or suffix.count(FILLER_TOKEN_MARKER) != 1:
                raise RuntimeError(
                    f"Could not construct one Filler-only marker for {record.get('experiment_id')}."
                )
            before_answer = suffix.split("<ANSWER>", 1)[0]
            if before_answer != FILLER_TOKEN_MARKER + "\n":
                raise RuntimeError(
                    f"Filler-only target contains non-filler reasoning for {record.get('experiment_id')}."
                )
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"Difficulty values differ: expected {sorted(expected_keys)}, found {sorted(actual_keys)}."
        )
    return {
        "task": task_name,
        "cells": cells,
        "unique_records": len(records),
        "records_by_difficulty": dict(counts),
    }


def _capacity_report(
    config: dict[str, Any],
    task_name: str,
    model_names: list[str] | None = None,
    selected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from chain_of_lies.training.shared.trainer_utils import _public_cot_prefix

    task_spec = _task(config, task_name)
    records = _records(
        task_spec, selected_seeds or [int(seed) for seed in config["seeds"]]
    )
    prefixes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        prefix = _public_cot_prefix(_private_as_public(record))
        if prefix is None:
            raise RuntimeError(
                f"Could not construct canonical private CoT for {record.get('experiment_id')}."
            )
        key = str((record.get("spec") or {}).get(task_spec["difficulty_field"]))
        prefixes[key].append(prefix)

    maxima: dict[str, dict[str, int]] = {}
    token = os.environ.get("HF_TOKEN")
    selected_models = model_names or list(config["models"])
    for model_name in selected_models:
        model_id = config["models"][model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, trust_remote_code=True
        )
        filler_ids = tokenizer(
            config["filler_token_text"], add_special_tokens=False
        ).input_ids
        if len(filler_ids) != 1:
            raise RuntimeError(
                f"{model_name}: filler text must map to exactly one token, got {filler_ids}."
            )
        model_maxima: dict[str, int] = {}
        for difficulty, texts in prefixes.items():
            maximum = 0
            for start in range(0, len(texts), 256):
                encoded = tokenizer(
                    texts[start : start + 256], add_special_tokens=False
                ).input_ids
                maximum = max(maximum, *(len(ids) for ids in encoded))
            model_maxima[difficulty] = maximum
        maxima[model_name] = model_maxima

    verification = {}
    for difficulty, budget_value in task_spec["filler_token_counts"].items():
        required = max(maxima[model][difficulty] for model in maxima)
        budget = int(budget_value)
        if budget < required:
            raise RuntimeError(
                f"{task_name}/{difficulty}: filler budget {budget} is below private-CoT maximum {required}."
            )
        verification[difficulty] = {
            "budget": budget,
            "required_maximum": required,
            "slack": budget - required,
        }
    return {"max_private_cot_tokens_by_model": maxima, "verified_budgets": verification}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=("multiplication", "knowledge", "s5", "parity"),
    )
    parser.add_argument("--model", action="append", choices=("qwen", "llama"))
    parser.add_argument("--seed", action="append", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Skip tokenizer downloads and capacity verification.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = args.task or list(config["tasks"])
    report = {"schema_version": 1, "experiment": config["experiment_name"], "tasks": {}}
    for task_name in tasks:
        task_report = _data_report(
            config, task_name, args.seed, validate_targets=not args.data_only
        )
        if not args.data_only:
            task_report["capacity"] = _capacity_report(
                config, task_name, args.model, args.seed
            )
        report["tasks"][task_name] = task_report
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
