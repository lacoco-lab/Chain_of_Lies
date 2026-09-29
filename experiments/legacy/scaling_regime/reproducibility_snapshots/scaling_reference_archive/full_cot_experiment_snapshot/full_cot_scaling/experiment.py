#!/usr/bin/env python3
"""Matched full-CoT baseline, isolated from all finalized experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))
from experiments.scaling_regime.protocols.filler.experiment import (
    load_config,
    _directories,
    _training,
    _training_complete,
    _source_config,
    _source_manifest,
    _validate_cell_inputs,
)
from experiments.scaling_regime.protocols.full_cot.protocol import derive, encoded

SOURCE_CONFIG = ROOT / "experiments/scaling_regime/protocols/full_cot/config.json"


def write_json(destination, value):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )


def prepare(config, task, model, seed):
    from chain_of_lies.training.shared.trainer_utils import (
        _public_cot_prefix,
        _answer_block_suffix,
    )
    from transformers import AutoTokenizer

    spec = config["tasks"][task]
    train, evaluation, manifest = _validate_cell_inputs(spec, seed)
    base = Path("generated_data/full_cot_scaling_v1") / task / model / f"seed_{seed}"
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"][model], token=os.environ.get("HF_TOKEN")
    )
    if not tokenizer(
        "A normal worked solution: 2 + 3 = 5.", add_special_tokens=False
    ).input_ids:
        raise RuntimeError(
            "Tokenizer returned no IDs; refusing invalid capacity verification"
        )
    maximum = 0
    source_hash = hashlib.sha256()
    derived_hash = hashlib.sha256()
    expected = {}
    for split, source in [("train", train), ("eval", evaluation)]:
        files = sorted(source.glob("*.json"))
        expected[split] = len(files)
        for file in files:
            raw = file.read_bytes()
            source_hash.update(split.encode() + file.name.encode() + raw)
            record = json.loads(raw)
            result = derive(record, _public_cot_prefix, _answer_block_suffix)
            text = encoded(result)
            derived_hash.update(split.encode() + file.name.encode() + text.encode())
            destination = base / split / file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                not destination.exists()
                or destination.read_text(encoding="utf-8") != text
            ):
                destination.write_text(text, encoding="utf-8")
            if split == "eval":
                difficulty = str(record["spec"][spec["difficulty_field"]])
                bucket = base / "per_length" / difficulty / file.name
                bucket.parent.mkdir(parents=True, exist_ok=True)
                if not bucket.exists():
                    bucket.symlink_to(os.path.relpath(destination, bucket.parent))
            maximum = max(
                maximum,
                len(
                    tokenizer(
                        result["supervised_suffix"], add_special_tokens=False
                    ).input_ids
                ),
            )
        if len(list((base / split).glob("*.json"))) != len(files):
            raise RuntimeError(f"Unexpected stale derived records at {base / split}")
    settings = _training(spec, seed)
    if maximum >= min(
        settings["training_max_new_tokens"], settings["evaluation_max_new_tokens"]
    ):
        raise RuntimeError(
            f"Full-CoT target needs {maximum} tokens; configured generation limit is insufficient. Do not run truncated evaluation."
        )
    provenance = {
        "experiment": "full_cot_scaling_v1",
        "task": task,
        "model": model,
        "seed": seed,
        "source_manifest": str(manifest),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_config": str(_source_config(spec, seed)),
        "source_settings": settings,
        "source_records_sha256": source_hash.hexdigest(),
        "derived_records_sha256": derived_hash.hexdigest(),
        "counts": expected,
        "max_gold_continuation_tokens": maximum,
        "changes": [
            "visible public AND private CoT target",
            "remove concealment instructions at train AND eval",
        ],
        "base_model": config["models"][model],
        "primary_metric": "private_exact_rate",
        "concealment_metric_applicable": False,
    }
    return base, provenance


def cell(config, task, model, seed):
    import torch
    from chain_of_lies.evaluation.experiment_evaluation import run_variant_inference
    from chain_of_lies.evaluation.rewards import summarize_variant_results
    from chain_of_lies.inference import clear_model_cache

    spec = config["tasks"][task]
    settings = _training(spec, seed)
    data, provenance = prepare(config, task, model, seed)
    condition = (
        Path("artifacts/full_cot_scaling_v1")
        / task
        / model
        / f"seed_{seed}"
        / "full_cot"
    )
    adapter = condition / spec["variant"]
    provenance["runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(condition / "provenance.json", provenance)
    if not _training_complete(adapter):
        if adapter.exists() and any(adapter.iterdir()):
            raise RuntimeError(
                f"Incomplete training at {adapter}; preserve/move it before retrying, do not silently resume."
            )
        steps = provenance["counts"]["train"] * settings["epochs"]
        args = [
            sys.executable,
            "scripts/shared/training/run_ce.py",
            "--variant",
            spec["variant"],
            "--model",
            config["models"][model],
            "--output-root",
            str(condition),
            "--train-prompts-dir",
            str(data / "train"),
            "--supervision-mode",
            "record_target",
            "--seed",
            str(seed),
            "--deterministic-training",
            "--save-every",
            str(steps),
            "--eval-every",
            str(steps),
            "--expected-train-prompts",
            str(provenance["counts"]["train"]),
        ]
        for key in [
            "epochs",
            "batch_size",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]:
            args += ["--" + key.replace("_", "-"), str(settings[key])]
        args += [
            "--max-new-tokens",
            str(settings["training_max_new_tokens"]),
            "--validation-batch-size",
            str(settings["validation_batch_size"]),
        ]
        if settings.get("run_training_validation", True):
            args += [
                "--val-prompts-dir",
                str(data / "eval"),
                "--validation-sample-size",
                str(settings["validation_sample_size"]),
                "--expected-val-prompts",
                str(provenance["counts"]["eval"]),
            ]
        if settings.get("memory_efficient_ce"):
            args += [
                "--memory-efficient-ce",
                "--ce-token-chunk-size",
                str(settings["ce_token_chunk_size"]),
            ]
        if settings.get("activation_cpu_offload"):
            args += ["--activation-cpu-offload"]
        subprocess.run(args, check=True)
    if not _training_complete(adapter):
        raise RuntimeError("Training incomplete; refusing evaluation")
    # One model load across ALL lengths, with no redundant base-model inference pass.
    response_root = (
        Path("generated_data/full_cot_scaling_v1_eval_responses")
        / task
        / model
        / f"seed_{seed}"
    )
    all_responses = response_root / "all"
    batch = 2
    while True:
        try:
            run_variant_inference(
                data / "eval",
                all_responses,
                model_id=str(adapter / "ckpt_final"),
                max_new_tokens=settings["evaluation_max_new_tokens"],
                temperature=0.0,
                do_sample=False,
                resume=True,
                batch_size=batch,
            )
            break
        except torch.cuda.OutOfMemoryError:
            clear_model_cache()
            if batch == 1:
                raise
            batch = 1
            print(
                "[Full-CoT] evaluation OOM: retry pending examples at batch size 1",
                flush=True,
            )
    for difficulty in spec["difficulty_values"]:
        prompts = data / "per_length" / str(difficulty)
        responses = response_root / f"length_{difficulty}"
        responses.mkdir(parents=True, exist_ok=True)
        for prompt_file in prompts.glob("*.json"):
            experiment_id = json.loads(prompt_file.read_text())["experiment_id"]
            original = all_responses / f"{experiment_id}.json"
            view = responses / original.name
            if not original.is_file():
                raise RuntimeError(f"Missing evaluated response: {original}")
            if not view.exists():
                view.symlink_to(os.path.relpath(original, view.parent))
        metrics = summarize_variant_results(prompts, responses)
        expected = spec["eval_examples_per_difficulty"]
        if (
            metrics.get("missing_responses")
            or int(metrics.get("num_examples", 0)) != expected
        ):
            raise RuntimeError(f"Incomplete evaluation at length {difficulty}")
        report = {
            "task": task,
            "model": model,
            "seed": seed,
            "condition": "full_cot",
            "length": difficulty,
            "metrics": metrics,
            "evaluation_batch_size": batch,
            "concealment_metric_applicable": False,
        }
        write_json(
            condition / "per_length_eval" / "ckpt_final" / f"length_{difficulty}.json",
            report,
        )
        print(
            "[Full-CoT]",
            task,
            model,
            seed,
            difficulty,
            "private_exact=",
            metrics["private_exact_rate"],
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "cell"])
    parser.add_argument(
        "--task", choices=["multiplication", "knowledge", "s5", "parity"], required=True
    )
    parser.add_argument("--model", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(SOURCE_CONFIG)
    if args.action == "check":
        from experiments.scaling_regime.protocols.filler.preflight import _data_report

        print(
            json.dumps(
                _data_report(config, args.task, validate_targets=False), indent=2
            )
        )
    else:
        cell(config, args.task, args.model, args.seed)


if __name__ == "__main__":
    main()
