"""Run one recoverable training or short evaluation stage."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import *
from common import _training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "eval"])
    p.add_argument("task")
    p.add_argument("model")
    p.add_argument("seed", type=int)
    p.add_argument("--shard", type=Path)
    p.add_argument("--shards-root", type=Path)
    a = p.parse_args()
    config = load_config(CONFIG)
    spec = config["tasks"][a.task]
    settings = _training(spec, a.seed)
    root = condition(a.task, a.model, a.seed)
    if a.action == "train":
        from experiments.scaling_regime.protocols.filler.experiment import (
            _directories,
            _validate_cell_inputs,
            _write_provenance,
        )

        train_dir, eval_dir, manifest = _validate_cell_inputs(spec, a.seed)
        local = json.loads(json.dumps(config))
        local["tasks"][a.task]["artifacts_root"] = str(root.parents[2])
        # The existing helper appends model/seed/condition to artifacts_root.
        _write_provenance(
            CONFIG, local, a.task, a.model, a.seed, train_dir, eval_dir, manifest
        )
        total = spec["train_examples_per_difficulty"] * len(spec["difficulty_values"])
        steps = (
            (total + settings["batch_size"] - 1) // settings["batch_size"]
        ) * settings["epochs"]
        cmd = [
            sys.executable,
            "scripts/shared/training/run_ce.py",
            "--variant",
            spec["variant"],
            "--model",
            config["models"][a.model],
            "--output-root",
            str(root),
            "--train-prompts-dir",
            str(train_dir),
            "--supervision-mode",
            "filler_only",
            "--filler-token-count-field",
            spec["difficulty_field"],
            "--filler-token-counts-json",
            json.dumps(spec["filler_token_counts"]),
            "--expected-train-prompts",
            str(total),
            "--seed",
            str(a.seed),
            "--deterministic-training",
            "--save-every",
            str(steps),
            "--eval-every",
            str(steps),
            "--max-new-tokens",
            str(settings["training_max_new_tokens"]),
            "--recovery-state-path",
            str(cell(a.task, a.model, a.seed) / "state/training_state.pt"),
        ]
        for key in [
            "epochs",
            "batch_size",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]:
            cmd += ["--" + key.replace("_", "-"), str(settings[key])]
        if settings["memory_efficient_ce"]:
            cmd += [
                "--memory-efficient-ce",
                "--ce-token-chunk-size",
                str(settings["ce_token_chunk_size"]),
            ]
        if settings["activation_cpu_offload"]:
            cmd += ["--activation-cpu-offload"]
        subprocess.run(cmd, check=True)
        checkpoint_valid(root, a.task, a.model, a.seed, config)
        return
    weight = checkpoint_valid(root, a.task, a.model, a.seed, config)
    if bool(a.shard) == bool(a.shards_root):
        raise ValueError("Specify exactly one of --shard and --shards-root")
    shards = (
        [a.shard]
        if a.shard
        else sorted(
            a.shards_root.iterdir(),
            key=lambda s: (
                int(s.name.split("_")[1]) if s.name.split("_")[1].isdigit() else 0,
                s.name,
            ),
        )
    )
    for index, shard in enumerate(shards):
        print(
            f"[Recovery evaluation] chunk={index+1}/{len(shards)} path={shard}",
            flush=True,
        )
        evaluate_shard(
            shard,
            root,
            spec,
            settings,
            weight,
            reset_model_cache=(index == 0),
            keep_model_cache=bool(a.shards_root),
        )
    if a.shards_root:
        from chain_of_lies.inference import clear_model_cache

        clear_model_cache()


def evaluate_shard(
    shard, root, spec, settings, weight, reset_model_cache=True, keep_model_cache=False
):
    binding = json.loads((shard / "binding.json").read_text())
    if binding.get("checkpoint_sha256") not in (None, weight):
        raise ValueError("Shard belongs to different checkpoint")
    binding["checkpoint_sha256"] = weight
    (shard / "binding.json").write_text(json.dumps(binding, indent=2))
    prompts = shard / "prompts"
    responses = shard / "responses"
    cap = settings["evaluation_max_new_tokens"]
    for name, checksum in binding["prompts"].items():
        if sha(prompts / name) != checksum:
            raise ValueError("Prompt changed: " + name)
        output = responses / name
        if output.exists():
            try:
                response_valid(output, Path(name).stem, cap)
            except (ValueError, TypeError, KeyError):
                quarantine = shard / "quarantine"
                quarantine.mkdir(exist_ok=True)
                output.replace(quarantine / name)
    from chain_of_lies.evaluation import run_variant_inference
    from chain_of_lies.evaluation.rewards import summarize_variant_results

    run_variant_inference(
        prompts,
        responses,
        model_id=str(root / spec["variant"] / "ckpt_final"),
        max_new_tokens=cap,
        temperature=0.0,
        do_sample=False,
        resume=True,
        batch_size=1,
        reset_model_cache=reset_model_cache,
        keep_model_cache=keep_model_cache,
    )
    for name in binding["prompts"]:
        response_valid(responses / name, Path(name).stem, cap)
    metrics = summarize_variant_results(prompts, responses)
    if (
        metrics["num_examples"] != len(binding["prompts"])
        or metrics["missing_responses"]
    ):
        raise ValueError("Incomplete shard")
    (shard / "complete.json").write_text(
        json.dumps({"checkpoint_sha256": weight, "metrics": metrics}, indent=2)
    )


if __name__ == "__main__":
    main()
