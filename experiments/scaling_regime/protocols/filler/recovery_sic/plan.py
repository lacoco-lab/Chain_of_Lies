"""Validate snapshots and prepare isolated, eviction-resumable DAG jobs."""

import argparse
import json
import shutil
from pathlib import Path
from common import *
from common import _training
from capture import JOBS
from experiments.scaling_regime.protocols.filler.experiment import (
    _directories,
    _source_config,
    _source_manifest,
    _validate_cell_inputs,
)
from chain_of_lies.evaluation.rewards import summarize_variant_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Validate all eight sources and print status without creating work",
    )
    p.add_argument(
        "--retrain-job",
        action="append",
        choices=list(JOBS),
        default=[],
        help="Explicitly permit fresh training only for this missing checkpoint",
    )
    a = p.parse_args()
    config = load_config(CONFIG)
    if WORK.exists() and not a.inspect:
        raise ValueError(f"{WORK} exists; do not overwrite a recovery plan")
    capture = json.loads((a.snapshot / "capture.json").read_text())
    rows = {row["job"]: row for row in capture["cells"]}
    selected = []
    blocked = []
    # Validate all sources BEFORE creating any work or authorizing old job removal.
    for job, (task, model, seed) in JOBS.items():
        _validate_cell_inputs(config["tasks"][task], seed)
        source = None
        weight = None
        errors = []
        candidates = []
        if rows[job]["stable"]:
            candidates.append(a.snapshot / job / canonical_condition(task, model, seed))
        candidates.append(canonical_condition(task, model, seed))
        for candidate in candidates:
            try:
                weight = checkpoint_valid(candidate, task, model, seed, config)
                source = candidate
                break
            except (OSError, ValueError, KeyError, TypeError) as e:
                errors.append(str(e))
        if source is None and job not in a.retrain_job:
            blocked.append(job)
        selected.append((job, task, model, seed, source, weight, errors))
    if a.inspect:
        inspection = []
        for job, task, model, seed, source, weight, errors in selected:
            spec = config["tasks"][task]
            valid = 0
            invalid = 0
            if source:
                base = (
                    canonical_responses(task, model, seed)
                    if source == canonical_condition(task, model, seed)
                    else a.snapshot / job / canonical_responses(task, model, seed)
                )
                folders = (
                    [base / "ckpt_final/finetuned" / spec["variant"]]
                    if task == "parity"
                    else [
                        base / f"length_{length}/ckpt_final/trained" / spec["variant"]
                        for length in spec["difficulty_values"]
                    ]
                )
                for folder in folders:
                    for file in folder.glob("*.json"):
                        try:
                            response_valid(
                                file,
                                file.stem,
                                _training(spec, seed)["evaluation_max_new_tokens"],
                            )
                            valid += 1
                        except (OSError, ValueError, KeyError, TypeError):
                            invalid += 1
            inspection.append(
                {
                    "job": job,
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "validated_final_checkpoint": bool(source),
                    "checkpoint_source": str(source) if source else None,
                    "checkpoint_sha256": weight,
                    "valid_saved_responses": valid,
                    "invalid_saved_responses": invalid,
                    "expected_responses": spec["eval_examples_per_difficulty"]
                    * len(spec["difficulty_values"]),
                    "checkpoint_errors": errors,
                }
            )
        print(
            json.dumps(
                {
                    "cells": inspection,
                    "missing_final_checkpoints": blocked,
                    "work_created": False,
                },
                indent=2,
            )
        )
        return
    if blocked:
        raise ValueError(
            "No complete checkpoint for jobs "
            + ", ".join(blocked)
            + ". Run --inspect to see ALL captured cells. Do not bypass failed captures with fresh training."
        )
    WORK.mkdir(parents=True)
    submits = WORK / "submit"
    submits.mkdir()
    logs = WORK / "logs"
    logs.mkdir()
    dag = []
    report = []

    def submit(name, arguments, inputs, output, memory):
        path = submits / (name + ".sub")
        text = f"""universe = docker
docker_image = pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
initialdir = /home/momo00016/Faithfulness-Safety
executable = /bin/bash
arguments = experiments/scaling_regime/protocols/filler/recovery_sic/run.sh {arguments}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = {','.join(dict.fromkeys(map(str,inputs)))}
transfer_output_files = {output}
environment = "HF_TOKEN=$ENV(HF_TOKEN)"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
request_cpus = 4
request_memory = {memory}G
request_disk = 40G
request_gpus = 1
requirements = (GPUs_GlobalMemoryMb >= 40000)
max_job_retirement_time = 28800
+MaxWallTime = 604800
output = {logs}/{name}.out
error = {logs}/{name}.err
log = {logs}/{name}.log
queue
"""
        path.write_text(text)
        dag.extend([f"JOB {name} {path}", f"CATEGORY {name} recovery"])

    for job, task, model, seed, source, weight, errors in selected:
        spec = config["tasks"][task]
        settings = _training(spec, seed)
        c = cell(task, model, seed)
        state = c / "state"
        state.mkdir(parents=True)
        root = condition(task, model, seed)
        if source:
            shutil.copytree(source, root)
        else:
            root.mkdir(parents=True)
        train, eval_dir = _directories(spec, seed)
        base = [
            "chain_of_lies",
            "scripts",
            "experiments/scaling_regime/protocols/filler",
            _source_config(spec, seed),
            _source_manifest(spec, seed),
        ]
        prefix = f"{task}_{model}_{seed}"
        if source is None:
            submit(
                prefix + "_train",
                f"train {task} {model} {seed}",
                base + [train, eval_dir, state],
                state,
                128 if task == "parity" else 64,
            )
        groups = []
        if task == "s5":
            for length in spec["difficulty_values"]:
                folder = (
                    Path(spec["data_root"])
                    / "shared_eval"
                    / f"length_{length}"
                    / spec["variant"]
                )
                groups.append((str(length), sorted(folder.glob("*.json"))))
        else:
            groups = [("all", sorted(eval_dir.glob("*.json")))]
        copied = 0
        nodes = 0
        for length, files in groups:
            expected = (
                spec["eval_examples_per_difficulty"]
                if task == "s5"
                else spec["eval_examples_per_difficulty"]
                * len(spec["difficulty_values"])
            )
            if len(files) != expected:
                raise ValueError(
                    f"Wrong eval count {task}/{length}: {len(files)} != {expected}"
                )
            for part, start in enumerate(range(0, len(files), 100)):
                shard = c / "shards" / f"length_{length}_part_{part}"
                (shard / "prompts").mkdir(parents=True)
                (shard / "responses").mkdir()
                binding = {
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "length": length,
                    "checkpoint_sha256": weight,
                    "prompts": {},
                }
                for file in files[start : start + 100]:
                    shutil.copy2(file, shard / "prompts" / file.name)
                    binding["prompts"][file.name] = sha(file)
                    if source:
                        # Reuse responses only from the same validated snapshot/checkpoint.
                        if source == canonical_condition(task, model, seed):
                            response_base = canonical_responses(task, model, seed)
                        else:
                            response_base = (
                                a.snapshot
                                / job
                                / canonical_responses(task, model, seed)
                            )
                        raw = (
                            response_base
                            / (
                                f'length_{length}/ckpt_final/trained/{spec["variant"]}'
                                if task == "s5"
                                else f'ckpt_final/finetuned/{spec["variant"]}'
                            )
                            / file.name
                        )
                        try:
                            response_valid(
                                raw, file.stem, settings["evaluation_max_new_tokens"]
                            )
                            shutil.copy2(raw, shard / "responses" / file.name)
                            copied += 1
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                (shard / "binding.json").write_text(json.dumps(binding, indent=2))
                if weight and all(
                    (shard / "responses" / name).exists() for name in binding["prompts"]
                ):
                    metrics = summarize_variant_results(
                        shard / "prompts", shard / "responses"
                    )
                    if (
                        metrics["num_examples"] != len(binding["prompts"])
                        or metrics["missing_responses"]
                    ):
                        raise ValueError("Imported shard scoring incomplete")
                    (shard / "complete.json").write_text(
                        json.dumps(
                            {"checkpoint_sha256": weight, "metrics": metrics}, indent=2
                        )
                    )
                    continue  # Fully captured chunks need no GPU job.
                name = prefix + f"_l{length}_p{part}"
                submit(
                    name,
                    f"eval {task} {model} {seed} --shard {shard}",
                    base + [root, shard],
                    shard,
                    32,
                )
                if source is None:
                    dag.append(f"PARENT {prefix}_train CHILD {name}")
                nodes += 1
        report.append(
            {
                "job": job,
                "task": task,
                "model": model,
                "seed": seed,
                "checkpoint_source": str(source) if source else None,
                "checkpoint_sha256": weight,
                "reused_responses": copied,
                "shards": nodes,
                "validation_errors": errors,
            }
        )
    dag.append("MAXJOBS recovery 12")
    (WORK / "recovery.dag").write_text("\n".join(dag) + "\n")
    (WORK / "plan.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Plan complete. No old jobs stopped. DAG:", WORK / "recovery.dag")


if __name__ == "__main__":
    main()
