"""Replace unstarted evaluation nodes, without interrupting active work."""

import argparse
import os
import json
import shlex
import subprocess
import sys
from pathlib import Path
from common import *

COMPACT = WORK / "compact"


def query(controller):
    fields = "ClusterId,ProcId,JobStatus,NumJobStarts,Args,DAGManJobId,Cmd"
    raw = subprocess.check_output(
        [
            "condor_q",
            "-constraint",
            f"DAGManJobId == {controller} || ClusterId == {controller}",
            "-json",
            "-attributes",
            fields,
        ],
        text=True,
    )
    return json.loads(raw)


def identity(row):
    args = shlex.split(row.get("Args", ""))
    if (
        len(args) >= 5
        and args[0] == "experiments/scaling_regime/protocols/filler/recovery_sic/run.sh"
        and args[1] in ["train", "eval"]
    ):
        return args[1], args[2], args[3], int(args[4])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", type=int, required=True)
    p.add_argument(
        "--apply",
        action="store_true",
        help="Hold old controller and cancel ONLY never-started idle evaluations",
    )
    a = p.parse_args()
    if a.apply and not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN must be exported before applying the transition; no jobs changed"
        )
    if COMPACT.exists():
        raise ValueError(f"{COMPACT} already exists; do not repeat this transition")
    before = query(a.controller)
    controller = [
        r for r in before if r["ClusterId"] == a.controller and r["ProcId"] == 0
    ]
    if len(controller) != 1:
        raise ValueError("Old DAG controller not found; refusing transition")
    if Path(controller[0].get("Cmd", "")).name != "condor_dagman":
        raise ValueError(
            "Specified job is not a DAGMan controller; refusing transition"
        )
    expected = set(
        (r["task"], r["model"], r["seed"])
        for r in json.loads((WORK / "plan.json").read_text())
    )
    for row in before:
        if row["ClusterId"] == a.controller:
            continue
        ident = identity(row)
        if not ident or ident[1:] not in expected:
            raise ValueError("Unexpected old DAG node; refusing transition")
    preview = {
        "controller": a.controller,
        "preserve_training": [],
        "preserve_started_evaluation": [],
        "cancel_never_started_idle_evaluation": [],
    }
    for row in before:
        ident = identity(row)
        if not ident:
            continue
        label = f'{row["ClusterId"]}.{row["ProcId"]}'
        key = (
            "preserve_training"
            if ident[0] == "train"
            else (
                "cancel_never_started_idle_evaluation"
                if row["JobStatus"] == 1 and row.get("NumJobStarts") == 0
                else "preserve_started_evaluation"
            )
        )
        preview[key].append(label)
    print(json.dumps(preview, indent=2), flush=True)
    if not a.apply:
        print("Read-only preview. Rerun with --apply to perform the transition.")
        return
    if controller[0]["JobStatus"] != 5:
        subprocess.run(["condor_hold", f"{a.controller}.0"], check=False)
    rows = query(a.controller)
    if not any(r["ClusterId"] == a.controller and r["JobStatus"] == 5 for r in rows):
        raise ValueError("Controller is not held; refusing to cancel evaluation jobs")
    removed = []
    # Re-query each candidate to protect a job that started since the preview.
    for row in rows:
        ident = identity(row)
        if (
            not ident
            or ident[0] != "eval"
            or row["JobStatus"] != 1
            or row.get("NumJobStarts") != 0
        ):
            continue
        label = f'{row["ClusterId"]}.{row["ProcId"]}'
        subprocess.run(
            [
                "condor_rm",
                "-constraint",
                f'ClusterId == {row["ClusterId"]} && ProcId == {row["ProcId"]} && JobStatus == 1 && NumJobStarts == 0',
            ],
            check=True,
        )
        removed.append(label)
    rows = query(a.controller)
    config = load_config(CONFIG)
    plan = json.loads((WORK / "plan.json").read_text())
    COMPACT.mkdir()
    (COMPACT / "logs").mkdir()
    dag = []
    additional_training = []
    # Existing training keeps its exact job and optimizer state; submit only a
    # genuinely unqueued/missing training cell, never a duplicate active trainer.
    for item in plan:
        task, model, seed = item["task"], item["model"], item["seed"]
        prefix = f"{task}_{model}_{seed}"
        active_training = any(
            identity(r) == ("train", task, model, seed)
            and r["JobStatus"] in [1, 2, 4, 5, 6, 7]
            for r in rows
        )
        try:
            checkpoint_valid(condition(task, model, seed), task, model, seed, config)
            complete = True
        except (OSError, ValueError, KeyError, TypeError):
            complete = False
        parent = None
        if not active_training and not complete:
            parent = prefix + "_train"
            source = WORK / "submit" / f"{parent}.sub"
            if not source.exists():
                raise ValueError(
                    "Missing original training submit file: " + str(source)
                )
            content = source.read_text()
            for suffix in ["out", "err", "log"]:
                content = content.replace(
                    str(WORK / "logs" / f"{parent}.{suffix}"),
                    str(COMPACT / "logs" / f"{parent}.{suffix}"),
                )
            target = COMPACT / f"{parent}.sub"
            target.write_text(content)
            dag.append(f"JOB {parent} {target}")
            dag.append(f"PRIORITY {parent} 100")
            additional_training.append({"task": task, "model": model, "seed": seed})
        shards = cell(task, model, seed) / "shards"
        root = condition(task, model, seed)
        name = prefix + "_eval"
        target = COMPACT / f"{name}.sub"
        from experiments.scaling_regime.protocols.filler.experiment import (
            _source_config,
            _source_manifest,
        )

        spec = config["tasks"][task]
        inputs = [
            "chain_of_lies",
            "experiments/scaling_regime/protocols/filler",
            str(_source_config(spec, seed)),
            str(_source_manifest(spec, seed)),
            str(root),
            str(shards),
        ]
        target.write_text(f"""universe = docker
docker_image = pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
initialdir = /home/momo00016/Faithfulness-Safety
executable = /bin/bash
arguments = experiments/scaling_regime/protocols/filler/recovery_sic/run.sh eval {task} {model} {seed} --shards-root {shards}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = {','.join(dict.fromkeys(inputs))}
transfer_output_files = {shards}
environment = "HF_TOKEN=$ENV(HF_TOKEN)"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
request_cpus = 4
request_memory = 32G
request_disk = 40G
request_gpus = 1
requirements = (GPUs_GlobalMemoryMb >= 40000)
max_job_retirement_time = 28800
+MaxWallTime = 604800
output = {COMPACT}/logs/{name}.out
error = {COMPACT}/logs/{name}.err
log = {COMPACT}/logs/{name}.log
queue
""")
        dag.append(f"JOB {name} {target}")
        interpreter = Path(sys.executable).resolve()
        dag.append(
            f"SCRIPT DEFER 75 60 PRE {name} {interpreter} -S experiments/scaling_regime/protocols/filler/recovery_sic/ready.py --controller {a.controller} --task {task} --model {model} --seed {seed}"
        )
        if parent:
            dag.append(f"PARENT {parent} CHILD {name}")
    (COMPACT / "compact.dag").write_text("\n".join(dag) + "\n")
    record = {
        "old_controller": a.controller,
        "old_controller_held": True,
        "cancelled_unstarted_candidates": removed,
        "additional_training": additional_training,
        "evaluation_jobs": len(plan),
        "preserved_old_jobs": [
            f'{r["ClusterId"]}.{r["ProcId"]}' for r in rows if identity(r)
        ],
    }
    (COMPACT / "transition.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print("Submit:", COMPACT / "compact.dag")
    print("Do not release/remove the old controller until final audit succeeds.")


if __name__ == "__main__":
    main()
