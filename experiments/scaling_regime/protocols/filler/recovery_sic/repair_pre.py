"""Repair failed PRE launches without removing/restarting any training node."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from compact import COMPACT, query, identity
from common import *


def ensure_held(controller):
    def held(rows):
        return any(
            r["ClusterId"] == controller and r["ProcId"] == 0 and r["JobStatus"] == 5
            for r in rows
        )

    if held(query(controller)):
        print(
            f"Controller {controller}.0 already held; leaving it unchanged.", flush=True
        )
        return
    # Some sites return 1 if another operation has already held the job.
    result = subprocess.run(["condor_hold", f"{controller}.0"], check=False)
    if not held(query(controller)):
        raise ValueError(
            f"Controller {controller}.0 hold not confirmed (exit {result.returncode}); no replacement submitted"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed-controller", type=int, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    target = COMPACT / "pre_fixed"
    if target.exists():
        raise ValueError("Repair already generated; do not repeat or submit twice")
    transition = json.loads((COMPACT / "transition.json").read_text())
    controllers = [transition["old_controller"], a.failed_controller]
    if len(set(controllers)) != 2:
        raise ValueError("Expected two different controller IDs")
    rows = []
    for controller in controllers:
        current = query(controller)
        rows.extend(current)
        manager = [
            r for r in current if r["ClusterId"] == controller and r["ProcId"] == 0
        ]
        if len(manager) != 1 or Path(manager[0].get("Cmd", "")).name != "condor_dagman":
            raise ValueError(f"{controller} is not an existing DAGMan controller")
    config = load_config(CONFIG)
    plan = json.loads((WORK / "plan.json").read_text())
    training = []
    for item in plan:
        task, model, seed = item["task"], item["model"], item["seed"]
        active = [
            r
            for r in rows
            if identity(r) == ("train", task, model, seed)
            and r["JobStatus"] in [1, 2, 4, 5, 6, 7]
        ]
        if len(active) > 1:
            raise ValueError("Multiple trainers for same cell; refusing repair")
        try:
            checkpoint_valid(condition(task, model, seed), task, model, seed, config)
            valid = True
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not active and not valid:
            raise ValueError(
                f"No queued trainer or valid checkpoint for {task}/{model}/{seed}; inspect before repair"
            )
        if active:
            training.append(f'{active[0]["ClusterId"]}.{active[0]["ProcId"]}')
        if not (COMPACT / f"{task}_{model}_{seed}_eval.sub").is_file():
            raise ValueError("Missing evaluation submit file")
    report = {
        "controllers_to_hold": controllers,
        "preserved_training_jobs": training,
        "evaluation_jobs": len(plan),
        "pre_interpreter": str(Path(sys.executable).resolve()),
    }
    print(json.dumps(report, indent=2), flush=True)
    if not a.apply:
        print("Read-only preview; use --apply to hold controllers and generate repair.")
        return
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Export HF_TOKEN first; no jobs changed")
    for controller in controllers:
        ensure_held(controller)
    target.mkdir()
    (target / "logs").mkdir()
    dag = []
    interpreter = Path(sys.executable).resolve()
    controller_flags = " ".join(f"--controller {c}" for c in controllers)
    for item in plan:
        task, model, seed = item["task"], item["model"], item["seed"]
        name = f"{task}_{model}_{seed}_eval"
        source = COMPACT / f"{name}.sub"
        content = source.read_text()
        content = content.replace(
            str(COMPACT / "logs" / name), str(target / "logs" / name)
        )
        submit = target / f"{name}.sub"
        submit.write_text(content)
        dag.append(f"JOB {name} {submit}")
        dag.append(
            f"SCRIPT DEFER 75 60 PRE {name} {interpreter} -S experiments/scaling_regime/protocols/filler/recovery_sic/ready.py {controller_flags} --task {task} --model {model} --seed {seed}"
        )
    (target / "evaluation.dag").write_text("\n".join(dag) + "\n")
    (target / "repair.json").write_text(json.dumps(report, indent=2))
    print("Submit once:", target / "evaluation.dag")
    print("Both prior controllers must stay held; training nodes remain undisturbed.")


if __name__ == "__main__":
    main()
