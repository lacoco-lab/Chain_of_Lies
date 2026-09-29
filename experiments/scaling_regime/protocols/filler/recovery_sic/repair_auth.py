#!/usr/bin/env python3
"""Repair only the four known held Llama workers and future repaired-DAG submissions."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

JOBS = {
    "63497.0": ("train", "parity", "llama", "0"),
    "63513.0": ("train", "s5", "llama", "2"),
    "63516.0": ("eval", "s5", "llama", "0"),
    "63518.0": ("eval", "s5", "llama", "1"),
}
ROOT = Path("/home/momo00016/Faithfulness-Safety")


def atomic_text(path, text, mode):
    temporary = path.with_name(path.name + ".auth.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(text)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "")
    if not re.fullmatch(r"hf_[A-Za-z0-9]+", token):
        raise SystemExit(
            "Export a valid HF_TOKEN first; credentials will not be printed."
        )
    request = urllib.request.Request(
        "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(request, timeout=20):
            pass
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Current token failed Llama access check: HTTP {error.code}")
    except urllib.error.URLError:
        raise SystemExit("Network access check failed; no changes made.")

    for job, expected in JOBS.items():
        rows = json.loads(
            subprocess.check_output(
                ["condor_q", job, "-json", "-attributes", "JobStatus,Args"], text=True
            )
        )
        if len(rows) != 1 or rows[0]["JobStatus"] != 5:
            raise SystemExit(
                f"{job} is not held; refusing changes. Inspect current queue."
            )
        words = rows[0].get("Args", "").split()
        if (
            len(words) < 5
            or words[0]
            != "experiments/scaling_regime/protocols/filler/recovery_sic/run.sh"
            or tuple(words[1:5]) != expected
        ):
            raise SystemExit(f"Unexpected identity for {job}; no changes made.")

    secret = Path(
        "/home/momo00016/.config/faithfulness-safety/filler_recovery_hf.environment"
    )
    target = ROOT / "recovery_work/filler_only_sic_v1/compact/pre_fixed"
    files = list(target.glob("*_eval.sub"))
    if len(files) != 8:
        raise SystemExit(
            "Expected exactly eight repaired evaluation submit files; refusing changes."
        )
    llama_files = [file for file in files if "_llama_" in file.name]
    original = 'environment = "HF_TOKEN=$ENV(HF_TOKEN)"'
    replacement = "include : " + str(secret)
    for file in llama_files:
        if original not in file.read_text() and replacement not in file.read_text():
            raise SystemExit(f"Unexpected environment definition in {file.name}")
    print(
        "Llama access OK. Four held workers and four future Llama evaluation submit files verified."
    )
    if not args.apply:
        print(
            "Preview only. Use --apply to update credentials. No jobs will be released or stopped."
        )
        return
    secret.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(secret.parent, 0o700)
    atomic_text(secret, f'environment = "HF_TOKEN={token}"\n', 0o600)
    for file in llama_files:
        atomic_text(
            file,
            file.read_text().replace(original, replacement),
            file.stat().st_mode & 0o777,
        )
    # Generated workers have exactly one configured environment variable: HF_TOKEN.
    # Suppress qedit output so it cannot accidentally echo credentials.
    for job in JOBS:
        completed = subprocess.run(
            ["condor_qedit", job, "Environment", json.dumps("HF_TOKEN=" + token)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode:
            raise SystemExit(
                f"Credential update failed for {job}; do not release jobs. Re-run repair after inspection."
            )
        print(job + ": queued credential updated; still held")
    print(
        "Future repaired-DAG Llama submissions now read a private credential include, not the old controller environment."
    )
    print(
        "No training settings, checkpoints, responses, running Qwen jobs, or controllers were changed."
    )


if __name__ == "__main__":
    main()
