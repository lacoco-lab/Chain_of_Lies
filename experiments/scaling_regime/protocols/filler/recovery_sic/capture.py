#!/usr/bin/env python3
"""Copy live outputs over noninteractive SSH (or SFTP); never stops jobs."""

import argparse
import datetime
import json
from pathlib import Path, PurePosixPath
import subprocess
import shlex
import tarfile
import gzip


def extract_snapshot(archive, destination, allowed):
    # Read through the gzip trailer: tar's end marker alone is insufficient.
    with gzip.open(archive, "rb") as stream:
        while stream.read(1048576):
            pass
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("Unsafe archive entry: " + member.name)
            if not any(
                name == PurePosixPath(prefix) or PurePosixPath(prefix) in name.parents
                for prefix in allowed
            ):
                raise ValueError("Unexpected archive prefix: " + member.name)
        for member in members:
            bundle.extract(member, destination)


def capture_ssh(job, task, model, seed, destination, log):
    paths = [
        f"artifacts/filler_only_scaling_v1/{task}/{model}/seed_{seed}/filler_only",
        f"generated_data/filler_only_scaling_v1_eval_responses/{task}/{model}/seed_{seed}/filler_only",
    ]
    code = (
        "import sys,tarfile; from pathlib import Path; paths=" + repr(paths) + "; "
        "assert Path(paths[0]).is_dir(), 'Cell artifacts not found in job working directory'; "
        "bundle=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); "
        "[bundle.add(p,arcname=p) for p in paths if Path(p).exists()]; bundle.close()"
    )
    command = "python3 -c " + shlex.quote(code)
    archive = destination / "outputs.tar.gz"
    with archive.open("wb") as output:
        status = subprocess.run(
            ["condor_ssh_to_job", "-ssh", "ssh -T", job, command],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=log,
        ).returncode
    if status == 0:
        extract_snapshot(archive, destination, paths)
    return status


JOBS = {
    "63134.0": ("parity", "qwen", 0),
    "63134.3": ("parity", "llama", 0),
    **{f"63136.{i}": ("s5", "qwen" if i < 3 else "llama", i % 3) for i in range(6)},
}


def query(job):
    output = subprocess.check_output(
        ["condor_q", job, "-af", "JobStatus", "NumJobStarts"], text=True
    ).strip()
    return output.split() if output else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--transport", choices=["ssh", "sftp"], default="ssh")
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cells": [],
    }
    for job, (task, model, seed) in JOBS.items():
        before = query(job)
        row = {
            "job": job,
            "task": task,
            "model": model,
            "seed": seed,
            "before": before,
            "stable": False,
        }
        report["cells"].append(row)
        if not before or before[0] != "2":
            row["reason"] = (
                "No running worker; inspect canonical returned outputs or retrain missing cell"
            )
        else:
            destination = root / job
            (destination / "artifacts").mkdir(parents=True)
            (destination / "generated_data").mkdir()
            batch = (
                f"get -r artifacts/filler_only_scaling_v1 {destination}/artifacts/filler_only_scaling_v1\n"
                f"-get -r generated_data/filler_only_scaling_v1_eval_responses {destination}/generated_data/filler_only_scaling_v1_eval_responses\n"
            )
            print(
                f"Capturing {job} via {args.transport}; log: {destination}/capture.log",
                flush=True,
            )
            with (destination / "capture.log").open("w") as log:
                try:
                    if args.transport == "ssh":
                        status = capture_ssh(job, task, model, seed, destination, log)
                    else:
                        status = subprocess.run(
                            ["condor_ssh_to_job", "-ssh", "sftp -b -", job],
                            input=batch,
                            text=True,
                            stdout=log,
                            stderr=log,
                        ).returncode
                except (OSError, ValueError, EOFError, tarfile.TarError) as error:
                    log.write(
                        "\nCapture/archive validation failed: " + str(error) + "\n"
                    )
                    status = 1
            after = query(job)
            row.update(
                after=after,
                transport=args.transport,
                transport_exit=status,
                stable=(status == 0 and before == after),
            )
            if not row["stable"]:
                row["reason"] = (
                    "Copy/validation failed or attempt changed; snapshot must not be reused automatically"
                )
        (root / "capture.json").write_text(json.dumps(report, indent=2))
        print(job, "CAPTURED" if row["stable"] else row.get("reason"), flush=True)
    print("Snapshot:", root)
    print("No jobs were stopped. Run planner validation before removing old jobs.")


if __name__ == "__main__":
    main()
