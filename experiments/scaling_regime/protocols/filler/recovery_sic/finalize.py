"""Validate all recovered shards before publishing ONLY the eight recovered cells."""

import datetime
import json
import shutil
from pathlib import Path
from common import *
from common import _training
from audit import locations, verify
from chain_of_lies.evaluation.rewards import summarize_variant_results


def main():
    config = load_config(CONFIG)
    plan = json.loads((WORK / "plan.json").read_text())
    assembled = WORK / "assembled"
    if assembled.exists():
        raise ValueError("Assembly exists; inspect it rather than overwrite it")
    assembled.mkdir()
    for row in plan:
        task, model, seed = row["task"], row["model"], row["seed"]
        spec = config["tasks"][task]
        weight = checkpoint_valid(
            condition(task, model, seed), task, model, seed, config
        )
        dest = assembled / canonical_condition(task, model, seed)
        shutil.copytree(condition(task, model, seed), dest)
        for shard in sorted((cell(task, model, seed) / "shards").iterdir()):
            binding = json.loads((shard / "binding.json").read_text())
            complete = json.loads((shard / "complete.json").read_text())
            if (
                binding["checkpoint_sha256"] != weight
                or complete["checkpoint_sha256"] != weight
            ):
                raise ValueError("Checkpoint hash mismatch")
            if {f.name for f in (shard / "prompts").glob("*.json")} != set(
                binding["prompts"]
            ):
                raise ValueError("Shard prompt coverage changed")
            for name, checksum in binding["prompts"].items():
                if sha(shard / "prompts" / name) != checksum:
                    raise ValueError("Prompt checksum changed")
            verify(
                shard / "prompts",
                shard / "responses",
                _training(spec, seed)["evaluation_max_new_tokens"],
            )
            response_dir = assembled / canonical_responses(task, model, seed)
            response_dir /= (
                f'length_{binding["length"]}/ckpt_final/trained/{spec["variant"]}'
                if task != "parity"
                else f'ckpt_final/finetuned/{spec["variant"]}'
            )
            response_dir.mkdir(parents=True, exist_ok=True)
            for file in (shard / "responses").glob("*.json"):
                if (response_dir / file.name).exists():
                    raise ValueError("Duplicate evaluation ID")
                shutil.copy2(file, response_dir / file.name)
        for length, prompts, responses, path, key in locations(task, model, seed, spec):
            expected = spec["eval_examples_per_difficulty"] * (
                len(spec["difficulty_values"]) if task == "parity" else 1
            )
            if len(list(prompts.glob("*.json"))) != expected:
                raise ValueError("Wrong source evaluation count")
            for shard in (cell(task, model, seed) / "shards").iterdir():
                binding = json.loads((shard / "binding.json").read_text())
                if binding["length"] != length:
                    continue
                for name, checksum in binding["prompts"].items():
                    if sha(prompts / name) != checksum:
                        raise ValueError(
                            "Source evaluation record changed after planning"
                        )
            metrics, hits = verify(
                prompts,
                assembled / responses,
                _training(spec, seed)["evaluation_max_new_tokens"],
            )
            report = {
                "schema_version": 1,
                "task": task,
                "model": model,
                "seed": seed,
                "condition": "filler_only",
                "variant": spec["variant"],
                "checkpoint": "ckpt_final",
                "checkpoint_sha256": weight,
                "adapter_root": str(
                    canonical_condition(task, model, seed) / spec["variant"]
                ),
                "prompts_dir": str(prompts),
                "responses_dir": str(responses),
                "generation_cap_hits": hits,
                key: metrics,
            }
            if task == "parity":
                report.update(
                    variant_name=spec["variant"],
                    baseline=None,
                    delta=None,
                    baseline_note="Recovery scores trained model only; redundant base-model pass not repeated.",
                )
                report = {
                    "schema_version": 1,
                    "checkpoint": "ckpt_final",
                    "reports": [report],
                }
            else:
                report["length"] = int(length)
            target = assembled / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2))
    # No canonical writes occur until EVERY recovered cell passes validation.
    backup = WORK / "previous_canonical"
    backup.mkdir(exist_ok=False)
    for row in plan:
        for target in [
            canonical_condition(row["task"], row["model"], row["seed"]),
            canonical_responses(row["task"], row["model"], row["seed"]),
        ]:
            if target.exists():
                saved = backup / target
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(saved))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(assembled / target, target)
    (WORK / "published.json").write_text(
        json.dumps({"published": True, "cells": plan, "backup": str(backup)}, indent=2)
    )
    print(
        "Published eight verified cells; previous canonical outputs retained at", backup
    )


if __name__ == "__main__":
    main()
