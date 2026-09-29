"""Audit exact prompt coverage, model files and recomputed metrics (no Torch)."""

import argparse
import json
import math
from pathlib import Path
from common import *
from common import _training
from capture import JOBS
from experiments.scaling_regime.protocols.filler.experiment import (
    _directories,
    _validate_cell_inputs,
)
from chain_of_lies.evaluation.rewards import summarize_variant_results


def locations(task, model, seed, spec):
    root = canonical_condition(task, model, seed)
    responses = canonical_responses(task, model, seed)
    if task == "parity":
        yield "all", _directories(spec, seed)[
            1
        ], responses / "ckpt_final/finetuned" / spec["variant"], root / spec[
            "variant"
        ] / "per_variant_eval/ckpt_final" / f'{spec["variant"]}.json', "rl"
    else:
        for length in spec["difficulty_values"]:
            yield str(length), Path(
                spec["data_root"]
            ) / "shared_eval" / f"length_{length}" / spec[
                "variant"
            ], responses / f"length_{length}/ckpt_final/trained" / spec[
                "variant"
            ], root / f"per_length_eval/ckpt_final/length_{length}.json", "metrics"


def verify(prompts, responses, cap):
    names = {f.name for f in prompts.glob("*.json")}
    if names != {f.name for f in responses.glob("*.json")}:
        raise ValueError(f"Prompt/response coverage mismatch: {responses}")
    hits = 0
    for name in names:
        raw = response_valid(responses / name, Path(name).stem, cap)
        hits += len(raw["generated_token_ids"]) == cap
    metrics = summarize_variant_results(prompts, responses)
    if metrics["num_examples"] != len(names) or metrics["missing_responses"]:
        raise ValueError("Incomplete metrics")
    return metrics, hits


def report_metrics(document, key, variant):
    if key == "rl" and "reports" in document:
        candidates = [
            r for r in document["reports"] if r.get("variant_name") == variant
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one parity variant report")
        return candidates[0]["rl"]
    return document[key]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completed-16", action="store_true")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    config = load_config(CONFIG)
    rows = []
    pending = set(JOBS.values())
    for task, spec in config["tasks"].items():
        for model in config["models"]:
            for seed in config["seeds"]:
                if a.completed_16 and (task, model, seed) in pending:
                    continue
                row = {"task": task, "model": model, "seed": seed, "complete": False}
                rows.append(row)
                try:
                    _validate_cell_inputs(spec, seed)
                    row["checkpoint_sha256"] = checkpoint_valid(
                        canonical_condition(task, model, seed),
                        task,
                        model,
                        seed,
                        config,
                    )
                    row["evaluations"] = []
                    for length, prompts, responses, path, key in locations(
                        task, model, seed, spec
                    ):
                        expected = spec["eval_examples_per_difficulty"] * (
                            len(spec["difficulty_values"]) if task == "parity" else 1
                        )
                        if len(list(prompts.glob("*.json"))) != expected:
                            raise ValueError("Wrong expected evaluation prompt count")
                        metrics, hits = verify(
                            prompts,
                            responses,
                            _training(spec, seed)["evaluation_max_new_tokens"],
                        )
                        stored = report_metrics(
                            json.loads(path.read_text()), key, spec["variant"]
                        )
                        for metric, value in metrics.items():
                            if type(value) not in (int, float):
                                continue
                            if not math.isclose(value, stored[metric], abs_tol=1e-10):
                                raise ValueError(
                                    f"Report metric mismatch {path}: {metric}"
                                )
                        row["evaluations"].append(
                            {
                                "length": length,
                                "num_examples": metrics["num_examples"],
                                "task_success_rate": metrics["task_success_rate"],
                                "private_exact_rate": metrics["private_exact_rate"],
                                "generation_cap_hits": hits,
                            }
                        )
                    row["complete"] = True
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row["error"] = str(error)
    result = {"complete": all(r["complete"] for r in rows), "cells": rows}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
