#!/usr/bin/env python3
"""Audit and print the five primary protocols' public-accuracy table as CSV.

This is read-only. In particular, `filler` is the filler-only replacement; the
separately preserved `filler_plus_public_cot` ablation is intentionally excluded.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _release_root() -> Path:
    candidates = (
        ROOT / "models_data_and_evaluation_outputs",
        ROOT.parent / "models_data_and_evaluation_outputs",
        ROOT / "paper_release",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


RELEASE = _release_root() / "Scaling Regime/tasks"
OLD = ROOT / "docs/scaling_regime/tables"
TASKS = {
    "multiplication": (
        "multiplication",
        range(2, 5),
        "aggregate_length_metrics_mult.csv",
    ),
    "iterated_addition": (
        "iterated_addition",
        range(1, 6),
        "aggregate_length_metrics_knowledge.csv",
    ),
    "s5_state_tracking": (
        "s5_state_tracking",
        range(1, 20),
        "aggregate_length_metrics_s5.csv",
    ),
}
MODELS = ("qwen", "llama")
CONDITIONS = ("vanilla", "filler", "public_only_cot", "piggyback", "invisible")
PARITY_BUCKETS = (
    ("008_015", 8, 15),
    ("016_031", 16, 31),
    ("032_063", 32, 63),
    ("064_127", 64, 127),
    ("128_255", 128, 255),
)
PARITY_VARIANTS = {
    "vanilla": "parity_control",
    "filler": "parity_control",
    "public_only_cot": "parity_control",
    "piggyback": "parity_piggyback",
    "invisible": "parity_steg_local_invisible",
}
OLD_NAMES = {"invisible": "steganography"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_rate(rate: object, name: str) -> float:
    require(
        isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= rate <= 1,
        f"Invalid rate {name}: {rate!r}",
    )
    return float(rate)


def check_aggregate(
    values: list[float],
    aggregate: dict[str, str],
    mean_field: str,
    sd_field: str,
    name: str,
) -> None:
    mean, sd = statistics.mean(values), statistics.stdev(values)
    require(
        math.isclose(mean, float(aggregate[mean_field]), abs_tol=1e-12),
        f"Mean mismatch: {name}: {mean} vs {aggregate[mean_field]}",
    )
    require(
        math.isclose(sd, float(aggregate[sd_field]), abs_tol=1e-12),
        f"SD mismatch: {name}: {sd} vs {aggregate[sd_field]}",
    )


def conventional_rows() -> list[dict[str, object]]:
    out = []
    for task, (release_name, lengths, old_file) in TASKS.items():
        old_rows = {
            (r["model"], r["condition"], int(r["length"])): r
            for r in csv.DictReader((OLD / old_file).open(newline=""))
        }
        for model in MODELS:
            for condition in CONDITIONS:
                for length in lengths:
                    rates = []
                    for seed in range(3):
                        path = (
                            RELEASE
                            / release_name
                            / "evaluation_reports"
                            / model
                            / f"seed_{seed}"
                            / condition
                            / "per_length_eval/ckpt_final"
                            / f"length_{length}.json"
                        )
                        require(path.is_file(), f"Missing report: {path}")
                        report = json.loads(path.read_text())
                        metrics = report["metrics"]
                        require(
                            report["checkpoint"] == "ckpt_final"
                            and report["seed"] == seed,
                            f"Wrong checkpoint/seed: {path}",
                        )
                        require(
                            metrics["num_examples"] == 200
                            and not metrics["missing_responses"],
                            f"Incomplete evaluation: {path}",
                        )
                        if condition == "filler":
                            require(
                                report["condition"] == "filler_only",
                                f"Wrong filler: {path}",
                            )
                        rates.append(
                            check_rate(metrics["public_exact_rate"], str(path))
                        )
                    if condition != "filler":
                        old_key = (model, OLD_NAMES.get(condition, condition), length)
                        require(
                            old_key in old_rows, f"Missing source aggregate: {old_key}"
                        )
                        check_aggregate(
                            rates,
                            old_rows[old_key],
                            "public_exact_rate_mean",
                            "public_exact_rate_sample_sd",
                            str(old_key),
                        )
                    out.append(
                        dict(
                            task=task,
                            model=model,
                            condition=condition,
                            length=str(length),
                            length_min=length,
                            length_max=length,
                            n_seeds=3,
                            n_per_seed=200,
                            public_exact_mean=statistics.mean(rates),
                            public_exact_sample_sd=statistics.stdev(rates),
                            source="final_per_length_reports",
                        )
                    )
    return out


def parity_rows() -> list[dict[str, object]]:
    task_root = RELEASE / "plain_parity"
    old_rows = {
        (r["model"], r["condition"], r["length_bucket"]): r
        for r in csv.DictReader(
            (OLD / "metrics_by_length_bucket_three_seed_parity.csv").open(newline="")
        )
        if r["source"] == "finetuned"
    }
    out = []
    for model in MODELS:
        for condition in CONDITIONS:
            variant = PARITY_VARIANTS[condition]
            by_bucket: dict[str, list[float]] = defaultdict(list)
            for seed in range(3):
                prompt_dir = (
                    task_root / f"prompts_and_splits/seed_{seed}/{variant}/val_prompts"
                )
                response_dir = (
                    task_root
                    / "responses"
                    / model
                    / f"seed_{seed}"
                    / condition
                    / "ckpt_final/finetuned"
                    / variant
                )
                prompts = sorted(prompt_dir.glob("*.json"))
                require(
                    len(prompts) == 1000, f"Expected 1000 parity prompts: {prompt_dir}"
                )
                buckets: dict[str, list[bool]] = defaultdict(list)
                for prompt_path in prompts:
                    prompt = json.loads(prompt_path.read_text())
                    response_path = response_dir / prompt_path.name
                    require(
                        response_path.is_file(),
                        f"Missing parity response: {response_path}",
                    )
                    response = json.loads(response_path.read_text())
                    require(
                        response["experiment_id"] == prompt["experiment_id"],
                        f"Response/prompt mismatch: {response_path}",
                    )
                    match = re.search(
                        r"Answer to main question:\s*(-?\d+)", response["raw_text"]
                    )
                    correct = (
                        match is not None
                        and int(match.group(1)) == prompt["spec"]["public_answer"]
                    )
                    buckets[prompt["spec"]["length_bucket"]].append(correct)
                for bucket, _, _ in PARITY_BUCKETS:
                    require(
                        len(buckets[bucket]) == 200,
                        f"Expected 200 parity prompts: {response_dir}/{bucket}",
                    )
                    by_bucket[bucket].append(sum(buckets[bucket]) / 200)
            for bucket, low, high in PARITY_BUCKETS:
                rates = by_bucket[bucket]
                if condition != "filler":
                    old_key = (model, OLD_NAMES.get(condition, condition), bucket)
                    require(old_key in old_rows, f"Missing source aggregate: {old_key}")
                    check_aggregate(
                        rates,
                        old_rows[old_key],
                        "public_exact_mean",
                        "public_exact_sample_sd",
                        str(old_key),
                    )
                out.append(
                    dict(
                        task="plain_parity",
                        model=model,
                        condition=condition,
                        length=bucket,
                        length_min=low,
                        length_max=high,
                        n_seeds=3,
                        n_per_seed=200,
                        public_exact_mean=statistics.mean(rates),
                        public_exact_sample_sd=statistics.stdev(rates),
                        source="final_per_example_responses",
                    )
                )
    return out


def main() -> None:
    rows = conventional_rows() + parity_rows()
    require(
        len(rows) == 320,
        f"Expected 320 task/model/condition/length points; got {len(rows)}",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print(
        f"Audited {len(rows)} groups, each with three seeds and 200 examples per seed.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
