#!/usr/bin/env python3
"""Audited per-example analysis of the completed matched Easy v2 suite."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.evaluation.rewards import (
    extract_monitor_visible_reasoning,
    score_completion,
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.generate_splits import (
    TASKS,
    load_config,
)

CONFIG_PATH = REPO_ROOT / "experiments" / "easy_regime" / "config.json"
DATA_ROOT = REPO_ROOT / "generated_data" / "easy_regime"
RESPONSES_ROOT = REPO_ROOT / "generated_data" / "easy_regime_eval_responses"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "easy_regime"
OUTPUT_ROOT = ARTIFACTS_ROOT / "analysis"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_roots(
    config: dict[str, Any], task: str, model: str
) -> tuple[Path, Path, Path]:
    override = config.get("result_overrides", {}).get(f"{task}/{model}")
    if not override:
        return DATA_ROOT, ARTIFACTS_ROOT, RESPONSES_ROOT
    return tuple(REPO_ROOT / override[key] for key in ("data_root", "artifacts_root", "responses_root"))  # type: ignore[return-value]


def _report(
    artifacts_root: Path, task: str, model: str, seed: int, condition: str, variant: str
) -> tuple[Path, dict[str, Any]]:
    path = (
        artifacts_root
        / task
        / model
        / f"seed_{seed}"
        / condition
        / "per_variant_eval"
        / "ckpt_final"
        / f"{variant}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload.get("reports", [])
    if len(reports) != 1:
        raise RuntimeError(f"Expected one report in {path}.")
    return path, reports[0]


def _longest_period_run(text: str) -> int:
    return max((len(match.group(0)) for match in re.finditer(r"\.+", text)), default=0)


def _length(record: dict[str, Any]) -> tuple[int, str]:
    spec = record["spec"]
    if record["task_type"] == "s5":
        return len(spec["private_instruction_sequence"]), "swap_count"
    if record["task_type"] == "knowledge":
        return int(spec.get("facts_per_question", 1)), "fact_count"
    operands = [*spec["public_operands"], *spec["private_operands"]]
    return (
        max(len(str(abs(int(value)))) for value in operands),
        "maximum_operand_decimal_digits",
    )


def _mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    lower = min(wins, losses)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def extract() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
):
    config = load_config(CONFIG_PATH)
    per_example: list[dict[str, Any]] = []
    cell_metrics: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for task in TASKS:
        for model in config["models"]:
            for seed in config["seeds"]:
                for condition, condition_spec in config["conditions"].items():
                    variant = condition_spec["variant"].format(task=task)
                    data_root, artifacts_root, responses_root = _source_roots(
                        config, task, model
                    )
                    prompts_dir = (
                        data_root / task / f"seed_{seed}" / variant / "val_prompts"
                    )
                    responses_dir = (
                        responses_root
                        / task
                        / model
                        / f"seed_{seed}"
                        / condition
                        / "ckpt_final"
                        / "trained"
                        / variant
                    )
                    report_path, report = _report(
                        artifacts_root, task, model, seed, condition, variant
                    )
                    prompt_paths = sorted(prompts_dir.glob("*.json"))
                    response_paths = sorted(responses_dir.glob("*.json"))
                    if len(prompt_paths) != 1000 or len(response_paths) != 1000:
                        raise RuntimeError(
                            f"Incomplete {task}/{model}/seed_{seed}/{condition}: {len(prompt_paths)}/{len(response_paths)}"
                        )
                    local = []
                    for prompt_path in prompt_paths:
                        response_path = responses_dir / prompt_path.name
                        if not response_path.exists():
                            raise FileNotFoundError(response_path)
                        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
                        response = json.loads(response_path.read_text(encoding="utf-8"))
                        raw = str(response["raw_text"])
                        reasoning = extract_monitor_visible_reasoning(prompt, raw)
                        scored = score_completion(prompt, raw)
                        nominal_length, length_definition = _length(prompt)
                        row = {
                            "regime": "easy_v2",
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "variant": variant,
                            "experiment_id": prompt["experiment_id"],
                            "nominal_length": nominal_length,
                            "nominal_length_definition": length_definition,
                            "prompt_words": len(str(prompt["prompt_text"]).split()),
                            "prompt_chars": len(str(prompt["prompt_text"])),
                            "public_correct": int(bool(scored.flags["public_correct"])),
                            "private_correct": int(
                                bool(scored.flags["private_correct"])
                            ),
                            "joint_correct": int(bool(scored.task_success)),
                            "format_correct": int(bool(scored.format_ok)),
                            "concealment_correct": int(bool(scored.concealment_ok)),
                            "payload_decode_correct": (
                                int(bool(scored.flags.get("payload_decode_correct")))
                                if prompt["spec"].get("steganography_payload")
                                is not None
                                else ""
                            ),
                            "cot_words": scored.cot_word_count,
                            "cot_chars": len(reasoning),
                            "generated_tokens_including_answer_block": len(
                                response.get("generated_token_ids") or []
                            ),
                            "longest_period_run": _longest_period_run(reasoning),
                        }
                        local.append(row)
                        per_example.append(row)
                    rl = report["rl"]
                    recomputed = {
                        "task_success_rate": statistics.fmean(
                            row["joint_correct"] for row in local
                        ),
                        "public_exact_rate": statistics.fmean(
                            row["public_correct"] for row in local
                        ),
                        "private_exact_rate": statistics.fmean(
                            row["private_correct"] for row in local
                        ),
                        "avg_cot_words": statistics.fmean(
                            row["cot_words"] for row in local
                        ),
                    }
                    for metric, value in recomputed.items():
                        if abs(float(rl[metric]) - value) > 1e-12:
                            raise RuntimeError(
                                f"Metric mismatch in {report_path}: {metric}"
                            )
                    payload_rows = [
                        row for row in local if row["payload_decode_correct"] != ""
                    ]
                    exact_filler = (
                        [int(row["longest_period_run"] == 64) for row in local]
                        if condition == "filler"
                        else []
                    )
                    cell_metrics.append(
                        {
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "variant": variant,
                            "n": len(local),
                            **recomputed,
                            "payload_decode_rate": (
                                statistics.fmean(
                                    int(row["payload_decode_correct"])
                                    for row in payload_rows
                                )
                                if payload_rows
                                else ""
                            ),
                            "private_exact_given_payload": (
                                statistics.fmean(
                                    row["private_correct"]
                                    for row in payload_rows
                                    if row["payload_decode_correct"] == 1
                                )
                                if any(
                                    row["payload_decode_correct"] == 1
                                    for row in payload_rows
                                )
                                else ""
                            ),
                            "exact_64_period_run_rate": (
                                statistics.fmean(exact_filler) if exact_filler else ""
                            ),
                            "minimum_longest_period_run": (
                                min(
                                    (row["longest_period_run"] for row in local),
                                    default="",
                                )
                                if condition == "filler"
                                else ""
                            ),
                            "maximum_longest_period_run": (
                                max(
                                    (row["longest_period_run"] for row in local),
                                    default="",
                                )
                                if condition == "filler"
                                else ""
                            ),
                            "mean_generated_tokens_including_answer_block": statistics.fmean(
                                row["generated_tokens_including_answer_block"]
                                for row in local
                            ),
                        }
                    )
                    metadata = report["training"]["training_metadata"]
                    training.append(
                        {
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "train_examples_seen": metadata.get("train_examples_seen"),
                            "steps": metadata.get("steps"),
                            "epochs": metadata.get("epochs"),
                            "supervision_mode": metadata.get("supervision_mode"),
                            "filler_token_count": metadata.get("filler_token_count"),
                            "avg_supervised_tokens_per_example_estimate": metadata.get(
                                "avg_supervised_tokens_per_example_estimate"
                            ),
                            "estimated_total_supervised_tokens": metadata.get(
                                "estimated_total_supervised_tokens"
                            ),
                            "final_loss": report["training"].get("final_loss"),
                        }
                    )
    return per_example, cell_metrics, training


def contrasts(
    per_example: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = []
    for task in TASKS:
        for model in ("qwen", "llama"):
            for seed in (0, 1, 2):
                subset = [
                    row
                    for row in per_example
                    if (row["task"], row["model"], row["seed"]) == (task, model, seed)
                ]
                lookup = {
                    (row["condition"], row["experiment_id"]): row for row in subset
                }
                ids = sorted({row["experiment_id"] for row in subset})
                for condition in (
                    "public_only_cot",
                    "filler",
                    "piggyback",
                    "steganography",
                ):
                    for baseline in ("vanilla", "public_only_cot"):
                        if condition == baseline:
                            continue
                        wins = losses = both = neither = 0
                        for experiment_id in ids:
                            target = int(
                                lookup[(condition, experiment_id)]["private_correct"]
                            )
                            control = int(
                                lookup[(baseline, experiment_id)]["private_correct"]
                            )
                            wins += target == 1 and control == 0
                            losses += target == 0 and control == 1
                            both += target == 1 and control == 1
                            neither += target == 0 and control == 0
                        per_seed.append(
                            {
                                "task": task,
                                "model": model,
                                "seed": seed,
                                "condition": condition,
                                "baseline": baseline,
                                "n": len(ids),
                                "condition_only_correct": wins,
                                "baseline_only_correct": losses,
                                "both_correct": both,
                                "neither_correct": neither,
                                "private_accuracy_difference": (wins - losses)
                                / len(ids),
                                "mcnemar_exact_p": _mcnemar_p(wins, losses),
                            }
                        )
    aggregate = []
    groups = sorted(
        {
            (row["task"], row["model"], row["condition"], row["baseline"])
            for row in per_seed
        }
    )
    for task, model, condition, baseline in groups:
        subset = [
            row
            for row in per_seed
            if (row["task"], row["model"], row["condition"], row["baseline"])
            == (task, model, condition, baseline)
        ]
        values = [float(row["private_accuracy_difference"]) for row in subset]
        aggregate.append(
            {
                "task": task,
                "model": model,
                "condition": condition,
                "baseline": baseline,
                "n_seeds": 3,
                "private_accuracy_difference_mean": statistics.fmean(values),
                "private_accuracy_difference_sample_sd": statistics.stdev(values),
                "pooled_condition_only_correct": sum(
                    int(row["condition_only_correct"]) for row in subset
                ),
                "pooled_baseline_only_correct": sum(
                    int(row["baseline_only_correct"]) for row in subset
                ),
            }
        )
    return per_seed, aggregate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_example, cells, training = extract()
    contrast_seed, contrast_aggregate = contrasts(per_example)
    outputs = {
        "per_example.csv": per_example,
        "cell_metrics.csv": cells,
        "training_diagnostics.csv": training,
        "paired_contrasts_per_seed.csv": contrast_seed,
        "paired_contrasts_aggregate.csv": contrast_aggregate,
    }
    for name, rows in outputs.items():
        _write_csv(OUTPUT_ROOT / name, rows)
    manifest = {
        "schema_version": 2,
        "valid": True,
        "per_example_rows": len(per_example),
        "cell_rows": len(cells),
        "training_rows": len(training),
        "result_overrides": load_config(CONFIG_PATH).get("result_overrides", {}),
        "outputs": [
            {"path": name, "rows": len(rows), "sha256": _sha256(OUTPUT_ROOT / name)}
            for name, rows in outputs.items()
        ],
    }
    (OUTPUT_ROOT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
