#!/usr/bin/env python3
"""Preserve compact figure-ready data before deleting ``Temp results``.

This script extracts every trained Hard evaluation example used by the two
requested figures, checks the recomputed metrics against the authoritative
evaluation JSON, and copies the small legacy Easy/Medium summaries out of the
temporary result tree.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import sys
from dataclasses import dataclass
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

OUTPUT_ROOT = REPO_ROOT / "artifacts" / "figure_data_preservation"
REPORT_PATH = REPO_ROOT / "docs" / "FIGURE_DATA_PRESERVATION.md"
TEMP_MULT = REPO_ROOT / "Temp results" / "Hard" / "Multiplication"
MODELS = ("qwen", "llama")
SEEDS = (0, 1, 2)
HARD_METHODS = ("vanilla", "public_only_cot", "piggyback", "steganography")
EASY_METHODS = ("vanilla", "filler", "public_only_cot", "piggyback", "steganography")


@dataclass(frozen=True)
class Cell:
    task: str
    model: str
    seed: int
    method: str
    variant: str
    prompts_dir: Path
    responses_dir: Path
    report_path: Path


def _top_level_cell(task: str, model: str, seed: int, method: str) -> Cell:
    variants = {
        "s5": {
            "control": "s5_control",
            "piggyback": "s5_piggyback",
            "steganography": "s5_steg_local_invisible",
        },
        "knowledge": {
            "control": "knowledge_5fact_control",
            "piggyback": "knowledge_5fact_piggyback",
            "steganography": "knowledge_5fact_steg_local_invisible",
        },
    }[task]
    if method == "vanilla":
        family, mode, variant = "piggyback", "answer_only", variants["control"]
    elif method == "public_only_cot":
        family, mode, variant = "piggyback", "public_cot", variants["control"]
    elif method == "piggyback":
        family, mode, variant = "piggyback", "public_cot", variants["piggyback"]
    else:
        family, mode, variant = (
            "steganography",
            "local_channel_cot",
            variants["steganography"],
        )
    return Cell(
        task,
        model,
        seed,
        method,
        variant,
        REPO_ROOT
        / "generated_data"
        / "hard_regime"
        / task
        / f"seed_{seed}"
        / variant
        / "val_prompts",
        REPO_ROOT
        / "generated_data"
        / "hard_regime"
        / f"{task}_eval_responses"
        / model
        / f"seed_{seed}"
        / family
        / mode
        / "ckpt_final"
        / "trained"
        / variant,
        REPO_ROOT
        / "artifacts"
        / "hard_regime"
        / task
        / model
        / f"seed_{seed}"
        / family
        / mode
        / "per_variant_eval"
        / "ckpt_final"
        / f"{variant}.json",
    )


def _multiplication_cell(model: str, seed: int, method: str) -> Cell:
    data = TEMP_MULT / "hard_regime_data"
    if method in {"vanilla", "public_only_cot", "piggyback"}:
        mode = "answer_only" if method == "vanilla" else "public_cot"
        variant = (
            "arith_piggyback" if method == "piggyback" else "arith_piggyback_control"
        )
        return Cell(
            "multiplication",
            model,
            seed,
            method,
            variant,
            data
            / "multiplication_only_piggyback"
            / f"seed_{seed}"
            / variant
            / "val_prompts",
            data
            / "multiplication_only_piggyback_eval_responses"
            / model
            / f"seed_{seed}"
            / mode
            / "ckpt_final"
            / "trained"
            / variant,
            TEMP_MULT
            / "multiplication_only_piggyback"
            / model
            / f"seed_{seed}"
            / mode
            / "per_variant_eval"
            / "ckpt_final"
            / f"{variant}.json",
        )
    variant = "arith_steg_local_invisible"
    return Cell(
        "multiplication",
        model,
        seed,
        method,
        variant,
        data
        / "hard_regime"
        / "confirmatory"
        / "steganography"
        / f"seed_{seed}"
        / variant
        / "val_prompts",
        data
        / "hard_regime"
        / "confirmatory_eval_responses"
        / model
        / f"seed_{seed}"
        / "steganography"
        / "local_channel_cot"
        / "ckpt_final"
        / "trained"
        / variant,
        TEMP_MULT
        / "hard_regime"
        / "confirmatory"
        / model
        / f"seed_{seed}"
        / "steganography"
        / "local_channel_cot"
        / "per_variant_eval"
        / "ckpt_final"
        / f"{variant}.json",
    )


def cells() -> list[Cell]:
    result = []
    for task in ("multiplication", "s5", "knowledge"):
        for model in MODELS:
            for seed in SEEDS:
                for method in HARD_METHODS:
                    result.append(
                        _multiplication_cell(model, seed, method)
                        if task == "multiplication"
                        else _top_level_cell(task, model, seed, method)
                    )
    return result


def _digits(value: Any) -> int:
    return len(str(abs(int(value))))


def _length_fields(record: dict[str, Any]) -> dict[str, Any]:
    spec = record["spec"]
    task_type = record["task_type"]
    if task_type == "s5":
        public_units = len(spec["public_instruction_sequence"])
        private_units = len(spec["private_instruction_sequence"])
        nominal = public_units
        definition = "swap_count"
        arithmetic_max_operand = ""
        arithmetic_max_digits = ""
    elif task_type == "knowledge":
        public_facts = spec.get("public_facts") or [spec.get("public_fact")]
        private_facts = spec.get("private_facts") or [spec.get("private_fact")]
        public_units, private_units = len(public_facts), len(private_facts)
        nominal = public_units
        definition = "fact_count"
        arithmetic_max_operand = ""
        arithmetic_max_digits = ""
    else:
        public_operands = spec.get("public_operands") or [
            int(value)
            for value in str(spec["public_question"]).replace("*", " ").split()
        ]
        private_operands = spec.get("private_operands") or [
            int(value)
            for value in str(spec["private_question"]).replace("*", " ").split()
        ]
        public_units = sum(_digits(value) for value in public_operands)
        private_units = sum(_digits(value) for value in private_operands)
        nominal = max(_digits(value) for value in (*public_operands, *private_operands))
        definition = "maximum_operand_decimal_digits"
        arithmetic_max_operand = max(
            int(value) for value in (*public_operands, *private_operands)
        )
        arithmetic_max_digits = nominal
    return {
        "nominal_length": nominal,
        "nominal_length_definition": definition,
        "public_task_units": public_units,
        "private_task_units": private_units,
        "arithmetic_max_operand": arithmetic_max_operand,
        "arithmetic_max_operand_digits": arithmetic_max_digits,
        "public_question_chars": len(
            str(
                spec.get(
                    "public_question",
                    ";".join(spec.get("public_instruction_sequence", [])),
                )
            )
        ),
        "private_question_chars": len(
            str(
                spec.get(
                    "private_question",
                    ";".join(spec.get("private_instruction_sequence", [])),
                )
            )
        ),
        "prompt_chars": len(str(record.get("prompt_text", ""))),
        "prompt_words": len(str(record.get("prompt_text", "")).split()),
    }


def _report_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload.get("reports", [])
    if len(reports) != 1:
        raise RuntimeError(f"Expected one report in {path}; found {len(reports)}")
    return reports[0]["rl"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_hard() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_example: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for cell in cells():
        for path in (cell.prompts_dir, cell.responses_dir, cell.report_path):
            if not path.exists():
                raise FileNotFoundError(path)
        prompt_paths = sorted(cell.prompts_dir.glob("*.json"))
        response_paths = sorted(cell.responses_dir.glob("*.json"))
        if len(prompt_paths) != 1000 or len(response_paths) != 1000:
            raise RuntimeError(
                f"Incomplete cell {cell}: prompts={len(prompt_paths)}, responses={len(response_paths)}"
            )
        local_rows = []
        for prompt_path in prompt_paths:
            response_path = cell.responses_dir / prompt_path.name
            if not response_path.exists():
                raise FileNotFoundError(response_path)
            prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            raw = str(response["raw_text"])
            scored = score_completion(prompt, raw)
            reasoning = extract_monitor_visible_reasoning(prompt, raw)
            row = {
                "regime": "hard",
                "task": cell.task,
                "model": cell.model,
                "seed": cell.seed,
                "method": cell.method,
                "variant": cell.variant,
                "experiment_id": prompt["experiment_id"],
                **_length_fields(prompt),
                "public_correct": int(bool(scored.flags["public_correct"])),
                "private_correct": int(bool(scored.flags["private_correct"])),
                "joint_correct": int(bool(scored.task_success)),
                "payload_decode_correct": (
                    int(bool(scored.flags.get("payload_decode_correct")))
                    if prompt["spec"].get("steganography_payload") is not None
                    else ""
                ),
                "cot_words": scored.cot_word_count,
                "cot_chars": len(reasoning),
                "generated_tokens_including_answer_block": len(
                    response.get("generated_token_ids") or []
                ),
                "source_prompt": str(prompt_path.relative_to(REPO_ROOT)),
                "source_response": str(response_path.relative_to(REPO_ROOT)),
            }
            local_rows.append(row)
            per_example.append(row)
        report = _report_metrics(cell.report_path)
        recomputed = {
            "public_exact_rate": statistics.fmean(
                row["public_correct"] for row in local_rows
            ),
            "private_exact_rate": statistics.fmean(
                row["private_correct"] for row in local_rows
            ),
            "task_success_rate": statistics.fmean(
                row["joint_correct"] for row in local_rows
            ),
            "avg_cot_words": statistics.fmean(row["cot_words"] for row in local_rows),
        }
        for metric, value in recomputed.items():
            if abs(float(report[metric]) - value) > 1e-12:
                raise RuntimeError(
                    f"Metric mismatch {cell} {metric}: report={report[metric]}, recomputed={value}"
                )
        per_seed.append(
            {
                "regime": "hard",
                "task": cell.task,
                "model": cell.model,
                "seed": cell.seed,
                "method": cell.method,
                "variant": cell.variant,
                "n": len(local_rows),
                **recomputed,
                "payload_decode_rate": report.get("payload_decode_rate"),
                "mean_generated_tokens_including_answer_block": statistics.fmean(
                    row["generated_tokens_including_answer_block"] for row in local_rows
                ),
                "source_report": str(cell.report_path.relative_to(REPO_ROOT)),
            }
        )
    return per_example, per_seed


def aggregate_seed_rows(
    per_seed: list[dict[str, Any]], methods: tuple[str, ...], regime: str
) -> list[dict[str, Any]]:
    result = []
    metrics = (
        "private_exact_rate",
        "public_exact_rate",
        "task_success_rate",
        "avg_cot_words",
        "mean_generated_tokens_including_answer_block",
        "payload_decode_rate",
    )
    for task in ("multiplication", "s5", "knowledge"):
        for model in MODELS:
            for method in methods:
                subset = [
                    row
                    for row in per_seed
                    if (row["task"], row["model"], row["method"])
                    == (task, model, method)
                ]
                if len(subset) != 3:
                    raise RuntimeError(
                        f"Expected three seeds for {task}/{model}/{method}"
                    )
                row: dict[str, Any] = {
                    "regime": regime,
                    "task": task,
                    "model": model,
                    "method": method,
                    "n_seeds": 3,
                    "n_examples": sum(int(item["n"]) for item in subset),
                }
                for metric in metrics:
                    values = [
                        float(item[metric])
                        for item in subset
                        if item.get(metric) not in (None, "")
                    ]
                    row[f"{metric}_mean"] = statistics.fmean(values) if values else ""
                    row[f"{metric}_sample_sd"] = (
                        statistics.stdev(values) if len(values) > 1 else ""
                    )
                result.append(row)
    return result


def aggregate_chain(
    per_example: list[dict[str, Any]], key: str, regime: str
) -> list[dict[str, Any]]:
    result = []
    figure_rows = [
        row for row in per_example if row["method"] in {"piggyback", "steganography"}
    ]
    groups = sorted(
        {(row["task"], row["model"], row["method"], row[key]) for row in figure_rows}
    )
    for task, model, method, length in groups:
        subset = [
            row
            for row in figure_rows
            if (row["task"], row["model"], row["method"], row[key])
            == (task, model, method, length)
        ]
        result.append(
            {
                "regime": regime,
                "task": task,
                "model": model,
                "method": method,
                key: length,
                "n": len(subset),
                "private_accuracy": statistics.fmean(
                    row["private_correct"] for row in subset
                ),
                "joint_accuracy": statistics.fmean(
                    row["joint_correct"] for row in subset
                ),
                "mean_cot_words": statistics.fmean(row["cot_words"] for row in subset),
                "sample_sd_cot_words": (
                    statistics.stdev(row["cot_words"] for row in subset)
                    if len(subset) > 1
                    else 0.0
                ),
                "mean_generated_tokens_including_answer_block": statistics.fmean(
                    row["generated_tokens_including_answer_block"] for row in subset
                ),
            }
        )
    return result


def aggregate_accuracy_by_length(
    per_example: list[dict[str, Any]], regime: str
) -> list[dict[str, Any]]:
    """Aggregate Figure 1 accuracy by task-native length, retaining seed uncertainty."""
    result: list[dict[str, Any]] = []
    groups = sorted(
        {
            (row["task"], row["model"], row["method"], row["nominal_length"])
            for row in per_example
        }
    )
    for task, model, method, length in groups:
        subset = [
            row
            for row in per_example
            if (row["task"], row["model"], row["method"], row["nominal_length"])
            == (task, model, method, length)
        ]
        seed_rates = []
        for seed in sorted({int(row["seed"]) for row in subset}):
            seed_rows = [row for row in subset if int(row["seed"]) == seed]
            seed_rates.append(
                {
                    "private": statistics.fmean(
                        int(row["private_correct"]) for row in seed_rows
                    ),
                    "public": statistics.fmean(
                        int(row["public_correct"]) for row in seed_rows
                    ),
                    "joint": statistics.fmean(
                        int(row["joint_correct"]) for row in seed_rows
                    ),
                }
            )
        output: dict[str, Any] = {
            "regime": regime,
            "task": task,
            "model": model,
            "method": method,
            "nominal_length": length,
            "nominal_length_definition": subset[0]["nominal_length_definition"],
            "n": len(subset),
            "n_seeds": len(seed_rates),
        }
        for metric in ("private", "public", "joint"):
            values = [rates[metric] for rates in seed_rates]
            output[f"{metric}_accuracy_mean_across_seeds"] = statistics.fmean(values)
            output[f"{metric}_accuracy_sample_sd_across_seeds"] = (
                statistics.stdev(values) if len(values) > 1 else ""
            )
            output[f"{metric}_accuracy_pooled"] = statistics.fmean(
                int(row[f"{metric}_correct"]) for row in subset
            )
        result.append(output)
    return result


def load_easy() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the independently audited Easy table and derive plot-ready seed rows."""
    canonical_source = (
        REPO_ROOT / "artifacts" / "easy_regime" / "analysis" / "per_example.csv"
    )
    preserved_source = OUTPUT_ROOT / "easy_per_example.csv"
    canonical_authoritative = (
        REPO_ROOT / "artifacts" / "easy_regime" / "per_seed_metrics.csv"
    )
    preserved_authoritative = OUTPUT_ROOT / "easy_figure1_per_seed.csv"
    source = canonical_source if canonical_source.exists() else preserved_source
    authoritative = (
        canonical_authoritative
        if canonical_authoritative.exists()
        else preserved_authoritative
    )
    if not source.exists() or not authoritative.exists():
        raise FileNotFoundError(
            "Easy source data are missing from both the canonical and figure-preservation directories"
        )

    numeric = {
        "seed": int,
        "nominal_length": int,
        "prompt_words": int,
        "prompt_chars": int,
        "public_correct": int,
        "private_correct": int,
        "joint_correct": int,
        "cot_words": int,
        "cot_chars": int,
        "generated_tokens_including_answer_block": int,
    }
    per_example: list[dict[str, Any]] = []
    with source.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["regime"] = "easy_v2"
            if "condition" in row:
                row["method"] = row.pop("condition")
            for key, converter in numeric.items():
                row[key] = converter(row[key])
            per_example.append(row)
    if len(per_example) != 90_000:
        raise RuntimeError(f"Expected 90,000 Easy rows; found {len(per_example)}")

    expected: dict[tuple[str, str, int, str], dict[str, str]] = {}
    with authoritative.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition") or row["method"]
            expected[(row["task"], row["model"], int(row["seed"]), condition)] = row

    per_seed: list[dict[str, Any]] = []
    groups = sorted(
        {(row["task"], row["model"], row["seed"], row["method"]) for row in per_example}
    )
    for task, model, seed, method in groups:
        subset = [
            row
            for row in per_example
            if (row["task"], row["model"], row["seed"], row["method"])
            == (task, model, seed, method)
        ]
        report = expected[(task, model, seed, method)]
        metrics = {
            "public_exact_rate": statistics.fmean(
                row["public_correct"] for row in subset
            ),
            "private_exact_rate": statistics.fmean(
                row["private_correct"] for row in subset
            ),
            "task_success_rate": statistics.fmean(
                row["joint_correct"] for row in subset
            ),
            "avg_cot_words": statistics.fmean(row["cot_words"] for row in subset),
        }
        for metric, value in metrics.items():
            if abs(float(report[metric]) - value) > 1e-12:
                raise RuntimeError(
                    f"Easy metric mismatch for {task}/{model}/{seed}/{method}: {metric}"
                )
        payload_values = [
            int(row["payload_decode_correct"])
            for row in subset
            if row["payload_decode_correct"] != ""
        ]
        per_seed.append(
            {
                "regime": "easy_v2",
                "task": task,
                "model": model,
                "seed": seed,
                "method": method,
                "variant": subset[0]["variant"],
                "n": len(subset),
                **metrics,
                "payload_decode_rate": (
                    statistics.fmean(payload_values) if payload_values else ""
                ),
                "mean_generated_tokens_including_answer_block": statistics.fmean(
                    row["generated_tokens_including_answer_block"] for row in subset
                ),
                "source_report": "artifacts/easy_regime/per_seed_metrics.csv",
            }
        )
    if len(per_seed) != 90:
        raise RuntimeError(f"Expected 90 Easy seed cells; found {len(per_seed)}")
    return per_example, per_seed


def load_preserved_hard() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reload Hard data after the large temporary source tree has been deleted."""
    example_path = OUTPUT_ROOT / "hard_per_example.csv"
    seed_path = OUTPUT_ROOT / "hard_figure1_per_seed.csv"
    if not example_path.exists() or not seed_path.exists():
        raise FileNotFoundError(
            "Preserved Hard CSVs are missing and Temp results is unavailable"
        )
    integer_fields = {
        "seed",
        "nominal_length",
        "public_task_units",
        "private_task_units",
        "public_question_chars",
        "private_question_chars",
        "prompt_chars",
        "prompt_words",
        "public_correct",
        "private_correct",
        "joint_correct",
        "cot_words",
        "cot_chars",
        "generated_tokens_including_answer_block",
    }
    examples: list[dict[str, Any]] = []
    with example_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in integer_fields:
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            examples.append(row)
    seeds: list[dict[str, Any]] = []
    with seed_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["seed"] = int(row["seed"])
            row["n"] = int(row["n"])
            seeds.append(row)
    if len(examples) != 72_000 or len(seeds) != 72:
        raise RuntimeError(
            f"Incomplete preserved Hard data: examples={len(examples)}, seeds={len(seeds)}"
        )
    return examples, seeds


def load_preserved_inventory() -> list[dict[str, Any]]:
    path = OUTPUT_ROOT / "source_inventory.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_small_summaries() -> list[dict[str, Any]]:
    sources = {
        "legacy_easy": REPO_ROOT
        / "Temp results"
        / "Easy"
        / "artifacts"
        / "ce_easy_regime",
        "medium_multiplication": REPO_ROOT
        / "Temp results"
        / "Medium"
        / "Multiplication"
        / "artifacts"
        / "ce_multiplication_medium_calibration",
        "medium_s5": REPO_ROOT
        / "Temp results"
        / "Medium"
        / "S5"
        / "artifacts"
        / "ce_s5_medium_calibration",
        "medium_knowledge_filler": REPO_ROOT
        / "Temp results"
        / "Medium"
        / "llama31_8b_knowledge_filler_screen",
        "hard_multiplication_piggy": TEMP_MULT / "multiplication_only_piggyback",
    }
    allowed_names = {
        "REPORT.md",
        "summary.json",
        "pre_post_metrics.csv",
        "pre_post_metrics_mean_std.csv",
        "metrics_by_bucket.csv",
        "paired_comparisons.csv",
        "metrics_by_length.csv",
        "accuracy_by_task_and_filler.csv",
        "component_accuracy.csv",
        "per_seed_metrics.csv",
        "aggregate_metrics.csv",
        "per_seed_contrasts.csv",
        "aggregate_contrasts.csv",
    }
    inventory = []
    destination_root = OUTPUT_ROOT / "preserved_source_summaries"
    for label, source_root in sources.items():
        if not source_root.exists():
            raise FileNotFoundError(source_root)
        for source in sorted(
            path
            for path in source_root.iterdir()
            if path.is_file() and path.name in allowed_names
        ):
            destination = destination_root / label / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inventory.append(
                {
                    "label": label,
                    "original_path": str(source.relative_to(REPO_ROOT)),
                    "preserved_path": str(destination.relative_to(REPO_ROOT)),
                    "sha256": _sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
    # These authoritative summaries already live outside Temp results, but hash them in the same inventory.
    for task in ("s5", "knowledge"):
        for name in (
            "summary.json",
            "per_seed_metrics.csv",
            "aggregate_metrics.csv",
            "per_seed_contrasts.csv",
        ):
            source = REPO_ROOT / "artifacts" / "hard_regime" / task / name
            inventory.append(
                {
                    "label": f"hard_{task}",
                    "original_path": str(source.relative_to(REPO_ROOT)),
                    "preserved_path": str(source.relative_to(REPO_ROOT)),
                    "sha256": _sha256(source),
                    "bytes": source.stat().st_size,
                }
            )
    return inventory


def preserve_final_easy_sources(
    inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy compact final Easy provenance outside directories the user may delete."""
    sources = [
        REPO_ROOT / "artifacts" / "easy_regime" / "summary.json",
        REPO_ROOT / "artifacts" / "easy_regime" / "per_seed_metrics.csv",
        REPO_ROOT / "artifacts" / "easy_regime" / "aggregate_metrics.csv",
        REPO_ROOT / "artifacts" / "easy_regime" / "final_package_manifest.json",
        REPO_ROOT / "artifacts" / "easy_regime" / "analysis" / "analysis_manifest.json",
        REPO_ROOT / "artifacts" / "easy_regime" / "analysis" / "cell_metrics.csv",
        REPO_ROOT
        / "artifacts"
        / "easy_regime"
        / "analysis"
        / "training_diagnostics.csv",
        REPO_ROOT
        / "artifacts"
        / "easy_regime"
        / "analysis"
        / "paired_contrasts_per_seed.csv",
        REPO_ROOT
        / "artifacts"
        / "easy_regime"
        / "analysis"
        / "paired_contrasts_aggregate.csv",
        REPO_ROOT / "experiments" / "easy_regime" / "config.json",
        REPO_ROOT
        / "experiments"
        / "easy_regime"
        / "config_qwen_screened_knowledge.json",
        REPO_ROOT / "generated_data" / "easy_regime" / "split_manifest.json",
        REPO_ROOT
        / "generated_data"
        / "easy_regime_qwen_screened_knowledge_v1"
        / "split_manifest.json",
    ]
    destination_root = OUTPUT_ROOT / "preserved_source_summaries" / "final_easy"
    existing = {
        (row.get("label"), row.get("original_path")): row
        for row in inventory
        if row.get("label") != "final_easy"
    }
    for source in sources:
        relative = source.relative_to(REPO_ROOT)
        destination = destination_root / Path(*relative.parts[1:])
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if destination.exists():
            row = {
                "label": "final_easy",
                "original_path": str(relative),
                "preserved_path": str(destination.relative_to(REPO_ROOT)),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
            existing[(row["label"], row["original_path"])] = row
    return [existing[key] for key in sorted(existing)]


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{100 * float(value):.1f}%"


def _append_figure1_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Task | Model | Method | Private exact | Public exact | Joint exact | Mean CoT words |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['method']} | "
            f"{_fmt(row['private_exact_rate_mean'])} ± {_fmt(row['private_exact_rate_sample_sd'])} | "
            f"{_fmt(row['public_exact_rate_mean'])} ± {_fmt(row['public_exact_rate_sample_sd'])} | "
            f"{_fmt(row['task_success_rate_mean'])} ± {_fmt(row['task_success_rate_sample_sd'])} | "
            f"{float(row['avg_cot_words_mean']):.1f} ± {float(row['avg_cot_words_sample_sd']):.1f} |"
        )


def _append_figure2_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Task | Model | Method | Nominal input length | n | Private exact | Mean CoT words | SD CoT words | Mean generated tokens |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['method']} | {row['nominal_length']} | {row['n']} | {_fmt(row['private_accuracy'])} | {row['mean_cot_words']:.1f} | {row['sample_sd_cot_words']:.1f} | {row['mean_generated_tokens_including_answer_block']:.1f} |"
        )


def write_report(
    hard_aggregate: list[dict[str, Any]],
    hard_chain: list[dict[str, Any]],
    easy_aggregate: list[dict[str, Any]],
    easy_chain: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    hard_count: int,
    easy_count: int,
) -> None:
    lines = [
        "# Figure-data preservation report",
        "",
        "Status: Hard data preserved and final matched Easy data audited and appended on 2026-09-02.",
        "",
        "This report and the adjacent compact CSV files preserve all available Easy and Hard inputs needed for the two supervisor-requested figures, independently of the large Easy result directories and `Temp results`.",
        "",
        "## Requested figures and metric contract",
        "",
        "1. **Accuracy versus task length.** One panel per task/model in the appendix; colors are vanilla, filler, public-only CoT, piggyback, and steg. The primary accuracy is **private exact accuracy** because the mechanism is intended to enable the unshown private computation. Public and joint exact are retained in the CSV for sensitivity plots.",
        "2. **Input length versus chain length.** Piggyback and steg only, faceted by task. The primary chain-length measure is monitor-visible CoT words. Full generated-token count, including the answer block, and character counts are retained as alternatives.",
        "",
        "The committed x-axis definition is task-native nominal length: S5 = swap count, knowledge = fact count, and multiplication = maximum decimal digits among the public/private operands. Literal prompt words and characters remain preserved for sensitivity analyses.",
        "",
        "## Figure reconstruction contract",
        "",
        "For Figure 1, use `combined_figure1_by_nominal_length.csv`: facet by `task` and `model`, map `nominal_length` to x, `private_accuracy_mean_across_seeds` to y, and `method` to color. Use `private_accuracy_sample_sd_across_seeds` for error bars. Public and joint accuracy columns are retained as sensitivity outcomes. The regime-level `combined_figure1_aggregate.csv` is a compact cross-check, not the length-resolved plotting source.",
        "",
        "For Figure 2, use `combined_figure2_by_nominal_length.csv`, restrict to piggyback/steganography (already enforced), facet by task, map `nominal_length` to x and `mean_cot_words` to y, and distinguish method by color. Keep model as separate lines or subfacets. The per-example tables retain prompt length, CoT characters, and generated-token count if the plotting definition changes.",
        "",
        "The Easy x values are S5 = 1 swap, knowledge = 1 fact, and multiplication = 1 or 2 maximum operand digits. The Hard x values are S5 = 10–19 swaps, knowledge = 5 facts, and multiplication = 3 or 4 maximum operand digits. Small multiplication edge bins should be inspected through `n` before interpretation.",
        "",
        "## Coverage and missing cells",
        "",
        f"The figure inputs contain **{hard_count:,} Hard** and **{easy_count:,} Easy v2** trained-response examples. Hard has four methods; Easy has all five methods. Every available cell has three seeds and 1,000 held-out examples per seed.",
        "",
        "The frozen Hard experiment is complete for vanilla, public-only CoT, piggyback, and steganography. **No Hard filler adapters or evaluations exist for any task/model.** A complete five-series Easy–Hard Figure 1 therefore still requires either Hard filler runs or an explicit missing-value treatment.",
        "",
        "Do not merge preserved legacy Easy results with matched Easy v2: their private-question construction differs.",
        "",
        "## Easy v2 validity warning",
        "",
        "The Easy jobs completed correctly. S5 is intentionally nontrivial: it is 73–76% under answer-only training and should be interpreted separately from near-ceiling null claims. Multiplication is near ceiling for both models. Final knowledge is also near ceiling under answer-only training: Qwen uses its screened 20-fact bank and Llama retains the completed 100-fact run. Knowledge steganography remains a channel-learning diagnostic rather than a measure of independent Easy-task capacity.",
        "",
        "## Easy v2 Figure 1 values",
        "",
        "Means ± sample SD across seeds; `n=3,000` examples per aggregate.",
        "",
    ]
    _append_figure1_table(lines, easy_aggregate)
    lines.extend(
        [
            "",
            "## Easy v2 Figure 2 nominal-length values",
            "",
            "Pooled descriptive values for piggyback and steg. Easy multiplication contains both one- and two-digit maxima; the one-digit strata are small and are not balanced length bins.",
            "",
        ]
    )
    _append_figure2_table(lines, easy_chain)
    lines.extend(
        [
            "",
            "## Hard Figure 1 values",
            "",
            "Means ± sample SD across seeds; `n=3,000` examples per aggregate.",
            "",
        ]
    )
    _append_figure1_table(lines, hard_aggregate)
    lines.extend(
        [
            "",
            "## Hard Figure 2 nominal-length values",
            "",
            "Pooled descriptive values across seeds. For multiplication, nominal length is maximum operand decimal digits; prompt-word aggregations are preserved separately.",
            "",
        ]
    )
    _append_figure2_table(lines, hard_chain)
    lines.extend(
        [
            "",
            "## Preserved compact artifacts",
            "",
            "- `artifacts/figure_data_preservation/hard_per_example.csv`: all 72,000 scored Hard examples and alternative input/chain-length fields.",
            "- `hard_figure1_per_seed.csv` and `hard_figure1_aggregate.csv`: plot-ready Hard accuracy values.",
            "- `hard_figure2_by_nominal_length.csv` and `hard_figure2_by_prompt_words.csv`: plot-ready Hard chain values.",
            "- `easy_per_example.csv`: all 90,000 audited Easy examples, copied outside the removable canonical Easy directories.",
            "- `easy_figure1_per_seed.csv` and `easy_figure1_aggregate.csv`: plot-ready Easy accuracy values.",
            "- `easy_figure2_by_nominal_length.csv` and `easy_figure2_by_prompt_words.csv`: plot-ready Easy chain values.",
            "- `combined_figure1_by_nominal_length.csv`: the length-resolved plotting source for Figure 1.",
            "- `combined_figure1_aggregate.csv` and `combined_figure2_by_nominal_length.csv`: regime-level Figure 1 cross-checks and the plotting source for Figure 2.",
            "- `preserved_source_summaries/final_easy/`: final Easy summaries, diagnostics, configurations, and split manifests.",
            "- `preserved_source_summaries/` and `source_inventory.csv`: compact legacy/final summaries and their hashes.",
            "",
            "## Existing Medium and legacy Easy evidence",
            "",
            "The preserved legacy Easy and Medium summaries are appendix/calibration evidence, not matched replacements for Easy v2. The Medium knowledge screen is inference-time filler screening rather than the same CE-trained five-method design.",
            "",
            "## Safe-deletion checklist",
            "",
            "- [x] Recomputed Hard metrics match all 72 authoritative evaluation reports exactly.",
            f"- [x] {hard_count:,} Hard per-example rows preserved outside `Temp results`.",
            f"- [x] {easy_count:,} Easy per-example rows copied into `artifacts/figure_data_preservation`.",
            f"- [x] {len(inventory)} compact source files inventoried with SHA-256 hashes.",
            "- [x] Matched Easy v2 results appended and checked against all 90 evaluation cells.",
            "- [x] Final Easy summaries, diagnostics, configs, and split manifests preserved outside the removable directories.",
            "- [x] The preservation builder can reload Easy solely from the dedicated figure package.",
            "- [ ] Decision made about missing Hard filler cells.",
            "",
            "After backing up the full Easy package, deleting `artifacts/easy_regime`, `generated_data/easy_regime`, `generated_data/easy_regime_eval_responses`, and `generated_data/easy_regime_qwen_screened_knowledge_v1` is safe **for these two specified figures**. Keep `artifacts/figure_data_preservation` and this report. Deletion still removes adapters, raw prompts/completions, and the ability to perform new qualitative analyses not represented by the preserved columns.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if TEMP_MULT.exists():
        hard_examples, hard_seed = extract_hard()
        inventory = preserve_small_summaries()
    else:
        hard_examples, hard_seed = load_preserved_hard()
        inventory = load_preserved_inventory()
    inventory = preserve_final_easy_sources(inventory)
    hard_aggregate = aggregate_seed_rows(hard_seed, HARD_METHODS, "hard")
    hard_accuracy_by_length = aggregate_accuracy_by_length(hard_examples, "hard")
    hard_chain = aggregate_chain(hard_examples, "nominal_length", "hard")
    hard_chain_words = aggregate_chain(hard_examples, "prompt_words", "hard")
    easy_examples, easy_seed = load_easy()
    easy_aggregate = aggregate_seed_rows(easy_seed, EASY_METHODS, "easy_v2")
    easy_accuracy_by_length = aggregate_accuracy_by_length(easy_examples, "easy_v2")
    easy_chain = aggregate_chain(easy_examples, "nominal_length", "easy_v2")
    easy_chain_words = aggregate_chain(easy_examples, "prompt_words", "easy_v2")
    _write_csv(OUTPUT_ROOT / "hard_per_example.csv", hard_examples)
    _write_csv(OUTPUT_ROOT / "hard_figure1_per_seed.csv", hard_seed)
    _write_csv(OUTPUT_ROOT / "hard_figure1_aggregate.csv", hard_aggregate)
    _write_csv(
        OUTPUT_ROOT / "hard_figure1_by_nominal_length.csv", hard_accuracy_by_length
    )
    _write_csv(OUTPUT_ROOT / "hard_figure2_by_nominal_length.csv", hard_chain)
    _write_csv(OUTPUT_ROOT / "hard_figure2_by_prompt_words.csv", hard_chain_words)
    _write_csv(OUTPUT_ROOT / "easy_per_example.csv", easy_examples)
    _write_csv(OUTPUT_ROOT / "easy_figure1_per_seed.csv", easy_seed)
    _write_csv(OUTPUT_ROOT / "easy_figure1_aggregate.csv", easy_aggregate)
    _write_csv(
        OUTPUT_ROOT / "easy_figure1_by_nominal_length.csv", easy_accuracy_by_length
    )
    _write_csv(OUTPUT_ROOT / "easy_figure2_by_nominal_length.csv", easy_chain)
    _write_csv(OUTPUT_ROOT / "easy_figure2_by_prompt_words.csv", easy_chain_words)
    _write_csv(
        OUTPUT_ROOT / "combined_figure1_aggregate.csv", easy_aggregate + hard_aggregate
    )
    _write_csv(
        OUTPUT_ROOT / "combined_figure1_by_nominal_length.csv",
        easy_accuracy_by_length + hard_accuracy_by_length,
    )
    _write_csv(
        OUTPUT_ROOT / "combined_figure2_by_nominal_length.csv", easy_chain + hard_chain
    )
    _write_csv(OUTPUT_ROOT / "source_inventory.csv", inventory)
    write_report(
        hard_aggregate,
        hard_chain,
        easy_aggregate,
        easy_chain,
        inventory,
        len(hard_examples),
        len(easy_examples),
    )
    primary_outputs = [
        OUTPUT_ROOT / "hard_per_example.csv",
        OUTPUT_ROOT / "hard_figure1_per_seed.csv",
        OUTPUT_ROOT / "hard_figure1_aggregate.csv",
        OUTPUT_ROOT / "hard_figure1_by_nominal_length.csv",
        OUTPUT_ROOT / "hard_figure2_by_nominal_length.csv",
        OUTPUT_ROOT / "hard_figure2_by_prompt_words.csv",
        OUTPUT_ROOT / "easy_per_example.csv",
        OUTPUT_ROOT / "easy_figure1_per_seed.csv",
        OUTPUT_ROOT / "easy_figure1_aggregate.csv",
        OUTPUT_ROOT / "easy_figure1_by_nominal_length.csv",
        OUTPUT_ROOT / "easy_figure2_by_nominal_length.csv",
        OUTPUT_ROOT / "easy_figure2_by_prompt_words.csv",
        OUTPUT_ROOT / "combined_figure1_aggregate.csv",
        OUTPUT_ROOT / "combined_figure1_by_nominal_length.csv",
        OUTPUT_ROOT / "combined_figure2_by_nominal_length.csv",
        OUTPUT_ROOT / "source_inventory.csv",
        REPORT_PATH,
    ]
    manifest = {
        "schema_version": 2,
        "created_on": "2026-09-02",
        "valid": True,
        "hard_per_example_rows": len(hard_examples),
        "easy_per_example_rows": len(easy_examples),
        "hard_per_seed_rows": len(hard_seed),
        "easy_per_seed_rows": len(easy_seed),
        "hard_aggregate_rows": len(hard_aggregate),
        "easy_aggregate_rows": len(easy_aggregate),
        "inventory_files": len(inventory),
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in primary_outputs
        ],
    }
    (OUTPUT_ROOT / "preservation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**manifest, "report": str(REPORT_PATH.relative_to(REPO_ROOT))}, indent=2
        )
    )


if __name__ == "__main__":
    main()
