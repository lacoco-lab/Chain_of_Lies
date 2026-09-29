#!/usr/bin/env python3
"""Evaluate one task/model/seed cell using the transferred final Hard adapters."""

from __future__ import annotations

import argparse
import csv
import json
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

from chain_of_lies.evaluation import run_variant_inference
from chain_of_lies.evaluation.rewards import (
    extract_monitor_visible_reasoning,
    score_completion,
)

METHODS = ("vanilla", "public_only_cot", "piggyback", "steganography")


def checkpoint_dir(
    hard_root: Path, task: str, model: str, seed: int, method: str
) -> Path:
    if task == "multiplication":
        if method == "steganography":
            return (
                hard_root
                / "Multiplication"
                / "hard_regime"
                / "confirmatory"
                / model
                / f"seed_{seed}"
                / "steganography"
                / "local_channel_cot"
                / "arith_steg_local_invisible"
                / "ckpt_final"
            )
        mode = "answer_only" if method == "vanilla" else "public_cot"
        variant = (
            "arith_piggyback" if method == "piggyback" else "arith_piggyback_control"
        )
        return (
            hard_root
            / "Multiplication"
            / "multiplication_only_piggyback"
            / model
            / f"seed_{seed}"
            / mode
            / variant
            / "ckpt_final"
        )
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
    return (
        hard_root
        / task
        / model
        / f"seed_{seed}"
        / family
        / mode
        / variant
        / "ckpt_final"
    )


def _validate_checkpoint(path: Path) -> None:
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "training_metadata.json",
        "train_history.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {path}; missing {missing}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _score(
    prompts_dir: Path,
    responses_dir: Path,
    task: str,
    model: str,
    seed: int,
    method: str,
) -> list[dict[str, Any]]:
    prompt_paths = sorted(prompts_dir.glob("*.json"))
    response_paths = sorted(responses_dir.glob("*.json"))
    if len(prompt_paths) != len(response_paths):
        raise RuntimeError(
            f"Incomplete responses for {task}/{model}/seed_{seed}/{method}: {len(response_paths)}/{len(prompt_paths)}"
        )
    rows: list[dict[str, Any]] = []
    for prompt_path in prompt_paths:
        response_path = responses_dir / prompt_path.name
        if not response_path.is_file():
            raise FileNotFoundError(response_path)
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        response = json.loads(response_path.read_text(encoding="utf-8"))
        raw = str(response["raw_text"])
        scored = score_completion(prompt, raw)
        reasoning = extract_monitor_visible_reasoning(prompt, raw)
        rows.append(
            {
                "task": task,
                "model": model,
                "seed": seed,
                "method": method,
                "experiment_id": prompt["experiment_id"],
                "length": int(prompt["spec"]["evaluation_length"]),
                "public_correct": int(bool(scored.flags["public_correct"])),
                "private_correct": int(bool(scored.flags["private_correct"])),
                "joint_correct": int(bool(scored.task_success)),
                "format_ok": int(bool(scored.format_ok)),
                "concealment_ok": int(bool(scored.concealment_ok)),
                "payload_decode_correct": (
                    int(bool(scored.flags.get("payload_decode_correct")))
                    if scored.flags.get("payload_target") is not None
                    else ""
                ),
                "cot_words": scored.cot_word_count,
                "cot_chars": len(reasoning),
                "generated_tokens_including_answer_block": len(
                    response.get("generated_token_ids") or []
                ),
            }
        )
    return rows


def evaluate_cell(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    task_config = config["tasks"][args.task]
    all_rows: list[dict[str, Any]] = []
    for method in METHODS:
        checkpoint = checkpoint_dir(
            args.hard_artifacts_root, args.task, args.model, args.seed, method
        )
        _validate_checkpoint(checkpoint)
        prompts = args.prompts_root / args.task / f"seed_{args.seed}" / method
        expected = len(task_config["lengths"]) * int(task_config["examples_per_length"])
        if len(list(prompts.glob("*.json"))) != expected:
            raise RuntimeError(f"Expected {expected} prompts in {prompts}")
        responses = (
            args.responses_root / args.task / args.model / f"seed_{args.seed}" / method
        )
        run_variant_inference(
            prompts,
            responses,
            model_id=str(checkpoint),
            max_new_tokens=int(task_config["max_new_tokens"]),
            temperature=0.0,
            do_sample=False,
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )
        all_rows.extend(
            _score(prompts, responses, args.task, args.model, args.seed, method)
        )

    output_root = args.output_root / args.task / args.model / f"seed_{args.seed}"
    _write_csv(output_root / "per_example.csv", all_rows)
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        for length in task_config["lengths"]:
            subset = [
                row
                for row in all_rows
                if row["method"] == method and row["length"] == int(length)
            ]
            summaries.append(
                {
                    "task": args.task,
                    "model": args.model,
                    "seed": args.seed,
                    "method": method,
                    "length": int(length),
                    "n": len(subset),
                    "public_exact_rate": statistics.fmean(
                        row["public_correct"] for row in subset
                    ),
                    "private_exact_rate": statistics.fmean(
                        row["private_correct"] for row in subset
                    ),
                    "task_success_rate": statistics.fmean(
                        row["joint_correct"] for row in subset
                    ),
                    "avg_cot_words": statistics.fmean(
                        row["cot_words"] for row in subset
                    ),
                    "avg_generated_tokens_including_answer_block": statistics.fmean(
                        row["generated_tokens_including_answer_block"] for row in subset
                    ),
                }
            )
    _write_csv(output_root / "per_length_metrics.csv", summaries)
    print(
        f"Complete: {args.task}/{args.model}/seed_{args.seed}; {len(all_rows)} scored responses"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=("multiplication", "s5", "knowledge")
    )
    parser.add_argument("--model", required=True, choices=("qwen", "llama"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--hard-artifacts-root", type=Path, default=Path("Hard_regime"))
    parser.add_argument(
        "--prompts-root",
        type=Path,
        default=Path("generated_data/hard_checkpoint_length_sweep"),
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("generated_data/hard_checkpoint_length_sweep_eval_responses"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/hard_checkpoint_length_sweep"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    evaluate_cell(parser.parse_args())
