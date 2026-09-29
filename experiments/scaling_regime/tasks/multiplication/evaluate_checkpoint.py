#!/usr/bin/env python3
"""Evaluate one multiplication adapter by length without redundant baseline runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.evaluation import run_variant_inference
from chain_of_lies.evaluation.rewards import summarize_variant_results


def _checkpoint(adapter_root: Path, checkpoint: str) -> Path:
    metadata_path, history_path = (
        adapter_root / "training_metadata.json",
        adapter_root / "train_history.json",
    )
    checkpoint_root = adapter_root / checkpoint
    if (
        not metadata_path.is_file()
        or not history_path.is_file()
        or not (checkpoint_root / "adapter_config.json").is_file()
    ):
        raise RuntimeError(f"Incomplete training artifacts under {adapter_root}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history or int(history[-1].get("step", 0)) < int(metadata.get("steps", 0)):
        raise RuntimeError(
            f"Training did not reach its final step under {adapter_root}."
        )
    if int(metadata.get("train_examples_seen", 0)) < int(
        metadata.get("target_train_examples_seen", 0)
    ):
        raise RuntimeError(f"Training coverage is incomplete under {adapter_root}.")
    return checkpoint_root


def evaluate(
    *,
    adapter_root: Path,
    variant: str,
    prompts_dir: Path,
    responses_root: Path,
    report_path: Path,
    checkpoint: str,
    max_new_tokens: int,
    batch_size: int,
    model: str,
    seed: int,
    condition: str,
    length: int,
) -> dict:
    checkpoint_root = _checkpoint(adapter_root, checkpoint)
    response_dir = responses_root / checkpoint / "trained" / variant
    run_variant_inference(
        prompts_dir=prompts_dir,
        responses_dir=response_dir,
        model_id=str(checkpoint_root),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        resume=True,
        batch_size=batch_size,
    )
    metrics = summarize_variant_results(prompts_dir, response_dir)
    output = {
        "schema_version": 1,
        "model": model,
        "seed": seed,
        "condition": condition,
        "length": length,
        "variant": variant,
        "checkpoint": checkpoint,
        "adapter_root": str(adapter_root),
        "prompts_dir": str(prompts_dir),
        "responses_dir": str(response_dir),
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "private_exact_rate": metrics["private_exact_rate"],
            },
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--prompts-dir", type=Path, required=True)
    p.add_argument("--responses-root", type=Path, required=True)
    p.add_argument("--report-path", type=Path, required=True)
    p.add_argument("--checkpoint", default="ckpt_final")
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--length", type=int, required=True)
    a = p.parse_args()
    evaluate(
        adapter_root=a.adapter_root,
        variant=a.variant,
        prompts_dir=a.prompts_dir,
        responses_root=a.responses_root,
        report_path=a.report_path,
        checkpoint=a.checkpoint,
        max_new_tokens=a.max_new_tokens,
        batch_size=a.batch_size,
        model=a.model,
        seed=a.seed,
        condition=a.condition,
        length=a.length,
    )
