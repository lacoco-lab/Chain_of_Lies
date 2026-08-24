#!/usr/bin/env python3
"""Run one paper-style knowledge filler sweep on Llama 3.1 8B."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from experiment import (
    TASKS,
    build_component_messages,
    build_messages,
    component_probes,
    load_task_items,
    make_dot_filler,
    score_answer,
)


def _render(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_batches(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: Any,
) -> list[tuple[str, int]]:
    import torch

    outputs: list[tuple[str, int]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        prompt_width = int(encoded.input_ids.shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, prompt_tokens in zip(generated, encoded.attention_mask.sum(dim=1).tolist()):
            completion_ids = row[prompt_width:]
            raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            outputs.append((raw_text, int(prompt_tokens)))
        print(f"[inference] {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)
    return outputs


def run(*, config_path: Path, task: str, output_dir: Path) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    random.seed(int(config["seed"]))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(config["model_id"])
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required because the Llama checkpoint is gated.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    items = load_task_items(config, task)
    filler_lengths = [int(value) for value in config["filler_lengths"]]
    rows: list[dict[str, Any]] = []
    for k in filler_lengths:
        prompts = [_render(tokenizer, build_messages(item, k)) for item in items]
        generations = _generate_batches(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=int(config["batch_size"]),
            max_new_tokens=int(config["max_new_tokens"]),
            device=device,
        )
        filler_model_tokens = len(
            tokenizer(make_dot_filler(k), add_special_tokens=False).input_ids
        ) if k else 0
        for item, (raw_text, prompt_tokens) in zip(items, generations):
            predicted, correct = score_answer(raw_text, item["answer"], item["answer_type"])
            rows.append(
                {
                    "record_kind": "main",
                    "item_id": item["item_id"],
                    "task": task,
                    "subtask": item["subtask"],
                    "filler_length": k,
                    "filler_occurrences": 1 + len(item["few_shot"]),
                    "filler_model_tokens_per_region": filler_model_tokens,
                    "prompt_tokens": prompt_tokens,
                    "question": item["question"],
                    "expected": item["answer"],
                    "predicted": predicted,
                    "correct": correct,
                    "raw_response": raw_text,
                    "metadata": item["metadata"],
                }
            )

    probes = component_probes(items, task)
    probe_prompts = [_render(tokenizer, build_component_messages(probe)) for probe in probes]
    probe_generations = _generate_batches(
        model=model,
        tokenizer=tokenizer,
        prompts=probe_prompts,
        batch_size=int(config["batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        device=device,
    )
    probe_rows = []
    for probe, (raw_text, prompt_tokens) in zip(probes, probe_generations):
        predicted, correct = score_answer(raw_text, probe["answer"], probe["answer_type"])
        probe_rows.append(
            {
                "record_kind": "component",
                "item_id": probe["probe_id"],
                "task": task,
                "subtask": probe["probe_type"],
                "filler_length": 0,
                "prompt_tokens": prompt_tokens,
                "question": probe["question"],
                "expected": probe["answer"],
                "predicted": predicted,
                "correct": correct,
                "raw_response": raw_text,
            }
        )

    output = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model_id": model_id,
        "task": task,
        "seed": config["seed"],
        "filler_type": "dots",
        "filler_lengths": filler_lengths,
        "few_shot_per_task": config["few_shot_per_task"],
        "main_n": len(items),
        "component_n": len(probes),
        "decoding": {"do_sample": False, "max_new_tokens": config["max_new_tokens"]},
        "rows": rows,
        "component_rows": probe_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_data/llama31_8b_knowledge_filler_responses"),
    )
    args = parser.parse_args()
    run(config_path=args.config, task=args.task, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
