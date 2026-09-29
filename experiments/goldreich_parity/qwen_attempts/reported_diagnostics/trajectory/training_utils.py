"""Small training and evaluation utilities for encrypted trajectories."""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

TRACE_LINE = re.compile(
    r"\(use Input: x[A-Z]+ x[A-Z]+; use Seed: [A-P](?: [A-P]){4}\) C=([01])"
)
ANSWER_LINE = re.compile(r"ANSWER=([01])")


def parse_completion(text: str) -> tuple[list[int], int] | None:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "COT:":
        return None
    answer = ANSWER_LINE.fullmatch(lines[-1])
    trace = [TRACE_LINE.fullmatch(line) for line in lines[1:-1]]
    if answer is None or not trace or any(match is None for match in trace):
        return None
    return [int(match.group(1)) for match in trace if match is not None], int(
        answer.group(1)
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def prepare_row(
    tokenizer: Any, row: dict[str, Any], format_weight: float, bit_weight: float
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt_text"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    target_text = str(row["supervised_suffix"])
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    encoded = tokenizer(
        prompt + target_text, add_special_tokens=False, return_offsets_mapping=True
    )
    full_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Target changed the prompt tokenization: {row['experiment_id']}"
        )
    if (
        row["task_type"] != "goldreich_trajectory"
        and len(full_ids) != len(prompt_ids) + 1
    ):
        raise ValueError(
            f"Prerequisite target is not one token: {row['experiment_id']}"
        )
    weights = [0.0] * len(full_ids)
    for index in range(len(prompt_ids), len(full_ids)):
        weights[index] = format_weight
    bit_positions = []
    for match in re.finditer(r"[01]", target_text):
        character = len(prompt) + match.start()
        hits = [
            index
            for index, (start, end) in enumerate(offsets)
            if start <= character < end and end > start
        ]
        if len(hits) != 1:
            raise ValueError(f"Cannot locate target bit in {row['experiment_id']}")
        weights[hits[0]] = bit_weight
        bit_positions.append(hits[0])
    states = len(row["spec"].get("encrypted_states", []))
    expected_bits = states + 1 if row["task_type"] == "goldreich_trajectory" else 1
    if len(set(bit_positions)) != expected_bits:
        raise ValueError(f"Wrong variable-bit token count in {row['experiment_id']}")
    return {
        "id": row["experiment_id"],
        "task": row["spec"].get("task", row["task_type"]),
        "input_ids": full_ids,
        "prompt_ids": prompt_ids,
        "loss_weights": weights,
        "target_text": target_text,
        "target_token_count": len(full_ids) - len(prompt_ids),
        "first_target_id": full_ids[len(prompt_ids)],
        "encrypted_states": list(map(int, row["spec"].get("encrypted_states", []))),
        "final_parity": int(
            row["spec"]["final_parity"]
            if "final_parity" in row["spec"]
            else row["spec"]["gold_bit"]
        ),
        "length": int(row["spec"].get("length", 0)),
    }


def load_rows(
    tokenizer: Any, path: Path, format_weight: float, bit_weight: float
) -> list[dict[str, Any]]:
    return [
        prepare_row(tokenizer, row, format_weight, bit_weight)
        for row in read_jsonl(path)
    ]


def load_frozen_stack(
    config: dict[str, Any], adapters: list[Path]
) -> tuple[Any, Any, Any]:
    """Merge the robust prefix, but keep the successful final LoRA active.

    Merging the final composition LoRA into bfloat16 weights measurably weakens
    it. The last adapter is therefore loaded as the sole trainable adapter and
    continued on trajectories; prerequisite replay prevents forgetting.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(adapters) < 2:
        raise ValueError("Expected at least one frozen adapter and one active adapter")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    for adapter in adapters:
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / filename).is_file():
                raise FileNotFoundError(f"Missing adapter file: {adapter / filename}")
    for adapter in adapters[:-1]:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    active_adapter = adapters[-1]
    active_config = json.loads(
        (active_adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if int(active_config["r"]) != int(config["lora_rank"]):
        raise ValueError("The active composition adapter has the wrong rank")
    model = PeftModel.from_pretrained(model, str(active_adapter), is_trainable=True)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError(
            "Only the active composition/trajectory LoRA may be trainable"
        )
    device = torch.device("cuda")
    model.to(device)
    return tokenizer, model, device


def make_train_batch(
    rows: list[dict[str, Any]], pad_id: int, device: Any
) -> dict[str, Any]:
    import torch

    maximum = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    weights = torch.zeros((len(rows), maximum), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[index, :length] = torch.tensor(row["input_ids"], device=device)
        attention[index, :length] = 1
        weights[index, :length] = torch.tensor(row["loss_weights"], device=device)
    return {"input_ids": ids, "attention_mask": attention, "loss_weights": weights}


def weighted_causal_loss(logits: Any, input_ids: Any, weights: Any) -> Any:
    import torch

    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape[0], -1)
    shifted = weights[:, 1:]
    return (losses * shifted).sum() / shifted.sum().clamp_min(1.0)


def _prompt_batch(
    rows: list[dict[str, Any]], pad_id: int, device: Any
) -> tuple[Any, Any]:
    import torch

    maximum = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        values = row["prompt_ids"]
        ids[index, maximum - len(values) :] = torch.tensor(values, device=device)
        attention[index, maximum - len(values) :] = 1
    return ids, attention


def evaluate_prerequisite(
    model: Any, rows: list[dict[str, Any]], pad_id: int, device: Any, batch_size: int
) -> dict[str, float]:
    import torch

    correct = total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            current = rows[start : start + batch_size]
            # Reproduce the established single-step evaluator exactly. Left
            # padding changed Qwen's batched logits and made the known-good
            # adapter appear several points worse than its saved metrics.
            maximum = max(len(row["prompt_ids"]) for row in current)
            ids = torch.full(
                (len(current), maximum), pad_id, dtype=torch.long, device=device
            )
            attention = torch.zeros_like(ids)
            queries = []
            for index, row in enumerate(current):
                length = len(row["prompt_ids"])
                ids[index, :length] = torch.tensor(row["prompt_ids"], device=device)
                attention[index, :length] = 1
                queries.append(length - 1)
            outputs = model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
            indices = torch.arange(len(current), device=device)
            query_positions = torch.tensor(queries, dtype=torch.long, device=device)
            predicted = outputs.logits[indices, query_positions].argmax(dim=-1)
            targets = torch.tensor(
                [row["first_target_id"] for row in current], device=device
            )
            correct += int((predicted == targets).sum())
            total += len(current)
    return {"n": total, "accuracy": correct / total}


def evaluate_trajectory(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    pad_id: int,
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    totals = {
        "trace_bits": 0,
        "trace_bits_correct": 0,
        "trace_exact": 0,
        "answer_exact": 0,
        "joint_exact": 0,
        "strict": 0,
        "output_exact": 0,
    }
    samples = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            current = rows[start : start + batch_size]
            ids, attention = _prompt_batch(current, pad_id, device)
            counts = {row["target_token_count"] for row in current}
            if len(counts) != 1:
                raise ValueError("A generation batch must have one target token length")
            target_tokens = counts.pop()
            generated = model.generate(
                input_ids=ids,
                attention_mask=attention,
                min_new_tokens=target_tokens,
                max_new_tokens=target_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
            )
            for index, row in enumerate(current):
                text = tokenizer.decode(
                    generated[index, ids.shape[1] :], skip_special_tokens=True
                ).strip()
                parsed = parse_completion(text)
                expected_trace = row["encrypted_states"]
                trace = parsed[0] if parsed else []
                answer = parsed[1] if parsed else None
                length_ok = len(trace) == len(expected_trace)
                bit_correct = (
                    sum(a == b for a, b in zip(trace, expected_trace))
                    if length_ok
                    else 0
                )
                trace_exact = length_ok and trace == expected_trace
                answer_exact = answer == row["final_parity"]
                totals["trace_bits"] += len(expected_trace)
                totals["trace_bits_correct"] += bit_correct
                totals["trace_exact"] += int(trace_exact)
                totals["answer_exact"] += int(answer_exact)
                totals["joint_exact"] += int(trace_exact and answer_exact)
                totals["strict"] += int(parsed is not None)
                totals["output_exact"] += int(text == row["target_text"])
                if len(samples) < 8:
                    samples.append(
                        {
                            "id": row["id"],
                            "expected": row["target_text"],
                            "generated": text,
                        }
                    )
    n = len(rows)
    return {
        "n": n,
        "trace_bit_accuracy": totals["trace_bits_correct"] / totals["trace_bits"],
        "trace_exact": totals["trace_exact"] / n,
        "final_parity_accuracy": totals["answer_exact"] / n,
        "joint_exact": totals["joint_exact"] / n,
        "strict_format": totals["strict"] / n,
        "output_exact": totals["output_exact"] / n,
        "samples": samples,
    }


def epoch_batches(
    groups: list[list[dict[str, Any]]], batch_size: int, rng: random.Random
) -> list[list[dict[str, Any]]]:
    batches = []
    for group in groups:
        copied = group.copy()
        rng.shuffle(copied)
        batches.extend(
            copied[start : start + batch_size]
            for start in range(0, len(copied), batch_size)
        )
    rng.shuffle(batches)
    return batches


def sample(
    rows: list[dict[str, Any]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    return rng.sample(rows, min(n, len(rows)))


def clone_trainable(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(model: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[name].to(parameter.device))


def save_adapter(model: Any, directory: Path, metadata: dict[str, Any]) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)
    (directory / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
