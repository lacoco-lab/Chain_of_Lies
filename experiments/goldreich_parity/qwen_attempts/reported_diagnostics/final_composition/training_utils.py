"""Minimal shared training utilities for the final Goldreich experiment."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def prepare_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt_text"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(encoding["input_ids"])
    full_ids = list(
        tokenizer(prompt + row["supervised_suffix"], add_special_tokens=False)[
            "input_ids"
        ]
    )
    if (
        full_ids[: len(prompt_ids)] != prompt_ids
        or len(full_ids) != len(prompt_ids) + 1
    ):
        raise ValueError(
            f"Target is not exactly one appended token: {row['experiment_id']}"
        )
    groups = [list(map(int, group)) for group in row["spec"]["attention_groups"]]
    seed_positions = []
    if groups:
        offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
        for label, value in zip("ABCDEFGHIJKLMNOP", str(row["spec"]["prg_seed"])):
            marker = f"{label}={value}"
            character = prompt.find(marker) + len(marker) - 1
            hits = [
                index
                for index, (start, end) in enumerate(offsets)
                if start <= character < end and end > start
            ]
            if character < len(marker) - 1 or len(hits) != 1:
                raise ValueError(f"Cannot locate {marker}: {row['experiment_id']}")
            seed_positions.append(hits[0])
        if len(set(seed_positions)) != 16:
            raise ValueError(
                f"Seed values are not distinct tokens: {row['experiment_id']}"
            )
    return {
        "id": row["experiment_id"],
        "task": row["spec"]["task"],
        "input_ids": prompt_ids,
        "target_id": full_ids[-1],
        "gold_bit": int(row["spec"]["gold_bit"]),
        "seed_positions": seed_positions,
        "attention_groups": groups,
    }


def load_data(tokenizer: Any, data_root: Path) -> dict[str, list[dict[str, Any]]]:
    names = (
        "local_replay",
        "local_validation",
        "supplied_update_train",
        "supplied_update_validation",
    )
    names += tuple(
        f"{task}_{split}"
        for task in (
            "derive_previous",
            "derive_current",
            "mask_delta",
            "full_zero",
            "full_one",
            "full_two",
            "full_update",
        )
        for split in ("train", "validation")
    )
    return {
        name: [
            prepare_row(tokenizer, row)
            for row in read_jsonl(data_root / f"{name}.jsonl")
        ]
        for name in names
    }


def binary_tokens(data: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    for rows in data.values():
        for row in rows:
            previous = mapping.setdefault(row["gold_bit"], row["target_id"])
            if previous != row["target_id"]:
                raise ValueError("Context-dependent binary target token")
    if set(mapping) != {0, 1}:
        raise ValueError(f"Missing binary targets: {mapping}")
    return mapping[0], mapping[1]


def make_batch(rows: list[dict[str, Any]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), maximum), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    queries, targets = [], []
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[index, :length] = torch.tensor(row["input_ids"], device=device)
        attention_mask[index, :length] = 1
        queries.append(length - 1)
        targets.append(row["target_id"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_positions": torch.tensor(queries, dtype=torch.long, device=device),
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
        "rows": rows,
    }


def attention_loss(
    attentions: tuple[Any, ...],
    batch: dict[str, Any],
    layer_count: int,
    head_count: int,
    coverage_weight: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    if not attentions or any(layer is None for layer in attentions[-layer_count:]):
        raise RuntimeError("Model did not return attention weights")
    losses, masses, ratios = [], [], []
    layers = attentions[-layer_count:]
    for row_index, row in enumerate(batch["rows"]):
        if not row["attention_groups"]:
            continue
        query = int(batch["query_positions"][row_index].item())
        values = torch.stack([layer[row_index, :, query, :] for layer in layers], dim=0)
        seed_values = values[..., row["seed_positions"]].reshape(-1, 16)
        for raw_group in row["attention_groups"]:
            group = torch.tensor(
                sorted(set(raw_group)), dtype=torch.long, device=seed_values.device
            )
            absolute = seed_values[:, group].sum(dim=-1)
            top = torch.topk(absolute, k=min(head_count, absolute.numel())).indices
            chosen = seed_values[top]
            distribution = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mean_distribution = distribution.mean(dim=0)
            target = torch.zeros_like(mean_distribution)
            target[group] = 1.0 / len(group)
            focus = -torch.log(absolute[top].mean().clamp_min(1e-8))
            coverage = -(target * torch.log(mean_distribution.clamp_min(1e-8))).sum()
            losses.append(focus + coverage_weight * coverage)
            masses.append(absolute[top].mean().detach())
            ratios.append(mean_distribution[group].sum().detach())
    if not losses:
        zero = batch["input_ids"].new_zeros((), dtype=torch.float32)
        return zero, {"selected_attention_mass": 0.0, "selected_seed_ratio": 0.0}
    return torch.stack(losses).mean(), {
        "selected_attention_mass": float(torch.stack(masses).mean().cpu()),
        "selected_seed_ratio": float(torch.stack(ratios).mean().cpu()),
    }


def evaluate(
    model: Any,
    rows: list[dict[str, Any]],
    pad_id: int,
    device: Any,
    batch_size: int,
    bit_tokens: tuple[int, int],
) -> dict[str, float]:
    import torch

    model.eval()
    total = greedy_correct = binary_correct = 0
    ce_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = make_batch(rows[start : start + batch_size], pad_id, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
            indices = torch.arange(len(batch["rows"]), device=device)
            logits = outputs.logits[indices, batch["query_positions"]]
            greedy = logits.argmax(dim=-1)
            candidates = logits[:, list(bit_tokens)].argmax(dim=-1)
            binary = torch.tensor(bit_tokens, device=device)[candidates]
            greedy_correct += int((greedy == batch["targets"]).sum())
            binary_correct += int((binary == batch["targets"]).sum())
            ce_sum += float(
                torch.nn.functional.cross_entropy(
                    logits, batch["targets"], reduction="sum"
                ).cpu()
            )
            total += len(batch["rows"])
    return {
        "n": total,
        "greedy_accuracy": greedy_correct / total,
        "binary_accuracy": binary_correct / total,
        "bit_ce": ce_sum / total,
    }


def attention_diagnostic(
    model: Any,
    rows: list[dict[str, Any]],
    pad_id: int,
    device: Any,
    batch_size: int,
    config: dict[str, Any],
) -> dict[str, float] | None:
    import torch

    rows = [row for row in rows if row["attention_groups"]]
    if not rows:
        return None
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = make_batch(rows[start : start + batch_size], pad_id, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            loss, metrics = attention_loss(
                outputs.attentions,
                batch,
                int(config["attention_layers"]),
                int(config["attention_heads"]),
                float(config["attention_coverage_weight"]),
            )
            values.append((len(batch["rows"]), float(loss.cpu()), metrics))
    total = sum(n for n, _, _ in values)
    return {
        "n": total,
        "attention_aux": sum(n * loss for n, loss, _ in values) / total,
        "selected_attention_mass": sum(
            n * item["selected_attention_mass"] for n, _, item in values
        )
        / total,
        "selected_seed_ratio": sum(
            n * item["selected_seed_ratio"] for n, _, item in values
        )
        / total,
    }


def load_frozen_stack(
    config: dict[str, Any], adapter_paths: list[Path]
) -> tuple[Any, Any, Any]:
    """Merge every established adapter, freeze the stack, and add one fresh rank-32 LoRA."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    for path in adapter_paths:
        if (
            not (path / "adapter_config.json").exists()
            or not (path / "adapter_model.safetensors").exists()
        ):
            raise FileNotFoundError(f"Missing established adapter: {path}")
        model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    fresh = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=list(config["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [
        name for name, parameter in fresh.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only the fresh composition LoRA may be trainable")
    device = torch.device("cuda")
    fresh.to(device)
    return tokenizer, fresh, device


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


def save_adapter(model: Any, directory: Path, metadata: dict[str, Any]) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)
    (directory / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
