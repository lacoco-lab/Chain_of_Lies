#!/usr/bin/env python3
"""Train the one-bit Goldreich lookup with optional seed-position attention loss."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def chat_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["prompt_text"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def token_at_character(offsets: list[tuple[int, int]], character: int) -> int:
    hits = [
        index
        for index, (start, end) in enumerate(offsets)
        if start <= character < end and end > start
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one token at character {character}, found {hits}")
    return hits[0]


def prepare_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = chat_prompt(tokenizer, row)
    prompt_encoding = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    prompt_ids = list(prompt_encoding["input_ids"])
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
            "The one-character target must append exactly one token without changing prompt tokens; "
            f"prompt={len(prompt_ids)} full={len(full_ids)} id={row['experiment_id']}"
        )
    offsets = [tuple(pair) for pair in prompt_encoding["offset_mapping"]]
    seed = str(row["spec"]["prg_seed"])
    seed_positions = []
    for label, value in zip("ABCDEFGHIJKLMNOP", seed):
        marker = f"{label}={value}"
        marker_start = prompt.find(marker)
        if marker_start < 0:
            raise ValueError(f"Missing {marker} in {row['experiment_id']}")
        seed_positions.append(
            token_at_character(offsets, marker_start + len(marker) - 1)
        )
    if len(set(seed_positions)) != 16:
        raise ValueError(
            f"Seed values did not map to 16 distinct tokens: {seed_positions}"
        )
    return {
        "id": row["experiment_id"],
        "input_ids": prompt_ids,
        "target_id": full_ids[-1],
        "gold_bit": int(row["spec"]["gold_bit"]),
        "seed_positions": seed_positions,
        "selected_slots": list(map(int, row["spec"]["edge"])),
    }


def make_batch(rows: list[dict[str, Any]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), maximum), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(rows), maximum), dtype=torch.long, device=device)
    query_positions, targets = [], []
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[index, :length] = torch.tensor(row["input_ids"], device=device)
        attention_mask[index, :length] = 1
        query_positions.append(length - 1)
        targets.append(row["target_id"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_positions": torch.tensor(
            query_positions, dtype=torch.long, device=device
        ),
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
        "rows": rows,
    }


def attention_auxiliary_loss(
    attentions: tuple[Any, ...],
    batch: dict[str, Any],
    layer_count: int,
    head_count: int,
    coverage_weight: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    if not attentions or any(layer is None for layer in attentions[-layer_count:]):
        raise RuntimeError("The model did not return differentiable attention weights")
    losses, absolute_masses, selected_ratios = [], [], []
    selected_layers = attentions[-layer_count:]
    for row_index, row in enumerate(batch["rows"]):
        query = int(batch["query_positions"][row_index].item())
        # [layers, heads, keys] at the query that predicts the mask bit.
        values = torch.stack(
            [layer[row_index, :, query, :] for layer in selected_layers], dim=0
        )
        seed_values = values[..., row["seed_positions"]]
        flat_seed = seed_values.reshape(-1, len(row["seed_positions"]))
        selected_slots = torch.tensor(row["selected_slots"], device=flat_seed.device)
        selected_absolute = flat_seed[:, selected_slots].sum(dim=-1)
        take = min(head_count, selected_absolute.numel())
        top_indices = torch.topk(selected_absolute, k=take, dim=0).indices
        chosen = flat_seed[top_indices]
        chosen_distribution = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mean_distribution = chosen_distribution.mean(dim=0)
        target = torch.zeros_like(mean_distribution)
        target[selected_slots] = 1.0 / len(row["selected_slots"])
        focus_loss = -torch.log(selected_absolute[top_indices].mean().clamp_min(1e-8))
        coverage_loss = -(target * torch.log(mean_distribution.clamp_min(1e-8))).sum()
        losses.append(focus_loss + coverage_weight * coverage_loss)
        absolute_masses.append(selected_absolute[top_indices].mean().detach())
        selected_ratios.append(mean_distribution[selected_slots].sum().detach())
    return torch.stack(losses).mean(), {
        "selected_attention_mass": float(torch.stack(absolute_masses).mean().cpu()),
        "selected_seed_ratio": float(torch.stack(selected_ratios).mean().cpu()),
    }


def evaluate(
    model: Any,
    rows: list[dict[str, Any]],
    pad_id: int,
    device: Any,
    batch_size: int,
    bit_token_ids: tuple[int, int],
) -> dict[str, float]:
    import torch

    model.eval()
    correct_greedy = correct_binary = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = make_batch(rows[start : start + batch_size], pad_id, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            indices = torch.arange(len(batch["rows"]), device=device)
            logits = outputs.logits[indices, batch["query_positions"]]
            losses = torch.nn.functional.cross_entropy(
                logits, batch["targets"], reduction="none"
            )
            greedy = logits.argmax(dim=-1)
            candidates = logits[:, list(bit_token_ids)].argmax(dim=-1)
            binary_tokens = torch.tensor(bit_token_ids, device=device)[candidates]
            correct_greedy += int((greedy == batch["targets"]).sum().item())
            correct_binary += int((binary_tokens == batch["targets"]).sum().item())
            loss_sum += float(losses.sum().item())
            total += len(batch["rows"])
    return {
        "n": total,
        "greedy_accuracy": correct_greedy / total,
        "binary_accuracy": correct_binary / total,
        "bit_ce": loss_sum / total,
    }


def attention_diagnostic(
    model: Any,
    rows: list[dict[str, Any]],
    pad_id: int,
    device: Any,
    batch_size: int,
    layer_count: int,
    head_count: int,
    coverage_weight: float,
) -> dict[str, float]:
    import torch

    model.eval()
    values = []
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
            loss, metrics = attention_auxiliary_loss(
                outputs.attentions, batch, layer_count, head_count, coverage_weight
            )
            values.append((len(batch["rows"]), float(loss.cpu()), metrics))
    total = sum(item[0] for item in values)
    return {
        "n": total,
        "attention_aux_loss": sum(n * loss for n, loss, _ in values) / total,
        "selected_attention_mass": sum(
            n * metrics["selected_attention_mass"] for n, _, metrics in values
        )
        / total,
        "selected_seed_ratio": sum(
            n * metrics["selected_seed_ratio"] for n, _, metrics in values
        )
        / total,
    }


def save_adapter(model: Any, directory: Path, metadata: dict[str, Any]) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)
    (directory / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def train(
    config_path: Path,
    condition: str,
    data_root: Path,
    artifacts_root: Path,
    initial_adapter: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from peft import PeftModel
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(config_path.read_text(encoding="utf-8"))
    condition_config = config["conditions"][condition]
    use_attention = bool(condition_config["attention_regularization"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("This training job requires a CUDA GPU")
    if not (initial_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing initial adapter: {initial_adapter}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for exact seed-token alignment"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if use_attention:
        load_kwargs["attn_implementation"] = "eager"
    base = AutoModelForCausalLM.from_pretrained(config["model"], **load_kwargs)
    base.config.use_cache = False
    try:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(initial_adapter), is_trainable=True)
    device = torch.device("cuda")
    model.to(device)

    raw_train, raw_validation = read_jsonl(data_root / "train.jsonl"), read_jsonl(
        data_root / "validation.jsonl"
    )
    train_rows = [prepare_row(tokenizer, row) for row in raw_train]
    validation_rows = [prepare_row(tokenizer, row) for row in raw_validation]
    token_by_bit: dict[int, int] = {}
    for row in train_rows + validation_rows:
        previous = token_by_bit.setdefault(row["gold_bit"], row["target_id"])
        if previous != row["target_id"]:
            raise ValueError(
                "A bit maps to more than one target token in this prompt context"
            )
    if set(token_by_bit) != {0, 1} or token_by_bit[0] == token_by_bit[1]:
        raise ValueError(f"Invalid bit token mapping: {token_by_bit}")
    bit_token_ids = (token_by_bit[0], token_by_bit[1])

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    output_dir = artifacts_root / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["batch_size"])
    evaluation_batch_size = int(config["evaluation_batch_size"])
    attention_weight = float(config["attention_loss_weight"]) if use_attention else 0.0
    baseline = evaluate(
        model,
        validation_rows,
        tokenizer.pad_token_id,
        device,
        evaluation_batch_size,
        bit_token_ids,
    )
    history: list[dict[str, Any]] = []
    best_accuracy, best_epoch = -math.inf, 0
    rng = random.Random(seed)
    started = time.monotonic()

    for epoch in range(1, int(config["epochs"]) + 1):
        shuffled = train_rows.copy()
        rng.shuffle(shuffled)
        model.train()
        accumulated = {
            "examples": 0,
            "total": 0.0,
            "bit_ce": 0.0,
            "attention_aux": 0.0,
            "selected_attention_mass": 0.0,
            "selected_seed_ratio": 0.0,
        }
        for start in range(0, len(shuffled), batch_size):
            batch = make_batch(
                shuffled[start : start + batch_size], tokenizer.pad_token_id, device
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=use_attention,
                use_cache=False,
                return_dict=True,
            )
            indices = torch.arange(len(batch["rows"]), device=device)
            logits = outputs.logits[indices, batch["query_positions"]]
            bit_ce = torch.nn.functional.cross_entropy(logits, batch["targets"])
            auxiliary = torch.zeros((), device=device)
            attention_metrics = {
                "selected_attention_mass": 0.0,
                "selected_seed_ratio": 0.0,
            }
            if use_attention:
                auxiliary, attention_metrics = attention_auxiliary_loss(
                    outputs.attentions,
                    batch,
                    int(config["attention_layers"]),
                    int(config["attention_heads"]),
                    float(config["attention_coverage_weight"]),
                )
            loss = bit_ce + attention_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            count = len(batch["rows"])
            accumulated["examples"] += count
            accumulated["total"] += count * float(loss.detach().cpu())
            accumulated["bit_ce"] += count * float(bit_ce.detach().cpu())
            accumulated["attention_aux"] += count * float(auxiliary.detach().cpu())
            for key in ("selected_attention_mass", "selected_seed_ratio"):
                accumulated[key] += count * attention_metrics[key]

        train_evaluation = evaluate(
            model,
            train_rows[: min(1024, len(train_rows))],
            tokenizer.pad_token_id,
            device,
            evaluation_batch_size,
            bit_token_ids,
        )
        validation = evaluate(
            model,
            validation_rows,
            tokenizer.pad_token_id,
            device,
            evaluation_batch_size,
            bit_token_ids,
        )
        diagnostic = None
        if use_attention:
            diagnostic = attention_diagnostic(
                model,
                validation_rows[: int(config["attention_evaluation_n"])],
                tokenizer.pad_token_id,
                device,
                min(batch_size, evaluation_batch_size),
                int(config["attention_layers"]),
                int(config["attention_heads"]),
                float(config["attention_coverage_weight"]),
            )
        count = accumulated["examples"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": accumulated["total"] / count,
            "train_bit_ce": accumulated["bit_ce"] / count,
            "train_attention_aux": accumulated["attention_aux"] / count,
            "train_selected_attention_mass": accumulated["selected_attention_mass"]
            / count,
            "train_selected_seed_ratio": accumulated["selected_seed_ratio"] / count,
            "train_evaluation": train_evaluation,
            "validation": validation,
            "validation_attention": diagnostic,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if validation["greedy_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["greedy_accuracy"], epoch
            if condition_config["keep_best_adapter"]:
                save_adapter(
                    model,
                    output_dir / "best_adapter",
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "validation_greedy_accuracy": best_accuracy,
                        "initial_adapter": str(initial_adapter),
                    },
                )
        if epoch >= int(config["minimum_epochs"]) and validation[
            "greedy_accuracy"
        ] >= float(config["early_stop_validation_accuracy"]):
            break

    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "condition": condition,
        "seed": seed,
        "attention_regularization": use_attention,
        "attention_loss_weight": attention_weight,
        "initial_adapter": str(initial_adapter),
        "train_n": len(train_rows),
        "validation_n": len(validation_rows),
        "bit_token_ids": {str(bit): token for bit, token in token_by_bit.items()},
        "baseline_validation": baseline,
        "best_epoch": best_epoch,
        "best_validation_greedy_accuracy": best_accuracy,
        "gate_threshold": float(config["gate_validation_accuracy"]),
        "passes_gate": best_accuracy >= float(config["gate_validation_accuracy"]),
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--condition", choices=("bit_only", "bit_attention"), required=True
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_bit_attention/seed_0"),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_bit_attention"),
    )
    parser.add_argument("--initial-adapter", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    initial = args.initial_adapter or Path(config["initial_adapter"])
    print(
        json.dumps(
            train(
                args.config,
                args.condition,
                args.data_root,
                args.artifacts_root,
                initial,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
