#!/usr/bin/env python3
"""Train supplied and derived one-bit encryption bridges with attention supervision."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def chat_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt_text"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
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
            f"Target must append exactly one token for {row['experiment_id']}"
        )
    groups = [list(map(int, group)) for group in row["spec"]["attention_groups"]]
    seed_positions = []
    if groups:
        offsets = [tuple(pair) for pair in prompt_encoding["offset_mapping"]]
        seed = str(row["spec"]["prg_seed"])
        for label, value in zip("ABCDEFGHIJKLMNOP", seed):
            marker = f"{label}={value}"
            start = prompt.find(marker)
            if start < 0:
                raise ValueError(f"Missing {marker} in {row['experiment_id']}")
            seed_positions.append(token_at_character(offsets, start + len(marker) - 1))
        if len(set(seed_positions)) != 16:
            raise ValueError(
                f"Seed values are not distinct tokens in {row['experiment_id']}"
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


def attention_auxiliary_loss(
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
            take = min(head_count, absolute.numel())
            top = torch.topk(absolute, k=take).indices
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
        return zero, {
            "attention_groups": 0,
            "selected_attention_mass": 0.0,
            "selected_seed_ratio": 0.0,
        }
    return torch.stack(losses).mean(), {
        "attention_groups": len(losses),
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
            candidate = logits[:, list(bit_tokens)].argmax(dim=-1)
            binary = torch.tensor(bit_tokens, device=device)[candidate]
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
            n * m["selected_attention_mass"] for n, _, m in values
        )
        / total,
        "selected_seed_ratio": sum(n * m["selected_seed_ratio"] for n, _, m in values)
        / total,
    }


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


def train(
    config_path: Path, data_root: Path, output_root: Path, initial_adapter: Path
) -> dict[str, Any]:
    import numpy as np
    import torch
    from peft import PeftModel
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if not (initial_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Missing successful one-bit adapter: {initial_adapter}"
        )
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw_names = ["local_replay", "local_validation"] + [
        f"{task}_{split}"
        for task in (
            "supplied_start",
            "derived_start",
            "supplied_update",
            "derived_update",
        )
        for split in ("train", "validation")
    ]
    data = {
        name: [
            prepare_row(tokenizer, row)
            for row in read_jsonl(data_root / f"{name}.jsonl")
        ]
        for name in raw_names
    }
    token_by_bit = {}
    for rows in data.values():
        for row in rows:
            prior = token_by_bit.setdefault(row["gold_bit"], row["target_id"])
            if prior != row["target_id"]:
                raise ValueError("Context-dependent target bit token")
    bit_tokens = (token_by_bit[0], token_by_bit[1])

    base = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
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
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    started = time.monotonic()
    baseline_local = evaluate(
        model,
        data["local_validation"],
        tokenizer.pad_token_id,
        device,
        int(config["evaluation_batch_size"]),
        bit_tokens,
    )
    history, completed = [], []
    replay_cfg = config["replay"]

    for stage in config["stages"]:
        name, max_epochs = stage["name"], int(stage["epochs"])
        current, validation = data[f"{name}_train"], data[f"{name}_validation"]
        best_score, best_state, best_epoch = -math.inf, None, 0
        stage_history = []
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(config["learning_rate"]),
        )
        for epoch in range(1, max_epochs + 1):
            groups = [
                current,
                sample(data["local_replay"], int(replay_cfg["local_per_epoch"]), rng),
            ]
            if name != "supplied_start":
                groups.append(
                    sample(
                        data["supplied_start_train"],
                        int(replay_cfg["supplied_start_per_epoch"]),
                        rng,
                    )
                )
            if name in {"supplied_update", "derived_update"}:
                groups.append(
                    sample(
                        data["derived_start_train"],
                        int(replay_cfg["derived_start_per_epoch"]),
                        rng,
                    )
                )
            if name == "derived_update":
                groups.append(
                    sample(
                        data["supplied_update_train"],
                        int(replay_cfg["supplied_update_per_epoch"]),
                        rng,
                    )
                )
            model.train()
            totals = defaultdict(float)
            seen = 0
            for batch_rows in epoch_batches(groups, int(config["batch_size"]), rng):
                batch = make_batch(batch_rows, tokenizer.pad_token_id, device)
                needs_attention = any(row["attention_groups"] for row in batch_rows)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_attentions=needs_attention,
                    use_cache=False,
                    return_dict=True,
                )
                indices = torch.arange(len(batch_rows), device=device)
                logits = outputs.logits[indices, batch["query_positions"]]
                bit_ce = torch.nn.functional.cross_entropy(logits, batch["targets"])
                attention_loss = torch.zeros((), device=device)
                attention_metrics = {
                    "attention_groups": 0,
                    "selected_attention_mass": 0.0,
                    "selected_seed_ratio": 0.0,
                }
                if needs_attention:
                    attention_loss, attention_metrics = attention_auxiliary_loss(
                        outputs.attentions,
                        batch,
                        int(config["attention_layers"]),
                        int(config["attention_heads"]),
                        float(config["attention_coverage_weight"]),
                    )
                loss = bit_ce + float(config["attention_loss_weight"]) * attention_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip"])
                )
                optimizer.step()
                count = len(batch_rows)
                seen += count
                totals["loss"] += count * float(loss.detach().cpu())
                totals["bit_ce"] += count * float(bit_ce.detach().cpu())
                totals["attention_aux"] += count * float(attention_loss.detach().cpu())
                totals["selected_attention_mass"] += (
                    count * attention_metrics["selected_attention_mass"]
                )
                totals["selected_seed_ratio"] += (
                    count * attention_metrics["selected_seed_ratio"]
                )

            current_eval = evaluate(
                model,
                validation,
                tokenizer.pad_token_id,
                device,
                int(config["evaluation_batch_size"]),
                bit_tokens,
            )
            local_eval = evaluate(
                model,
                data["local_validation"],
                tokenizer.pad_token_id,
                device,
                int(config["evaluation_batch_size"]),
                bit_tokens,
            )
            supplied_start_eval = None
            if name != "supplied_start":
                supplied_start_eval = evaluate(
                    model,
                    data["supplied_start_validation"],
                    tokenizer.pad_token_id,
                    device,
                    int(config["evaluation_batch_size"]),
                    bit_tokens,
                )
            start_eval = None
            if name in {"supplied_update", "derived_update"}:
                start_eval = evaluate(
                    model,
                    data["derived_start_validation"],
                    tokenizer.pad_token_id,
                    device,
                    int(config["evaluation_batch_size"]),
                    bit_tokens,
                )
            diagnostic = attention_diagnostic(
                model,
                validation[: int(config["attention_evaluation_n"])],
                tokenizer.pad_token_id,
                device,
                min(int(config["batch_size"]), int(config["evaluation_batch_size"])),
                config,
            )
            score = (
                current_eval["greedy_accuracy"] + 0.25 * local_eval["greedy_accuracy"]
            )
            if supplied_start_eval:
                score += 0.10 * supplied_start_eval["greedy_accuracy"]
            if start_eval:
                score += 0.25 * start_eval["greedy_accuracy"]
            record = {
                "stage": name,
                "epoch": epoch,
                "train": {key: value / seen for key, value in totals.items()},
                "current_validation": current_eval,
                "local_retention": local_eval,
                "supplied_start_retention": supplied_start_eval,
                "derived_start_retention": start_eval,
                "validation_attention": diagnostic,
                "selection_score": score,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            stage_history.append(record)
            print(json.dumps(record), flush=True)
            if score > best_score:
                best_score, best_epoch, best_state = (
                    score,
                    epoch,
                    clone_trainable(model),
                )
            (output_root / "progress.json").write_text(
                json.dumps(
                    {
                        "experiment_name": config["experiment_name"],
                        "history": history,
                        "current_stage": name,
                        "current_epoch": epoch,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            retention_ok = (
                local_eval["greedy_accuracy"]
                >= config["success_gate"]["minimum_local_retention"]
            )
            if supplied_start_eval:
                retention_ok &= (
                    supplied_start_eval["greedy_accuracy"]
                    >= config["success_gate"]["minimum_supplied_retention"]
                )
            if start_eval:
                retention_ok &= (
                    start_eval["greedy_accuracy"]
                    >= config["success_gate"]["minimum_start_retention"]
                )
            if (
                epoch >= int(config["minimum_epochs"])
                and current_eval["greedy_accuracy"]
                >= float(config["early_stop_accuracy"])
                and retention_ok
            ):
                break
        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {name}")
        restore_trainable(model, best_state)
        completed.append(
            {
                "stage": name,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": stage_history,
            }
        )

    final = {}
    for task in (
        "local",
        "supplied_start",
        "derived_start",
        "supplied_update",
        "derived_update",
    ):
        key = "local_validation" if task == "local" else f"{task}_validation"
        final[task] = {
            "accuracy": evaluate(
                model,
                data[key],
                tokenizer.pad_token_id,
                device,
                int(config["evaluation_batch_size"]),
                bit_tokens,
            ),
            "attention": attention_diagnostic(
                model,
                data[key][: int(config["attention_evaluation_n"])],
                tokenizer.pad_token_id,
                device,
                min(int(config["batch_size"]), int(config["evaluation_batch_size"])),
                config,
            ),
        }
    adapter = output_root / "final_adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    model.save_pretrained(adapter, safe_serialization=True)
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "initial_adapter": str(initial_adapter),
        "baseline_local": baseline_local,
        "completed_stages": completed,
        "final_evaluations": final,
        "elapsed_seconds": time.monotonic() - started,
        "final_adapter": str(adapter),
    }
    (output_root / "training_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_encrypted_bridge/seed_0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_encrypted_bridge/training"),
    )
    parser.add_argument("--initial-adapter", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    initial = args.initial_adapter or Path(config["initial_adapter"])
    print(
        json.dumps(
            train(args.config, args.data_root, args.output_root, initial), indent=2
        )
    )


if __name__ == "__main__":
    main()
