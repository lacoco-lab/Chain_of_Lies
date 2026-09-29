#!/usr/bin/env python3
"""Train all encrypted-CoT lengths in one attention-guided GPU curriculum."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

COT_RE = re.compile(r"<COT>\s*(.*?)\s*</COT>", re.DOTALL)
LINE_RE = re.compile(r"^\(use Seed: ([A-P](?: [A-P]){4})\) ([01])$")
ANSWER_RE = re.compile(r"<ANSWER>\s*([01])\s*</ANSWER>")
SEED_VALUE_RE = re.compile(r"\b[A-P]\s*[:=]\s*[01]\b")


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
    prompt, suffix = chat_prompt(tokenizer, row), str(row["supervised_suffix"])
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    encoding = tokenizer(
        prompt + suffix, add_special_tokens=False, return_offsets_mapping=True
    )
    full_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Suffix changed prompt tokenization for {row['experiment_id']}"
        )
    spec = row["spec"]
    bit_indices = [
        token_at_character(offsets, len(prompt) + int(offset))
        for offset in spec["bit_offsets"]
    ]
    answer_index = None
    if spec["kind"] == "trace":
        answer_index = token_at_character(
            offsets, len(prompt) + int(spec["answer_offset"])
        )
    supervised = set(range(len(prompt_ids), len(full_ids)))
    variable = set(bit_indices) | (
        {answer_index} if answer_index is not None else set()
    )
    if not variable <= supervised:
        raise ValueError(
            f"Variable targets fall outside suffix for {row['experiment_id']}"
        )
    format_indices = sorted(supervised - variable)

    seed = str(spec["prg_seed"])
    seed_positions = []
    for label, value in zip("ABCDEFGHIJKLMNOP", seed):
        marker = f"{label}={value}"
        start = prompt.find(marker)
        if start < 0:
            raise ValueError(f"Missing {marker} in {row['experiment_id']}")
        seed_positions.append(token_at_character(offsets, start + len(marker) - 1))
    if len(set(seed_positions)) != 16:
        raise ValueError(f"Seed positions are not distinct in {row['experiment_id']}")
    edges = [spec["edge"]] if spec["kind"] == "local" else spec["edges"]
    return {
        "id": row["experiment_id"],
        "kind": spec["kind"],
        "length": int(spec.get("length", 1)),
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "bit_indices": bit_indices,
        "answer_index": answer_index,
        "format_indices": format_indices,
        "seed_positions": seed_positions,
        "attention_edges": edges,
        "raw": row,
    }


def make_batch(rows: list[dict[str, Any]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    maximum = max(len(row["full_ids"]) - 1 for row in rows)
    input_ids = torch.full(
        (len(rows), maximum), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(rows), maximum), dtype=torch.long, device=device)
    minimum_query = maximum
    for index, row in enumerate(rows):
        values = row["full_ids"][:-1]
        input_ids[index, : len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        attention_mask[index, : len(values)] = 1
        targets = [*row["bit_indices"], *row["format_indices"]]
        if row["answer_index"] is not None:
            targets.append(row["answer_index"])
        minimum_query = min(minimum_query, min(targets) - 1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "rows": rows,
        "logits_to_keep": maximum - minimum_query,
    }


def _indexed_ce(
    logits: Any,
    logit_offset: int,
    row_index: int,
    target_indices: list[int],
    full_ids: list[int],
) -> Any | None:
    import torch

    if not target_indices:
        return None
    queries = torch.tensor(
        [index - 1 - logit_offset for index in target_indices],
        dtype=torch.long,
        device=logits.device,
    )
    targets = torch.tensor(
        [full_ids[index] for index in target_indices],
        dtype=torch.long,
        device=logits.device,
    )
    return torch.nn.functional.cross_entropy(logits[row_index, queries], targets)


def model_forward(
    model: Any, batch: dict[str, Any], *, output_attentions: bool
) -> tuple[Any, int]:
    """Use sparse Qwen logits when supported, with a 4.44-compatible fallback."""
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "output_attentions": output_attentions,
        "use_cache": False,
        "return_dict": True,
    }
    try:
        outputs = model(**kwargs, logits_to_keep=int(batch["logits_to_keep"]))
    except TypeError as error:
        if "logits_to_keep" not in str(error):
            raise
        # Qwen2ForCausalLM in Transformers 4.44.2 predates logits_to_keep.
        outputs = model(**kwargs)
    logit_offset = batch["input_ids"].shape[1] - outputs.logits.shape[1]
    return outputs, logit_offset


def attention_auxiliary_loss(
    attentions: tuple[Any, ...],
    batch: dict[str, Any],
    layer_count: int,
    head_count: int,
    queries_per_example: int,
    coverage_weight: float,
    rng: random.Random,
) -> tuple[Any, dict[str, float]]:
    import torch

    if not attentions or any(layer is None for layer in attentions[-layer_count:]):
        raise RuntimeError("The model did not return attention weights")
    losses, masses, ratios, query_count = [], [], [], 0
    layers = attentions[-layer_count:]
    for row_index, row in enumerate(batch["rows"]):
        pairs = list(zip(row["bit_indices"], row["attention_edges"]))
        if len(pairs) > queries_per_example:
            pairs = rng.sample(pairs, queries_per_example)
        for target_index, selected_slots_raw in pairs:
            query = target_index - 1
            values = torch.stack(
                [layer[row_index, :, query, :] for layer in layers], dim=0
            )
            seed_values = values[..., row["seed_positions"]]
            flat_seed = seed_values.reshape(-1, 16)
            selected_slots = torch.tensor(
                selected_slots_raw, dtype=torch.long, device=flat_seed.device
            )
            selected_absolute = flat_seed[:, selected_slots].sum(dim=-1)
            take = min(head_count, selected_absolute.numel())
            top_indices = torch.topk(selected_absolute, k=take).indices
            chosen = flat_seed[top_indices]
            distribution = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mean_distribution = distribution.mean(dim=0)
            target = torch.zeros_like(mean_distribution)
            target[selected_slots] = 1.0 / len(selected_slots_raw)
            focus = -torch.log(selected_absolute[top_indices].mean().clamp_min(1e-8))
            coverage = -(target * torch.log(mean_distribution.clamp_min(1e-8))).sum()
            losses.append(focus + coverage_weight * coverage)
            masses.append(selected_absolute[top_indices].mean().detach())
            ratios.append(mean_distribution[selected_slots].sum().detach())
            query_count += 1
    return torch.stack(losses).mean(), {
        "attention_queries": query_count,
        "selected_attention_mass": float(torch.stack(masses).mean().cpu()),
        "selected_seed_ratio": float(torch.stack(ratios).mean().cpu()),
    }


def forward_losses(
    model: Any, batch: dict[str, Any], config: dict[str, Any], rng: random.Random
) -> tuple[Any, dict[str, float]]:
    import torch

    outputs, offset = model_forward(model, batch, output_attentions=True)
    mechanism, answers, formats = [], [], []
    for row_index, row in enumerate(batch["rows"]):
        value = _indexed_ce(
            outputs.logits, offset, row_index, row["bit_indices"], row["full_ids"]
        )
        if value is not None:
            mechanism.append(value)
        if row["answer_index"] is not None:
            answers.append(
                _indexed_ce(
                    outputs.logits,
                    offset,
                    row_index,
                    [row["answer_index"]],
                    row["full_ids"],
                )
            )
        value = _indexed_ce(
            outputs.logits, offset, row_index, row["format_indices"], row["full_ids"]
        )
        if value is not None:
            formats.append(value)
    mechanism_loss = torch.stack(mechanism).mean()
    answer_loss = (
        torch.stack(answers).mean()
        if answers
        else torch.zeros((), device=mechanism_loss.device)
    )
    format_loss = (
        torch.stack(formats).mean()
        if formats
        else torch.zeros((), device=mechanism_loss.device)
    )
    attention_loss, attention_metrics = attention_auxiliary_loss(
        outputs.attentions,
        batch,
        int(config["attention_layers"]),
        int(config["attention_heads"]),
        int(config["attention_queries_per_example"]),
        float(config["attention_coverage_weight"]),
        rng,
    )
    total = (
        mechanism_loss
        + float(config["answer_loss_weight"]) * answer_loss
        + float(config["format_loss_weight"]) * format_loss
        + float(config["attention_loss_weight"]) * attention_loss
    )
    return total, {
        "total_loss": float(total.detach().cpu()),
        "mechanism_bit_ce": float(mechanism_loss.detach().cpu()),
        "answer_ce": float(answer_loss.detach().cpu()),
        "format_ce": float(format_loss.detach().cpu()),
        "attention_aux": float(attention_loss.detach().cpu()),
        **attention_metrics,
    }


def teacher_evaluate(
    model: Any, rows: list[dict[str, Any]], pad_id: int, device: Any, batch_size: int
) -> dict[str, float]:
    import torch

    model.eval()
    counts = defaultdict(int)
    losses = defaultdict(float)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = make_batch(rows[start : start + batch_size], pad_id, device)
            outputs, offset = model_forward(model, batch, output_attentions=False)
            for row_index, row in enumerate(batch["rows"]):
                group_correct = []
                for name, target_indices in (
                    ("bit", row["bit_indices"]),
                    (
                        "answer",
                        (
                            [row["answer_index"]]
                            if row["answer_index"] is not None
                            else []
                        ),
                    ),
                    ("format", row["format_indices"]),
                ):
                    if not target_indices:
                        continue
                    queries = torch.tensor(
                        [index - 1 - offset for index in target_indices], device=device
                    )
                    targets = torch.tensor(
                        [row["full_ids"][index] for index in target_indices],
                        device=device,
                    )
                    logits = outputs.logits[row_index, queries]
                    prediction = logits.argmax(dim=-1)
                    correct = prediction == targets
                    counts[f"{name}_correct"] += int(correct.sum().item())
                    counts[f"{name}_total"] += len(target_indices)
                    losses[f"{name}_ce_sum"] += float(
                        torch.nn.functional.cross_entropy(
                            logits, targets, reduction="sum"
                        ).cpu()
                    )
                    if name == "bit":
                        group_correct = correct.tolist()
                counts["examples"] += 1
                counts["bit_exact"] += bool(group_correct) and all(group_correct)
    result = {
        "n": counts["examples"],
        "bit_exact": counts["bit_exact"] / counts["examples"],
    }
    for name in ("bit", "answer", "format"):
        total = counts[f"{name}_total"]
        if total:
            result[f"{name}_accuracy"] = counts[f"{name}_correct"] / total
            result[f"{name}_ce"] = losses[f"{name}_ce_sum"] / total
    return result


def _parse_trace(raw: str) -> tuple[list[str], list[int]] | None:
    matches = COT_RE.findall(raw)
    if len(matches) != 1:
        return None
    selectors, bits = [], []
    for line in matches[0].strip().splitlines():
        match = LINE_RE.fullmatch(line.strip())
        if match is None:
            return None
        selectors.append(f"(use Seed: {match.group(1)})")
        bits.append(int(match.group(2)))
    return selectors, bits


def autoregressive_evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    batch_size: int,
    save_path: Path | None = None,
) -> dict[str, float]:
    import torch

    model.eval()
    totals = defaultdict(int)
    twin_predictions: dict[str, list[list[int] | None]] = defaultdict(list)
    response_rows = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            current = rows[start : start + batch_size]
            prompts = [row["prompt_ids"] for row in current]
            maximum = max(map(len, prompts))
            # Left padding is required for batched decoder-only generation.
            input_ids = torch.full(
                (len(current), maximum),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, prompt_ids in enumerate(prompts):
                input_ids[index, -len(prompt_ids) :] = torch.tensor(
                    prompt_ids, device=device
                )
                attention_mask[index, -len(prompt_ids) :] = 1
            max_target = max(
                len(row["full_ids"]) - len(row["prompt_ids"]) for row in current
            )
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_target + 8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            for index, row in enumerate(current):
                raw = tokenizer.decode(
                    generated[index, maximum:], skip_special_tokens=True
                )
                spec = row["raw"]["spec"]
                parsed = _parse_trace(raw)
                predicted_selectors, predicted_bits = parsed if parsed else ([], [])
                gold_bits, gold_selectors = spec["gold_sequence"], spec["selectors"]
                answers = ANSWER_RE.findall(raw)
                totals["n"] += 1
                totals["parsed"] += parsed is not None
                totals["trace_exact"] += predicted_bits == gold_bits
                totals["selector_exact"] += predicted_selectors == gold_selectors
                totals["full_exact"] += (
                    predicted_bits == gold_bits
                    and predicted_selectors == gold_selectors
                    and bool(answers)
                    and int(answers[-1]) == int(spec["final_parity"])
                )
                totals["bit_correct"] += sum(
                    left == right
                    for left, right in zip(predicted_bits[: len(gold_bits)], gold_bits)
                )
                totals["bit_total"] += len(gold_bits)
                totals["selector_correct"] += sum(
                    left == right
                    for left, right in zip(
                        predicted_selectors[: len(gold_selectors)], gold_selectors
                    )
                )
                totals["selector_total"] += len(gold_selectors)
                totals["answer_correct"] += bool(answers) and int(answers[-1]) == int(
                    spec["final_parity"]
                )
                totals["leak"] += bool(
                    SEED_VALUE_RE.search(raw)
                    or "<PRIVATE_SEED>" in raw
                    or "<PRIVATE_INPUT>" in raw
                )
                predicted_complete = (
                    predicted_bits if len(predicted_bits) == len(gold_bits) else None
                )
                twin_predictions[spec["twin_id"]].append(predicted_complete)
                response_rows.append(
                    {
                        "experiment_id": row["id"],
                        "raw_text": raw,
                        "gold_sequence": gold_bits,
                        "predicted_sequence": predicted_complete,
                        "gold_answer": spec["final_parity"],
                    }
                )
    complete_twins = matching_twins = 0
    for values in twin_predictions.values():
        if len(values) == 2:
            complete_twins += 1
            matching_twins += values[0] is not None and values[0] == values[1]
    n = totals["n"]
    result = {
        "n": n,
        "parse_rate": totals["parsed"] / n,
        "trace_exact": totals["trace_exact"] / n,
        "trace_bit_accuracy": totals["bit_correct"] / totals["bit_total"],
        "selector_exact": totals["selector_exact"] / n,
        "selector_accuracy": totals["selector_correct"] / totals["selector_total"],
        "answer_accuracy": totals["answer_correct"] / n,
        "full_exact": totals["full_exact"] / n,
        "seed_leak_rate": totals["leak"] / n,
        "complete_twin_pairs": complete_twins,
        "twin_trace_match": matching_twins / complete_twins if complete_twins else 0.0,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as handle:
            for response in response_rows:
                handle.write(json.dumps(response) + "\n")
    return result


def _clone_trainable(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable(model: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[name].to(parameter.device))


def _epoch_groups(
    current: list[dict[str, Any]],
    previous: list[list[dict[str, Any]]],
    local: list[dict[str, Any]],
    config: dict[str, Any],
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    groups = [current.copy()]
    local_n = int(config["local_replay_per_epoch"])
    groups.append(rng.sample(local, min(local_n, len(local))))
    previous_n = int(config["previous_trace_replay_per_epoch"])
    if previous:
        per = max(1, previous_n // len(previous))
        replay = []
        for rows in previous:
            replay.extend(rng.sample(rows, min(per, len(rows))))
        groups.append(replay[:previous_n])
    return groups


def _make_epoch_batches(
    groups: list[list[dict[str, Any]]], batch_size: int, rng: random.Random
) -> list[list[dict[str, Any]]]:
    batches = []
    for group in groups:
        shuffled = group.copy()
        rng.shuffle(shuffled)
        batches.extend(
            shuffled[start : start + batch_size]
            for start in range(0, len(shuffled), batch_size)
        )
    rng.shuffle(batches)
    return batches


def _passes_early_stop(
    auto: dict[str, float], local: dict[str, float], gate: dict[str, float]
) -> bool:
    return (
        auto["trace_bit_accuracy"] >= gate["trace_bit_accuracy"]
        and auto["selector_accuracy"] >= gate["selector_accuracy"]
        and auto["answer_accuracy"] >= gate["answer_accuracy"]
        and local["bit_accuracy"] >= gate["local_bit_accuracy"]
    )


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
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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

    local_replay = [
        prepare_row(tokenizer, row)
        for row in read_jsonl(data_root / "local_replay.jsonl")
    ]
    local_validation = [
        prepare_row(tokenizer, row)
        for row in read_jsonl(data_root / "local_validation.jsonl")
    ]
    prepared_train, prepared_validation = {}, {}
    for stage in config["stages"]:
        length = int(stage["length"])
        prepared_train[length] = [
            prepare_row(tokenizer, row)
            for row in read_jsonl(data_root / f"trace_train_{length}.jsonl")
        ]
        prepared_validation[length] = [
            prepare_row(tokenizer, row)
            for row in read_jsonl(data_root / f"trace_validation_{length}.jsonl")
        ]

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    started = time.monotonic()
    baseline_local = teacher_evaluate(
        model,
        local_validation,
        tokenizer.pad_token_id,
        device,
        int(config["teacher_evaluation_batch_size"]),
    )
    history, completed_lengths = [], []
    previous_train: list[list[dict[str, Any]]] = []

    for stage_index, stage in enumerate(config["stages"], start=1):
        length, maximum_epochs, batch_size = (
            int(stage["length"]),
            int(stage["epochs"]),
            int(stage["batch_size"]),
        )
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]),
        )
        best_score, best_state, best_epoch = -math.inf, None, 0
        stage_history = []
        for epoch in range(1, maximum_epochs + 1):
            model.train()
            batches = _make_epoch_batches(
                _epoch_groups(
                    prepared_train[length], previous_train, local_replay, config, rng
                ),
                batch_size,
                rng,
            )
            totals = defaultdict(float)
            seen = 0
            for batch_rows in batches:
                batch = make_batch(batch_rows, tokenizer.pad_token_id, device)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = forward_losses(model, batch, config, rng)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip"])
                )
                optimizer.step()
                count = len(batch_rows)
                seen += count
                for key, value in metrics.items():
                    if key != "attention_queries":
                        totals[key] += count * value
                totals["attention_queries"] += metrics["attention_queries"]

            evaluation_rows = prepared_validation[length][
                : min(
                    int(config["generation_evaluation_n"]),
                    len(prepared_validation[length]),
                )
            ]
            teacher = teacher_evaluate(
                model,
                evaluation_rows,
                tokenizer.pad_token_id,
                device,
                min(batch_size, int(config["teacher_evaluation_batch_size"])),
            )
            autoregressive = autoregressive_evaluate(
                model,
                tokenizer,
                evaluation_rows,
                device,
                min(batch_size, int(config["generation_batch_size"])),
            )
            local = teacher_evaluate(
                model,
                local_validation[: int(config["local_evaluation_n"])],
                tokenizer.pad_token_id,
                device,
                min(8, int(config["teacher_evaluation_batch_size"])),
            )
            score = (
                autoregressive["trace_bit_accuracy"]
                + 0.25 * autoregressive["selector_accuracy"]
                + 0.25 * autoregressive["answer_accuracy"]
                + 0.25 * local["bit_accuracy"]
            )
            record = {
                "stage_index": stage_index,
                "length": length,
                "epoch": epoch,
                "train_examples": seen,
                "train": {
                    key: value / seen
                    for key, value in totals.items()
                    if key != "attention_queries"
                },
                "train_attention_queries": int(totals["attention_queries"]),
                "teacher_validation": teacher,
                "autoregressive_validation": autoregressive,
                "local_retention": local,
                "selection_score": score,
                "elapsed_seconds": time.monotonic() - started,
            }
            stage_history.append(record)
            history.append(record)
            print(json.dumps(record), flush=True)
            if score > best_score:
                best_score, best_epoch, best_state = (
                    score,
                    epoch,
                    _clone_trainable(model),
                )
            partial = {
                "experiment_name": config["experiment_name"],
                "history": history,
                "current_length": length,
                "current_epoch": epoch,
            }
            (output_root / "progress.json").write_text(
                json.dumps(partial, indent=2) + "\n", encoding="utf-8"
            )
            if epoch >= int(config["minimum_epochs"]) and _passes_early_stop(
                autoregressive, local, config["early_stop"]
            ):
                break
        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for length {length}")
        _restore_trainable(model, best_state)
        completed_lengths.append(
            {
                "length": length,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": stage_history,
            }
        )
        previous_train.append(prepared_train[length])

    final_local = teacher_evaluate(
        model,
        local_validation,
        tokenizer.pad_token_id,
        device,
        int(config["teacher_evaluation_batch_size"]),
    )
    final_evaluations = {}
    for length in prepared_validation:
        rows = prepared_validation[length]
        final_evaluations[str(length)] = {
            "teacher": teacher_evaluate(
                model,
                rows,
                tokenizer.pad_token_id,
                device,
                min(
                    next(
                        int(stage["batch_size"])
                        for stage in config["stages"]
                        if int(stage["length"]) == length
                    ),
                    int(config["teacher_evaluation_batch_size"]),
                ),
            ),
            "autoregressive": autoregressive_evaluate(
                model,
                tokenizer,
                rows,
                device,
                min(
                    next(
                        int(stage["batch_size"])
                        for stage in config["stages"]
                        if int(stage["length"]) == length
                    ),
                    int(config["generation_batch_size"]),
                ),
                output_root / "final_responses" / f"length_{length}.jsonl",
            ),
        }
    adapter_dir = output_root / "final_adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "initial_adapter": str(initial_adapter),
        "baseline_local": baseline_local,
        "completed_stages": completed_lengths,
        "final_local_retention": final_local,
        "final_evaluations": final_evaluations,
        "elapsed_seconds": time.monotonic() - started,
        "final_adapter": str(adapter_dir),
    }
    (output_root / "training_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": config["experiment_name"],
                "initial_adapter": str(initial_adapter),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
        default=Path("generated_data/parity_goldreich_full_attention/seed_0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_full_attention/training"),
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
