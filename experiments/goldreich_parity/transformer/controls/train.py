#!/usr/bin/env python3
"""Train matched standard-Transformer PARITY controls."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from model import CONDITIONS, StandardParityControl


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bits_tensor(values: list[str]) -> torch.Tensor:
    width = len(values[0])
    packed = bytearray("".join(values), "ascii")
    return (
        torch.frombuffer(packed, dtype=torch.uint8).reshape(len(values), width).sub(48)
    )


def load_data(path: Path) -> dict[str, Any]:
    """Read only IDs, private inputs, and answers; ignore all encryption fields."""
    ids, strings, answers = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            strings.append(row["input_bits"])
            answers.append(int(row["answer"]))
    inputs = bits_tensor(strings)
    return {
        "ids": ids,
        "input_strings": strings,
        "inputs": inputs,
        "normal_states": inputs.long().cumsum(1).remainder(2),
        "answers": torch.tensor(answers, dtype=torch.long),
    }


def select(data: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    cpu = indices.cpu().tolist()
    return {
        key: (
            [value[index] for index in cpu]
            if isinstance(value, list)
            else value[indices.cpu()]
        )
        for key, value in data.items()
    }


def sanitize_and_audit(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Keep train/test fixed and remove any validation/test input collisions."""

    def digest_set(values: list[str]) -> set[bytes]:
        return {
            hashlib.blake2b(value.encode(), digest_size=16).digest() for value in values
        }

    sets = {name: digest_set(part["input_strings"]) for name, part in parts.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"]:
        raise RuntimeError("A training input appears in a held-out control split")
    collisions = sets["validation"] & sets["test"]
    before = len(parts["validation"]["ids"])
    if collisions:
        keep = [
            index
            for index, value in enumerate(parts["validation"]["input_strings"])
            if hashlib.blake2b(value.encode(), digest_size=16).digest()
            not in collisions
        ]
        per_label = min(
            sum(int(parts["validation"]["answers"][index]) == label for index in keep)
            for label in (0, 1)
        )
        used = {0: 0, 1: 0}
        balanced = []
        for index in keep:
            label = int(parts["validation"]["answers"][index])
            if used[label] < per_label:
                balanced.append(index)
                used[label] += 1
        parts["validation"] = select(parts["validation"], torch.tensor(balanced))
    balance = {
        name: {
            "zero": int((part["answers"] == 0).sum()),
            "one": int((part["answers"] == 1).sum()),
        }
        for name, part in parts.items()
    }
    if any(item["zero"] != item["one"] for item in balance.values()):
        raise RuntimeError(f"Unbalanced control split: {balance}")
    return {
        "train_test_overlap": 0,
        "train_validation_overlap": 0,
        "validation_test_collisions_removed": len(collisions),
        "validation_rows_before": before,
        "validation_rows_after": len(parts["validation"]["ids"]),
        "label_balance": balance,
    }


def main_parameters(model: StandardParityControl) -> list[torch.nn.Parameter]:
    modules = (
        model.token_embedding,
        model.position_embedding,
        model.q_proj,
        model.k_proj,
        model.v_proj,
        model.out_proj,
        model.ff_in,
        model.ff_out,
        model.output_head,
    )
    return [parameter for module in modules for parameter in module.parameters()]


def route_parameters(model: StandardParityControl) -> list[torch.nn.Parameter]:
    modules = (
        model.token_embedding,
        model.position_embedding,
        model.q_proj,
        model.k_proj,
    )
    return [parameter for module in modules for parameter in module.parameters()]


def representation_parameters(model: StandardParityControl) -> list[torch.nn.Parameter]:
    return [*model.v_proj.parameters(), *model.out_proj.parameters()]


def task_parameters(model: StandardParityControl) -> list[torch.nn.Parameter]:
    return [
        *model.ff_in.parameters(),
        *model.ff_out.parameters(),
        *model.output_head.parameters(),
    ]


def sampled_batch(
    data: dict[str, Any],
    model: StandardParityControl,
    condition: str,
    iteration: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    rows = torch.randint(len(data["ids"]), (batch_size,), generator=generator)
    batch = select(data, rows)
    inputs = batch["inputs"].to(device).long()
    answers = batch["answers"].to(device)
    prefixes = (
        (
            batch["normal_states"]
            if condition == "normal_cot"
            else torch.zeros_like(batch["normal_states"])
        )
        .to(device)
        .long()
    )
    if condition == "no_cot":
        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        targets = answers
    elif condition == "filler_cot":
        # Give answer and filler tokens equal weight even for very long traces.
        answer_rows = torch.arange(batch_size, device=device).remainder(2) == 0
        steps = torch.randint(model.length, (batch_size,), generator=generator).to(
            device
        )
        steps[answer_rows] = model.length
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        targets[answer_rows] = answers[answer_rows]
    else:
        start = ((iteration - 1) * batch_size) % (model.length + 1)
        steps = (torch.arange(batch_size, device=device) + start).remainder(
            model.length + 1
        )
        targets = answers.clone()
        ordinary = steps < model.length
        if ordinary.any():
            local_rows = torch.arange(batch_size, device=device)[ordinary]
            targets[ordinary] = batch["normal_states"][
                local_rows.cpu(), steps[ordinary].cpu()
            ].to(device)
    token_ids, valid = model.control_sequence(inputs, prefixes, steps)
    return token_ids, valid, steps, targets, inputs, prefixes


def route_loss(
    model: StandardParityControl,
    outputs: dict[str, torch.Tensor],
    steps: torch.Tensor,
    condition: str,
    direct_lengths: set[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = model.control_targets(steps, condition, direct_lengths)
    if targets is None:
        raise ValueError("This condition has no supervised public route")
    logits = outputs["attention_logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    mass = outputs["attention"].gather(-1, targets[..., None]).squeeze(-1)
    return loss, {
        "routing_accuracy": float((logits.argmax(-1) == targets).float().mean()),
        "minimum_target_attention": float(mass.min().detach()),
        "mean_target_attention": float(mass.mean().detach()),
    }


def train_routes(
    model: StandardParityControl,
    data: dict[str, Any],
    condition: str,
    direct_lengths: set[int],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        route_parameters(model),
        lr=float(config["route_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(10_000 + model.length)
    maximum = int(config["route_steps_by_length"][str(model.length)])
    history = []
    for iteration in range(1, maximum + 1):
        token_ids, valid, steps, *_ = sampled_batch(
            data, model, condition, iteration, batch_size, generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(token_ids, valid, steps)
        loss, metrics = route_loss(model, outputs, steps, condition, direct_lengths)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == maximum:
            record = {"step": iteration, "loss": float(loss.detach()), **metrics}
            history.append(record)
            print(json.dumps({"event": "control_route", **record}), flush=True)
            if (
                metrics["routing_accuracy"] == 1.0
                and metrics["minimum_target_attention"] >= 0.9995
            ):
                break
    return {
        "enabled": True,
        "steps": iteration,
        "final_batch": metrics,
        "history": history,
    }


def canonical_target(
    model: StandardParityControl, source_bits: torch.Tensor, steps: torch.Tensor
) -> torch.Tensor:
    query_positions = model.cot_start + steps
    target, *_ = model._project_query(query_positions)
    target = target.detach().clone()
    target[:, : model.heads] = source_bits.float().mul(2).sub(1).mul(4.0)
    return target


def train_representation(
    model: StandardParityControl,
    data: dict[str, Any],
    condition: str,
    direct_lengths: set[int],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        representation_parameters(model),
        lr=float(config["representation_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(20_000 + model.length)
    maximum = int(config["representation_steps_by_length"][str(model.length)])
    best = (0.0, float("inf"))
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for iteration in range(1, maximum + 1):
        token_ids, valid, steps, *_ = sampled_batch(
            data, model, condition, iteration, batch_size, generator, device
        )
        targets = model.control_targets(steps, condition, direct_lengths)
        assert targets is not None
        optimizer.zero_grad(set_to_none=True)
        outputs = model(token_ids, valid, steps)
        source_bits = token_ids.gather(1, targets)
        target = canonical_target(model, source_bits, steps)
        bit_loss = F.mse_loss(
            outputs["attention_residual"][:, : model.heads], target[:, : model.heads]
        )
        nuisance = F.mse_loss(
            outputs["attention_residual"][:, model.heads :], target[:, model.heads :]
        )
        loss = bit_loss + 0.02 * nuisance
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 250 == 0 or iteration == maximum:
            actual = outputs["attention_residual"][:, : model.heads]
            accuracy = float(((actual > 0).long() == source_bits).float().mean())
            mse = float(bit_loss.detach())
            if (accuracy, -mse) > (best[0], -best[1]):
                best = (accuracy, mse)
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "step": iteration,
                "routed_bit_accuracy": accuracy,
                "bit_mse": mse,
                "loss": float(loss.detach()),
            }
            history.append(record)
            print(json.dumps({"event": "control_representation", **record}), flush=True)
            if accuracy == 1.0 and mse <= 0.002:
                break
    model.load_state_dict(best_state)
    return {
        "enabled": True,
        "steps": iteration,
        "best_accuracy": best[0],
        "best_mse": best[1],
        "history": history,
    }


@torch.inference_mode()
def public_accuracy(
    model: StandardParityControl, condition: str, device: torch.device
) -> float:
    relevant = 2 if condition == "normal_cot" else min(model.length, model.heads)
    integers = torch.arange(1 << relevant, device=device)
    shifts = torch.arange(relevant - 1, -1, -1, device=device)
    values = ((integers[:, None] >> shifts) & 1).long()
    if condition == "normal_cot":
        values = values.repeat(model.length + 1, 1)
        steps = torch.arange(model.length + 1, device=device).repeat_interleave(4)
    else:
        steps = torch.full(
            (len(values),),
            model.length if condition == "filler_cot" else 0,
            dtype=torch.long,
            device=device,
        )
    bits = torch.zeros((len(values), model.heads), dtype=torch.long, device=device)
    bits[:, :relevant] = values
    target = values.sum(1).remainder(2)
    residual = canonical_target(model, bits, steps)
    return float(
        (model.finish_from_residual(residual)["output_logits"].argmax(-1) == target)
        .float()
        .mean()
    )


def train_public(
    model: StandardParityControl,
    condition: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    relevant = 2 if condition == "normal_cot" else min(model.length, model.heads)
    optimizer = torch.optim.AdamW(
        task_parameters(model),
        lr=float(config["public_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(30_000 + model.length)
    maximum = int(config["public_steps_by_length"][str(model.length)])
    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    # Direct PARITY is introduced by increasing the number of varying input bits.
    # This is ordinary training data curriculum, not an inference shortcut.
    phase_steps = max(250, maximum // (relevant + 2))
    for iteration in range(1, maximum + 1):
        active = min(relevant, 1 + (iteration - 1) // phase_steps)
        values = torch.zeros((512, relevant), dtype=torch.long, device=device)
        values[:, :active] = torch.randint(2, (512, active), generator=generator).to(
            device
        )
        bits = torch.zeros((512, model.heads), dtype=torch.long, device=device)
        bits[:, :relevant] = values
        if condition == "normal_cot":
            steps = torch.randint(model.length + 1, (512,), generator=generator).to(
                device
            )
        else:
            step_value = model.length if condition == "filler_cot" else 0
            steps = torch.full((512,), step_value, dtype=torch.long, device=device)
        targets = values.sum(1).remainder(2)
        predictions = model.finish_from_residual(canonical_target(model, bits, steps))
        loss = F.cross_entropy(predictions["output_logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 250 == 0 or iteration == maximum:
            # With any unsampled input bit still varying at evaluation, direct
            # PARITY is necessarily 50%; avoid an expensive exhaustive pass.
            accuracy = (
                0.5 if active < relevant else public_accuracy(model, condition, device)
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "step": iteration,
                "active_input_bits": active,
                "loss": float(loss.detach()),
                "exhaustive_accuracy": accuracy,
            }
            history.append(record)
            print(json.dumps({"event": "control_public", **record}), flush=True)
            if accuracy == 1.0:
                break
    model.load_state_dict(best_state)
    return {
        "enabled": True,
        "steps": iteration,
        "best_exhaustive_accuracy": best_accuracy,
        "history": history,
    }


@torch.inference_mode()
def evaluate(
    model: StandardParityControl,
    data: dict[str, Any],
    condition: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    answer_correct = cot_bit_correct = cot_exact = 0
    samples = []
    total = len(data["ids"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inputs = data["inputs"][start:end].to(device).long()
        answers = data["answers"][start:end].to(device)
        cache = model.control_cache(inputs)
        if condition == "no_cot":
            predicted_answer = model.decode_step(cache, 0)["output_logits"].argmax(-1)
            predicted_cot = gold_cot = None
        else:
            predictions = []
            for step in range(model.length):
                predicted = model.decode_step(cache, step)["output_logits"].argmax(-1)
                predictions.append(predicted)
                model.append_prediction(cache, step, predicted)
            predicted_cot = torch.stack(predictions, 1)
            gold_cot = (
                data["normal_states"][start:end].to(device)
                if condition == "normal_cot"
                else torch.zeros_like(predicted_cot)
            )
            correct = predicted_cot == gold_cot
            cot_bit_correct += int(correct.sum())
            cot_exact += int(correct.all(1).sum())
            predicted_answer = model.decode_step(cache, model.length)[
                "output_logits"
            ].argmax(-1)
        answer_correct += int((predicted_answer == answers).sum())
        if len(samples) < 4:
            for offset in range(min(4 - len(samples), end - start)):
                sample = {
                    "id": data["ids"][start + offset],
                    "expected_answer": int(answers[offset]),
                    "predicted_answer": int(predicted_answer[offset]),
                }
                if predicted_cot is not None:
                    sample["expected_cot"] = "".join(
                        map(str, gold_cot[offset].cpu().tolist())
                    )
                    sample["predicted_cot"] = "".join(
                        map(str, predicted_cot[offset].cpu().tolist())
                    )
                samples.append(sample)
    return {
        "n": total,
        "cot_length": 0 if condition == "no_cot" else model.length,
        "cot_bit_accuracy": (
            None if condition == "no_cot" else cot_bit_correct / (total * model.length)
        ),
        "cot_exact": None if condition == "no_cot" else cot_exact / total,
        "final_parity_accuracy": answer_correct / total,
        "samples": samples,
    }


def train_integration(
    model: StandardParityControl,
    train_data: dict[str, Any],
    validation_data: dict[str, Any],
    condition: str,
    staged: bool,
    config: dict[str, Any],
    batch_size: int,
    eval_batch: int,
    device: torch.device,
) -> dict[str, Any]:
    parameters = task_parameters(model) if staged else main_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["integration_learning_rate"]), weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(40_000 + model.length)
    maximum = int(config["integration_steps_by_length"][str(model.length)])
    subset_n = min(
        int(config["validation_subset_by_length"][str(model.length)]),
        len(validation_data["ids"]),
    )
    subset = select(validation_data, torch.arange(subset_n))
    best = (-1.0, -1.0)
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for iteration in range(1, maximum + 1):
        token_ids, valid, steps, targets, *_ = sampled_batch(
            train_data, model, condition, iteration, batch_size, generator, device
        )
        outputs = model(token_ids, valid, steps)
        loss = F.cross_entropy(outputs["output_logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 250 == 0 or iteration == maximum:
            metrics = evaluate(model, subset, condition, device, eval_batch)
            answer_score = metrics["final_parity_accuracy"]
            cot_score = 1.0 if metrics["cot_exact"] is None else metrics["cot_exact"]
            # Select for joint success rather than allowing a perfect answer to
            # hide a bad generated trace (especially in the filler condition).
            score = (min(answer_score, cot_score), answer_score + cot_score)
            if score > best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "step": iteration,
                "loss": float(loss.detach()),
                "validation": metrics,
            }
            history.append(record)
            print(json.dumps({"event": "control_integration", **record}), flush=True)
            if min(score) >= 0.999:
                break
    model.load_state_dict(best_state)
    return {"steps": iteration, "best_validation_score": best, "history": history}


def train(
    config_path: Path, data_root: Path, output_root: Path, length: int, condition: str
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if condition not in CONDITIONS or length not in map(int, config["lengths"]):
        raise ValueError("Unknown condition or length")
    set_seed(int(config["seed"]) + int(config["model_dim"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = {
        split: load_data(data_root / f"n{length}" / f"{split}.jsonl.gz")
        for split in ("train", "validation", "test")
    }
    audit = sanitize_and_audit(parts)
    seed_padding = int(config["seed_padding_by_length"][str(length)])
    model = StandardParityControl(
        length=length,
        seed_bits=seed_padding,
        graph=[[0, 1, 2, 3, 4] for _ in range(length)],
        model_dim=int(config["model_dim"]),
        heads=int(config["heads"]),
        token_dim=int(config["token_embedding_dim"]),
        ff_dim=int(config["model_dim"]) * int(config["ff_multiplier"]),
    ).to(device)
    batch_size = int(config["batch_size_by_length"][str(length)])
    eval_batch = int(config["evaluation_batch_size_by_length"][str(length)])
    direct_lengths = set(map(int, config["direct_routing_lengths"]))
    staged = condition == "normal_cot" or length in direct_lengths
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    if staged:
        routing = train_routes(
            model, parts["train"], condition, direct_lengths, config, batch_size, device
        )
        representation = train_representation(
            model, parts["train"], condition, direct_lengths, config, batch_size, device
        )
        public = train_public(model, condition, config, device)
    else:
        routing = representation = public = {
            "enabled": False,
            "reason": "No public local route can retrieve all input bits at this length",
        }
    integration = train_integration(
        model,
        parts["train"],
        parts["validation"],
        condition,
        staged,
        config,
        batch_size,
        eval_batch,
        device,
    )
    validation = evaluate(model, parts["validation"], condition, device, eval_batch)
    test = evaluate(model, parts["test"], condition, device, eval_batch)
    direct = length in direct_lengths and length <= model.heads
    cot_ok = condition == "no_cot" or test["cot_exact"] >= 0.99
    expected = (
        test["final_parity_accuracy"] >= 0.99 and cot_ok
        if condition == "normal_cot" or direct
        else abs(test["final_parity_accuracy"] - 0.5)
        <= float(config["chance_tolerance"])
        and cot_ok
    )
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "condition": condition,
        "length": length,
        "seed": config["seed"],
        "private_seed_given_to_model": False,
        "public_zero_padding": seed_padding,
        "data_fields_loaded": ["id", "input_bits", "answer"],
        "architecture": model.architecture_metadata(),
        "training_regime": "staged" if staged else "end_to_end_only",
        "staged_recipe": (
            {
                "routing_supervision": True,
                "representation_supervision": True,
                "public_function_curriculum": True,
                "integration_checkpoint_selection": "joint validation CoT and answer accuracy",
                "directly_supervised_input_bits": (
                    2 if condition == "normal_cot" else min(length, model.heads)
                ),
                "total_input_bits": length,
                "deviation_from_easy_recipe": (
                    None
                    if condition == "normal_cot" or length <= model.heads
                    else f"The fixed {model.heads}-head architecture directly supervises only "
                    f"the first {model.heads} of {length} inputs during routing, representation, "
                    "and public-function pretraining; final integration and evaluation use all inputs."
                ),
            }
            if staged
            else None
        ),
        "train_examples": len(parts["train"]["ids"]),
        "validation_examples": len(parts["validation"]["ids"]),
        "test_examples": len(parts["test"]["ids"]),
        "split_audit": audit,
        "routing_training": routing,
        "representation_training": representation,
        "public_training": public,
        "integration_training": integration,
        "validation_softmax_attention": validation,
        "test_softmax_attention": test,
        "matches_expected_pattern": expected,
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(
        {"model_state": model.state_dict(), "config": config, "condition": condition},
        output_root / "model.pt",
    )
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    train(args.config, args.data_root, args.output_root, args.length, args.condition)


if __name__ == "__main__":
    main()
