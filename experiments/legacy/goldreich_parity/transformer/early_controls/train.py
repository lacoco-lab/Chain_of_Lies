#!/usr/bin/env python3
"""Train matched no-CoT, filler-CoT, and ordinary-CoT PARITY controls."""

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

from model import HEADS, ParityControlTransformer

CONDITIONS = ("no_cot", "filler_cot", "normal_cot")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def bits_tensor(values: list[str]) -> torch.Tensor:
    width = len(values[0])
    packed = bytearray("".join(values), "ascii")
    return (
        torch.frombuffer(packed, dtype=torch.uint8).reshape(len(values), width).sub(48)
    )


def load_data(path: Path) -> dict[str, Any]:
    """Load only task inputs and answers; deliberately ignore every seed field."""
    ids, inputs, answers = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            inputs.append(row["input_bits"])
            answers.append(int(row["answer"]))
    tensor = bits_tensor(inputs)
    return {
        "ids": ids,
        "input_strings": inputs,
        "inputs": tensor,
        "normal_states": torch.cumsum(tensor.long(), dim=1).remainder(2).byte(),
        "answers": torch.tensor(answers, dtype=torch.long),
    }


def select_rows(data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    selected = torch.tensor(indices, dtype=torch.long)
    return {
        key: (
            [value[index] for index in indices]
            if isinstance(value, list)
            else value[selected]
        )
        for key, value in data.items()
    }


def sanitize_evaluation_overlap(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Keep train/test fixed; make validation disjoint and label-balanced."""
    train_inputs = set(parts["train"]["input_strings"])
    validation_inputs = set(parts["validation"]["input_strings"])
    test_inputs = set(parts["test"]["input_strings"])
    train_validation = train_inputs & validation_inputs
    train_test = train_inputs & test_inputs
    if train_validation or train_test:
        raise RuntimeError(
            "Training inputs overlap evaluation inputs: "
            f"train/validation={len(train_validation)}, train/test={len(train_test)}"
        )

    original_n = len(parts["validation"]["ids"])
    collision_inputs = validation_inputs & test_inputs
    candidates = [
        index
        for index, value in enumerate(parts["validation"]["input_strings"])
        if value not in collision_inputs
    ]
    counts = {
        label: sum(
            int(parts["validation"]["answers"][index]) == label for index in candidates
        )
        for label in (0, 1)
    }
    per_label = min(counts.values())
    kept, used = [], {0: 0, 1: 0}
    for index in candidates:
        label = int(parts["validation"]["answers"][index])
        if used[label] < per_label:
            kept.append(index)
            used[label] += 1
    parts["validation"] = select_rows(parts["validation"], kept)
    return {
        "policy": "train and test unchanged; remove validation/test collisions from validation, then rebalance validation",
        "validation_test_colliding_inputs": len(collision_inputs),
        "validation_rows_before": original_n,
        "validation_rows_after": len(kept),
        "validation_rows_removed": original_n - len(kept),
    }


def audit_splits(
    parts: dict[str, dict[str, Any]], require_disjoint: bool = True
) -> dict[str, Any]:
    def fingerprints(values: list[str]) -> set[bytes]:
        return {
            hashlib.blake2b(value.encode(), digest_size=16).digest() for value in values
        }

    sets = {name: fingerprints(data["input_strings"]) for name, data in parts.items()}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    if require_disjoint and any(overlaps.values()):
        raise RuntimeError(f"Control input splits overlap: {overlaps}")
    balance = {
        name: {
            "zero": int((data["answers"] == 0).sum()),
            "one": int((data["answers"] == 1).sum()),
        }
        for name, data in parts.items()
    }
    if any(item["zero"] != item["one"] for item in balance.values()):
        raise RuntimeError(f"Control labels are not balanced: {balance}")
    return {
        "input_overlap_counts": overlaps,
        "input_disjointness_required": require_disjoint,
        "label_balance": balance,
    }


def xor_truth_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=device)
    return values, targets


@torch.no_grad()
def gate_metrics(
    model: ParityControlTransformer, device: torch.device
) -> dict[str, float]:
    values, targets = xor_truth_table(device)
    probabilities = model.xor_gate(values).softmax(-1)
    return {
        "xor_accuracy": float((probabilities.argmax(-1) == targets).float().mean()),
        "xor_min_correct_probability": float(
            probabilities.gather(1, targets[:, None]).min()
        ),
    }


def train_gate(
    model: ParityControlTransformer, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    values, targets = xor_truth_table(device)
    optimizer = torch.optim.AdamW(
        model.xor_gate.parameters(),
        lr=float(config["gate_learning_rate"]),
        weight_decay=0.0,
    )
    history = []
    for epoch in range(1, int(config["gate_max_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.xor_gate(values), targets)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            metrics = gate_metrics(model, device)
            history.append({"epoch": epoch, "loss": float(loss.detach()), **metrics})
            print(json.dumps({"event": "xor_gate", **history[-1]}), flush=True)
            if (
                metrics["xor_accuracy"] == 1.0
                and metrics["xor_min_correct_probability"] >= 0.999
            ):
                break
    metrics = gate_metrics(model, device)
    if metrics["xor_accuracy"] != 1.0:
        raise RuntimeError(f"XOR gate training failed: {metrics}")
    return {"epochs": epoch, "metrics": metrics, "history": history}


def attention_loss(
    model: ParityControlTransformer, steps: torch.Tensor, routing_kind: str = "normal"
) -> torch.Tensor:
    scores = model.attention_scores(steps)
    targets = (
        model.normal_routing_targets(steps)
        if routing_kind == "normal"
        else model.direct_routing_targets(steps)
    )
    return F.cross_entropy(scores.reshape(-1, model.sources), targets.reshape(-1))


@torch.no_grad()
def routing_metrics(
    model: ParityControlTransformer, device: torch.device, routing_kind: str = "normal"
) -> dict[str, Any]:
    steps = (
        torch.arange(model.steps, device=device)
        if routing_kind == "normal"
        else torch.tensor([model.steps - 1], device=device)
    )
    scores = model.attention_scores(steps)
    probabilities = scores.softmax(-1)
    targets = (
        model.normal_routing_targets(steps)
        if routing_kind == "normal"
        else model.direct_routing_targets(steps)
    )
    selected = probabilities.argmax(-1)
    mass = probabilities.gather(-1, targets[..., None]).squeeze(-1)
    return {
        "accuracy": float((selected == targets).float().mean()),
        "all_routes_correct": bool(torch.equal(selected, targets)),
        "minimum_target_attention": float(mass.min()),
        "mean_target_attention": float(mass.mean()),
    }


def train_routing(
    model: ParityControlTransformer,
    config: dict[str, Any],
    device: torch.device,
    routing_kind: str = "normal",
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        [model.keys, model.queries],
        lr=float(config["routing_learning_rate"]),
        weight_decay=0.0,
    )
    steps = (
        torch.arange(model.steps, device=device)
        if routing_kind == "normal"
        else torch.tensor([model.steps - 1], device=device)
    )
    history = []
    for epoch in range(1, int(config["routing_max_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = attention_loss(model, steps, routing_kind)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            metrics = routing_metrics(model, device, routing_kind)
            history.append({"epoch": epoch, "loss": float(loss.detach()), **metrics})
            print(json.dumps({"event": "routing", **history[-1]}), flush=True)
            if metrics["all_routes_correct"] and metrics[
                "minimum_target_attention"
            ] >= float(config["success_gate"]["minimum_target_attention"]):
                break
    metrics = routing_metrics(model, device, routing_kind)
    if not metrics["all_routes_correct"]:
        raise RuntimeError(f"{routing_kind} routing failed: {metrics}")
    return {"epochs": epoch, "metrics": metrics, "history": history}


def train_normal(
    model: ParityControlTransformer,
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    for parameter in model.xor_gate.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [model.keys, model.queries],
        lr=float(config["integration_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(int(config["seed"]) + model.length)
    steps_n = int(config["integration_steps_by_length"][str(model.length)])
    batch_size = int(config["integration_batch_size"])
    best_state = copy.deepcopy(model.state_dict())
    best_mass = routing_metrics(model, device)["minimum_target_attention"]
    history = []
    for iteration in range(1, steps_n + 1):
        rows = torch.randint(len(data["ids"]), (batch_size,), generator=generator)
        steps = torch.randint(model.steps, (batch_size,), generator=generator)
        inputs = data["inputs"][rows].to(device)
        states = data["normal_states"][rows]
        previous = torch.zeros(batch_size, dtype=torch.long)
        active = steps > 0
        previous[active] = states[active, steps[active] - 1].long()
        targets = states[torch.arange(batch_size), steps].long().to(device)
        previous = previous.to(device)
        steps = steps.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model.soft_step(inputs, previous, steps)
        state_loss = F.cross_entropy(output["logits"], targets)
        route_loss = attention_loss(model, steps)
        loss = state_loss + float(config["attention_loss_weight"]) * route_loss
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0:
            route = routing_metrics(model, device)
            hard = model.hard_step(inputs, previous, steps)["state"]
            record = {
                "step": iteration,
                "loss": float(loss.detach()),
                "hard_state_accuracy": float((hard == targets).float().mean()),
                **route,
            }
            history.append(record)
            print(json.dumps({"event": "normal_integration", **record}), flush=True)
            if (
                route["all_routes_correct"]
                and route["minimum_target_attention"] > best_mass
            ):
                best_mass = route["minimum_target_attention"]
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"steps": steps_n, "history": history}


def train_direct(
    model: ParityControlTransformer,
    data: dict[str, Any],
    condition: str,
    config: dict[str, Any],
    device: torch.device,
    supervise_direct_routing: bool = False,
) -> dict[str, Any]:
    for parameter in model.xor_gate.parameters():
        parameter.requires_grad_(False)
    parameters = [model.keys, model.queries]
    if condition == "filler_cot":
        parameters.append(model.filler_logits)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["integration_learning_rate"]), weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(int(config["seed"]) + model.length)
    steps_n = int(config["integration_steps_by_length"][str(model.length)])
    batch_size = int(config["integration_batch_size"])
    history = []
    final_step = model.steps - 1
    for iteration in range(1, steps_n + 1):
        rows = torch.randint(len(data["ids"]), (batch_size,), generator=generator)
        inputs = data["inputs"][rows].to(device)
        answers = data["answers"][rows].to(device)
        previous = torch.zeros(batch_size, dtype=torch.long, device=device)
        steps = torch.full((batch_size,), final_step, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.soft_step(inputs, previous, steps)
        answer_loss = F.cross_entropy(output["logits"], answers)
        filler_loss = (
            F.cross_entropy(
                model.filler_logits[None, :],
                torch.zeros(1, dtype=torch.long, device=device),
            )
            if condition == "filler_cot"
            else torch.zeros((), device=device)
        )
        route_loss = (
            attention_loss(model, steps, "direct")
            if supervise_direct_routing
            else torch.zeros((), device=device)
        )
        loss = (
            answer_loss
            + filler_loss
            + float(config["attention_loss_weight"]) * route_loss
        )
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0:
            hard = model.hard_step(inputs, previous, steps)["state"]
            record = {
                "step": iteration,
                "loss": float(loss.detach()),
                "answer_loss": float(answer_loss.detach()),
                "hard_answer_accuracy": float((hard == answers).float().mean()),
                "filler_token": (
                    int(model.filler_logits.argmax())
                    if condition == "filler_cot"
                    else None
                ),
            }
            if supervise_direct_routing:
                record["routing"] = routing_metrics(model, device, "direct")
            history.append(record)
            print(json.dumps({"event": "direct_training", **record}), flush=True)
    return {"steps": steps_n, "history": history}


@torch.no_grad()
def evaluate(
    model: ParityControlTransformer,
    data: dict[str, Any],
    condition: str,
    device: torch.device,
    batch_size: int = 1024,
    hard: bool = True,
) -> dict[str, Any]:
    total = len(data["ids"])
    answer_correct = 0
    cot_bit_correct = cot_exact = 0
    cot_bits_total = 0
    samples = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inputs = data["inputs"][start:end].to(device)
        answers = data["answers"][start:end].to(device)
        size = end - start
        if condition == "normal_cot":
            previous = torch.zeros(size, dtype=torch.long, device=device)
            predictions = []
            for step in range(model.steps):
                steps = torch.full((size,), step, dtype=torch.long, device=device)
                output = (
                    model.hard_step(inputs, previous, steps)
                    if hard
                    else model.soft_step(inputs, previous, steps)
                )
                previous = output["state"] if hard else output["logits"].argmax(-1)
                predictions.append(previous)
            predicted_cot = torch.stack(predictions, dim=1)
            gold_cot = data["normal_states"][start:end].to(device)
            correct = predicted_cot == gold_cot
            cot_bit_correct += int(correct.sum())
            cot_bits_total += correct.numel()
            cot_exact += int(correct.all(dim=1).sum())
            predicted_answer = previous
        else:
            previous = torch.zeros(size, dtype=torch.long, device=device)
            steps = torch.full(
                (size,), model.steps - 1, dtype=torch.long, device=device
            )
            output = (
                model.hard_step(inputs, previous, steps)
                if hard
                else model.soft_step(inputs, previous, steps)
            )
            predicted_answer = output["state"] if hard else output["logits"].argmax(-1)
            if condition == "filler_cot":
                filler = int(model.filler_logits.argmax())
                cot_bits_total += size * model.steps
                if filler == 0:
                    cot_bit_correct += size * model.steps
                    cot_exact += size
        answer_correct += int((predicted_answer == answers).sum())
        if len(samples) < 8:
            for offset in range(min(8 - len(samples), size)):
                sample = {
                    "id": data["ids"][start + offset],
                    "input": "".join(map(str, inputs[offset].cpu().tolist())),
                    "expected_answer": int(answers[offset]),
                    "predicted_answer": int(predicted_answer[offset]),
                }
                if condition == "normal_cot":
                    sample["expected_cot"] = "".join(
                        map(str, gold_cot[offset].cpu().tolist())
                    )
                    sample["predicted_cot"] = "".join(
                        map(str, predicted_cot[offset].cpu().tolist())
                    )
                elif condition == "filler_cot":
                    sample["filler_token"] = int(model.filler_logits.argmax())
                    sample["filler_count"] = model.steps
                samples.append(sample)
    result = {
        "n": total,
        "cot_length": 0 if condition == "no_cot" else model.steps,
        "cot_bit_accuracy": (
            None if condition == "no_cot" else cot_bit_correct / cot_bits_total
        ),
        "cot_exact": None if condition == "no_cot" else cot_exact / total,
        "final_parity_accuracy": answer_correct / total,
        "samples": samples,
    }
    return result


def expected_behavior(
    condition: str, metrics: dict[str, Any], config: dict[str, Any], length: int
) -> bool:
    gate = config["success_gate"]
    directly_representable = length in set(
        map(int, config.get("direct_routing_lengths", []))
    )
    if condition == "normal_cot" or directly_representable:
        cot_ok = condition == "no_cot" or metrics["cot_exact"] >= float(
            gate["minimum_cot_exact"]
        )
        return (
            (
                metrics["cot_bit_accuracy"] >= float(gate["minimum_cot_bit_accuracy"])
                if condition != "no_cot"
                else True
            )
            and cot_ok
            and (
                metrics["final_parity_accuracy"]
                >= float(gate["minimum_final_parity_accuracy"])
            )
        )
    return abs(metrics["final_parity_accuracy"] - 0.5) <= float(
        gate["chance_tolerance"]
    )


def train(
    config_path: Path, data_root: Path, output_root: Path, length: int, condition: str
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {CONDITIONS}")
    if length not in map(int, config["lengths"]):
        raise ValueError(f"N={length} is not configured")
    if int(config["heads"]) != HEADS:
        raise ValueError(f"Matched controls require exactly {HEADS} heads")
    set_seed(int(config["seed"]))
    device = torch.device("cpu")
    parts = {
        split: load_data(data_root / f"n{length}" / f"{split}.jsonl.gz")
        for split in ("train", "validation", "test")
    }
    sanitization = sanitize_evaluation_overlap(parts)
    audit = audit_splits(
        parts, require_disjoint=bool(config.get("require_disjoint_inputs", True))
    )
    model = ParityControlTransformer(
        length=length,
        padding_bits=int(config["public_zero_padding_by_length"][str(length)]),
        key_dim=int(config["key_dim_by_length"][str(length)]),
        gate_hidden_dim=int(config["gate_hidden_dim"]),
    ).to(device)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    gate_stage = train_gate(model, config, device)
    directly_representable = length in set(
        map(int, config.get("direct_routing_lengths", []))
    )
    if condition == "normal_cot":
        routing_stage = train_routing(model, config, device, "normal")
        training_stage = train_normal(model, parts["train"], config, device)
        final_routing = routing_metrics(model, device)
    elif directly_representable:
        routing_stage = train_routing(model, config, device, "direct")
        training_stage = train_direct(
            model,
            parts["train"],
            condition,
            config,
            device,
            supervise_direct_routing=True,
        )
        final_routing = routing_metrics(model, device, "direct")
    else:
        routing_stage = {
            "enabled": False,
            "reason": "No sufficient set of public input positions computes parity directly",
        }
        training_stage = train_direct(model, parts["train"], condition, config, device)
        final_routing = None
    validation_hard = evaluate(model, parts["validation"], condition, device, hard=True)
    test_hard = evaluate(model, parts["test"], condition, device, hard=True)
    soft_n = min(
        int(config["soft_validation_n_by_length"][str(length)]),
        len(parts["validation"]["ids"]),
    )
    soft_part = {
        key: value[:soft_n] if isinstance(value, (list, torch.Tensor)) else value
        for key, value in parts["validation"].items()
    }
    validation_soft = evaluate(model, soft_part, condition, device, hard=False)
    passed = expected_behavior(condition, test_hard, config, length)
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "data_source": config["data_source"],
        "condition": condition,
        "seed": int(config["seed"]),
        "length": length,
        "private_seed_given_to_model": False,
        "data_fields_loaded": ["id", "input_bits", "answer"],
        "architecture": model.architecture_metadata(),
        "train_examples": len(parts["train"]["ids"]),
        "validation_examples": len(parts["validation"]["ids"]),
        "test_examples": len(parts["test"]["ids"]),
        "data_sanitization": sanitization,
        "data_audit": audit,
        "gate_stage": gate_stage,
        "routing_stage": routing_stage,
        "training_stage": training_stage,
        "final_routing_metrics": final_routing,
        "validation_hard_attention": validation_hard,
        "validation_soft_attention": validation_soft,
        "test_hard_attention": test_hard,
        "matches_expected_behavior": bool(passed),
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(
        {"model_state": model.state_dict(), "config": config, "condition": condition},
        output_root / "model.pt",
    )
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    result = train(
        args.config, args.data_root, args.output_root, args.length, args.condition
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
