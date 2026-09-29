#!/usr/bin/env python3
"""Train and evaluate a purpose-built hard-attention Goldreich transformer."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_data import fixed_graph, seed_bits_for_length
from model import GoldreichToyTransformer


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
    ids, seeds, inputs, masks, states, answers = [], [], [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            seeds.append(row["seed_bits"])
            inputs.append(row["input_bits"])
            masks.append(row["masks"])
            states.append(row["encrypted_states"])
            answers.append(int(row["answer"]))
    return {
        "ids": ids,
        "seeds": bits_tensor(seeds),
        "inputs": bits_tensor(inputs),
        "masks": bits_tensor(masks),
        "states": bits_tensor(states),
        "answers": torch.tensor(answers, dtype=torch.long),
    }


def truth_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    predicate_x = torch.tensor(
        [[(value >> shift) & 1 for shift in range(4, -1, -1)] for value in range(32)],
        dtype=torch.float32,
        device=device,
    )
    predicate_y = (
        predicate_x[:, 0].long()
        ^ predicate_x[:, 1].long()
        ^ predicate_x[:, 2].long()
        ^ (predicate_x[:, 3].long() & predicate_x[:, 4].long())
    )
    xor_x = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device
    )
    xor_y = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=device)
    return predicate_x, predicate_y, xor_x, xor_y


@torch.no_grad()
def gate_metrics(
    model: GoldreichToyTransformer, device: torch.device
) -> dict[str, float]:
    predicate_x, predicate_y, xor_x, xor_y = truth_tables(device)
    predicate_prob = model.predicate_gate(predicate_x).softmax(-1)
    xor_prob = model.xor_gate(xor_x).softmax(-1)
    return {
        "predicate_accuracy": float(
            (predicate_prob.argmax(-1) == predicate_y).float().mean()
        ),
        "predicate_min_correct_probability": float(
            predicate_prob.gather(1, predicate_y[:, None]).min()
        ),
        "xor_accuracy": float((xor_prob.argmax(-1) == xor_y).float().mean()),
        "xor_min_correct_probability": float(xor_prob.gather(1, xor_y[:, None]).min()),
    }


def train_gates(
    model: GoldreichToyTransformer, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    predicate_x, predicate_y, xor_x, xor_y = truth_tables(device)
    parameters = [*model.predicate_gate.parameters(), *model.xor_gate.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["gate_learning_rate"]), weight_decay=0.0
    )
    history = []
    for epoch in range(1, int(config["gate_max_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        predicate_loss = F.cross_entropy(model.predicate_gate(predicate_x), predicate_y)
        xor_loss = F.cross_entropy(model.xor_gate(xor_x), xor_y)
        loss = predicate_loss + xor_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            metrics = gate_metrics(model, device)
            history.append({"epoch": epoch, "loss": float(loss.detach()), **metrics})
            print(json.dumps({"event": "gate_training", **history[-1]}), flush=True)
            if (
                metrics["predicate_accuracy"] == 1.0
                and metrics["xor_accuracy"] == 1.0
                and min(
                    metrics["predicate_min_correct_probability"],
                    metrics["xor_min_correct_probability"],
                )
                >= 0.999
            ):
                break
    metrics = gate_metrics(model, device)
    if metrics["predicate_accuracy"] != 1.0 or metrics["xor_accuracy"] != 1.0:
        raise RuntimeError(f"Boolean truth-table training failed: {metrics}")
    return {"epochs": epoch, "metrics": metrics, "history": history}


@torch.no_grad()
def routing_metrics(
    model: GoldreichToyTransformer, device: torch.device
) -> dict[str, Any]:
    steps = torch.arange(model.steps, device=device)
    scores = model.attention_scores(steps)
    probabilities = scores.softmax(-1)
    targets = model.routing_targets(steps)
    predicted = probabilities.argmax(-1)
    target_mass = probabilities.gather(-1, targets[..., None]).squeeze(-1)
    per_head = (predicted == targets).float().mean(dim=0)
    return {
        "accuracy": float((predicted == targets).float().mean()),
        "all_routes_correct": bool(torch.equal(predicted, targets)),
        "minimum_target_attention": float(target_mass.min()),
        "mean_target_attention": float(target_mass.mean()),
        "per_head_accuracy": [float(value) for value in per_head],
    }


def attention_loss(model: GoldreichToyTransformer, steps: torch.Tensor) -> torch.Tensor:
    scores = model.attention_scores(steps)
    targets = model.routing_targets(steps)
    return F.cross_entropy(scores.reshape(-1, model.sources), targets.reshape(-1))


def train_routing(
    model: GoldreichToyTransformer, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        [model.keys, model.queries],
        lr=float(config["routing_learning_rate"]),
        weight_decay=0.0,
    )
    steps = torch.arange(model.steps, device=device)
    history = []
    for epoch in range(1, int(config["routing_max_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = attention_loss(model, steps)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            metrics = routing_metrics(model, device)
            history.append({"epoch": epoch, "loss": float(loss.detach()), **metrics})
            print(json.dumps({"event": "routing_training", **history[-1]}), flush=True)
            if metrics["all_routes_correct"] and metrics[
                "minimum_target_attention"
            ] >= float(config["success_gate"]["minimum_target_attention"]):
                break
    metrics = routing_metrics(model, device)
    if not metrics["all_routes_correct"]:
        raise RuntimeError(f"Attention routing failed: {metrics}")
    return {"epochs": epoch, "metrics": metrics, "history": history}


def integration_batch(
    data: dict[str, Any],
    batch_size: int,
    steps_n: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    rows = torch.randint(len(data["ids"]), (batch_size,), generator=generator)
    steps = torch.randint(steps_n, (batch_size,), generator=generator)
    seeds = data["seeds"][rows].to(device)
    inputs = data["inputs"][rows].to(device)
    states = data["states"][rows]
    masks = data["masks"][rows]
    batch_rows = torch.arange(batch_size)
    previous = torch.zeros(batch_size, dtype=torch.long)
    noninitial = steps > 0
    previous[noninitial] = states[noninitial, steps[noninitial] - 1].long()
    previous_masks = torch.zeros(batch_size, dtype=torch.long)
    previous_masks[noninitial] = masks[noninitial, steps[noninitial] - 1].long()
    final_previous = (
        states[:, -2].long()
        if steps_n > 1
        else torch.zeros(batch_size, dtype=torch.long)
    )
    return (
        seeds,
        inputs,
        previous.to(device),
        steps.to(device),
        states[batch_rows, steps].long().to(device),
        previous_masks.long().to(device),
        masks[batch_rows, steps].long().to(device),
        final_previous.to(device),
        data["answers"][rows].to(device),
    )


def task_gate_passes(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        metrics["state_bit_accuracy"] >= float(gate["minimum_state_bit_accuracy"])
        and metrics["trace_exact"] >= float(gate["minimum_trace_exact"])
        and metrics["final_parity_accuracy"]
        >= float(gate["minimum_final_parity_accuracy"])
        and metrics["joint_exact"] >= float(gate["minimum_joint_exact"])
    )


def compact_task_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in (
            "state_bit_accuracy",
            "trace_exact",
            "final_parity_accuracy",
            "joint_exact",
        )
    }


def train_integration(
    model: GoldreichToyTransformer,
    data: dict[str, Any],
    selection_data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    supervision: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    for parameter in model.predicate_gate.parameters():
        parameter.requires_grad_(False)
    for parameter in model.xor_gate.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [model.keys, model.queries],
        lr=float(config["integration_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(int(config["seed"]) + model.length)
    best_state = copy.deepcopy(model.state_dict())
    best_mass = routing_metrics(model, device)["minimum_target_attention"]
    best_selection_score = (-1.0, -1.0, -1.0, -1.0)
    history = []
    integration_steps = int(config["integration_steps_by_length"][str(model.length)])
    for step_index in range(1, integration_steps + 1):
        batch = integration_batch(
            data, int(config["integration_batch_size"]), model.steps, generator, device
        )
        (
            seeds,
            inputs,
            previous,
            steps,
            targets,
            previous_masks,
            current_masks,
            final_previous,
            answers,
        ) = batch
        optimizer.zero_grad(set_to_none=True)
        outputs = model.soft_step(seeds, inputs, previous, steps)
        state_loss = F.cross_entropy(outputs["state_logits"], targets)
        mask_loss = F.cross_entropy(
            outputs["previous_mask_logits"], previous_masks
        ) + F.cross_entropy(outputs["current_mask_logits"], current_masks)
        route_loss = (
            attention_loss(model, steps)
            if bool(settings["attention_supervision"])
            else torch.zeros((), device=device)
        )
        answer_weight = float(settings["answer_loss_weight"])
        if answer_weight:
            final_steps = torch.full_like(steps, model.steps - 1)
            final_outputs = model.soft_step(seeds, inputs, final_previous, final_steps)
            final_state = final_outputs["state_logits"].softmax(-1)[:, 1]
            final_mask = final_outputs["current_mask_logits"].softmax(-1)[:, 1]
            answer_loss = F.cross_entropy(
                model.soft_answer(final_state, final_mask), answers
            )
        else:
            final_outputs = None
            answer_loss = torch.zeros((), device=device)
        entropy = (
            -(
                outputs["attention"].clamp_min(1e-12)
                * outputs["attention"].clamp_min(1e-12).log()
            )
            .sum(-1)
            .mean()
        )
        if final_outputs is not None:
            final_entropy = (
                -(
                    final_outputs["attention"].clamp_min(1e-12)
                    * final_outputs["attention"].clamp_min(1e-12).log()
                )
                .sum(-1)
                .mean()
            )
            entropy = 0.5 * (entropy + final_entropy)
        loss = (
            state_loss
            + float(settings["mask_loss_weight"]) * mask_loss
            + float(settings["attention_loss_weight"]) * route_loss
            + answer_weight * answer_loss
            + float(settings["entropy_loss_weight"]) * entropy
        )
        loss.backward()
        optimizer.step()
        interval = (
            100
            if bool(settings["attention_supervision"])
            else int(config["selection_interval"])
        )
        if step_index == 1 or step_index % interval == 0:
            route = routing_metrics(model, device)
            hard = model.hard_step(seeds, inputs, previous, steps)["state"]
            record = {
                "step": step_index,
                "loss": float(loss.detach()),
                "soft_state_accuracy": float(
                    (outputs["state_logits"].argmax(-1) == targets).float().mean()
                ),
                "hard_state_accuracy": float((hard == targets).float().mean()),
                "routing_accuracy": route["accuracy"],
                "minimum_target_attention": route["minimum_target_attention"],
                "attention_entropy": float(entropy.detach()),
                "previous_mask_accuracy": float(
                    (outputs["previous_mask_logits"].argmax(-1) == previous_masks)
                    .float()
                    .mean()
                ),
                "current_mask_accuracy": float(
                    (outputs["current_mask_logits"].argmax(-1) == current_masks)
                    .float()
                    .mean()
                ),
            }
            if answer_weight:
                record["answer_loss"] = float(answer_loss.detach())
            if (
                not bool(settings["attention_supervision"])
                and step_index % interval == 0
            ):
                selected = evaluate(
                    model, selection_data, device, batch_size=256, hard=True
                )
                record["selection"] = compact_task_metrics(selected)
                score = (
                    selected["joint_exact"],
                    selected["trace_exact"],
                    selected["final_parity_accuracy"],
                    selected["state_bit_accuracy"],
                )
                if score > best_selection_score:
                    best_selection_score = score
                    best_state = copy.deepcopy(model.state_dict())
            history.append(record)
            print(json.dumps({"event": "integration_training", **record}), flush=True)
            if bool(settings["attention_supervision"]) and (
                route["all_routes_correct"]
                and route["minimum_target_attention"] > best_mass
            ):
                best_mass = route["minimum_target_attention"]
                best_state = copy.deepcopy(model.state_dict())
            if (
                not bool(settings["attention_supervision"])
                and "selection" in record
                and task_gate_passes(record["selection"], config["success_gate"])
            ):
                break
    model.load_state_dict(best_state)
    return {
        "steps": step_index,
        "supervision": supervision,
        "settings": settings,
        "routing": routing_metrics(model, device),
        "history": history,
    }


@torch.no_grad()
def evaluate(
    model: GoldreichToyTransformer,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int = 1024,
    hard: bool = True,
) -> dict[str, Any]:
    total = len(data["ids"])
    bit_correct = trace_correct = answer_correct = joint_correct = 0
    per_step_correct = torch.zeros(model.steps, dtype=torch.long)
    samples = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        seeds = data["seeds"][start:end].to(device)
        inputs = data["inputs"][start:end].to(device)
        gold_states = data["states"][start:end].to(device)
        gold_answers = data["answers"][start:end].to(device)
        previous = torch.zeros(end - start, dtype=torch.long, device=device)
        predictions, final_mask = [], None
        for step in range(model.steps):
            steps = torch.full((end - start,), step, dtype=torch.long, device=device)
            if hard:
                output = model.hard_step(seeds, inputs, previous, steps)
                previous, final_mask = output["state"], output["current_mask"]
            else:
                output = model.soft_step(seeds, inputs, previous, steps)
                previous = output["state_logits"].argmax(-1)
                final_mask = output["current_mask_logits"].argmax(-1)
            predictions.append(previous)
        predicted_states = torch.stack(predictions, dim=1)
        predicted_answer = model.hard_answer(previous, final_mask)
        correct = predicted_states == gold_states
        exact = correct.all(dim=1)
        answers = predicted_answer == gold_answers
        bit_correct += int(correct.sum())
        per_step_correct += correct.sum(dim=0).cpu()
        trace_correct += int(exact.sum())
        answer_correct += int(answers.sum())
        joint_correct += int((exact & answers).sum())
        if len(samples) < 8:
            for offset in range(min(8 - len(samples), end - start)):
                samples.append(
                    {
                        "id": data["ids"][start + offset],
                        "expected_states": "".join(
                            map(str, gold_states[offset].cpu().tolist())
                        ),
                        "predicted_states": "".join(
                            map(str, predicted_states[offset].cpu().tolist())
                        ),
                        "expected_answer": int(gold_answers[offset]),
                        "predicted_answer": int(predicted_answer[offset]),
                    }
                )
    return {
        "n": total,
        "state_bit_accuracy": bit_correct / (total * model.steps),
        "trace_exact": trace_correct / total,
        "final_parity_accuracy": answer_correct / total,
        "joint_exact": joint_correct / total,
        "per_step_accuracy": [int(value) / total for value in per_step_correct],
        "samples": samples,
    }


def passes(
    metrics: dict[str, Any],
    gate_metrics_: dict[str, float],
    routing: dict[str, Any],
    gate: dict[str, float],
    require_routing: bool,
) -> bool:
    return (
        min(gate_metrics_["predicate_accuracy"], gate_metrics_["xor_accuracy"])
        >= float(gate["minimum_gate_accuracy"])
        and (
            not require_routing
            or (
                routing["accuracy"] >= float(gate["minimum_routing_accuracy"])
                and routing["minimum_target_attention"]
                >= float(gate["minimum_target_attention"])
            )
        )
        and task_gate_passes(metrics, gate)
    )


def supervision_settings(config: dict[str, Any], supervision: str) -> dict[str, Any]:
    if supervision == "full":
        return {
            "attention_supervision": True,
            "attention_loss_weight": float(config.get("attention_loss_weight", 1.0)),
            "mask_loss_weight": float(config.get("mask_loss_weight", 0.5)),
            "answer_loss_weight": float(config.get("answer_loss_weight", 0.0)),
            "entropy_loss_weight": float(config.get("entropy_loss_weight", 0.0)),
        }
    modes = config.get("supervision_modes", {})
    if supervision not in modes:
        raise ValueError(
            f"Unknown supervision mode {supervision!r}; choose full or {sorted(modes)}"
        )
    selected = modes[supervision]
    return {
        "attention_supervision": bool(selected["attention_supervision"]),
        "attention_loss_weight": 0.0,
        "mask_loss_weight": float(selected["mask_loss_weight"]),
        "answer_loss_weight": float(selected["answer_loss_weight"]),
        "entropy_loss_weight": float(config.get("entropy_loss_weight", 0.0)),
    }


def train(
    config_path: Path,
    data_root: Path,
    output_root: Path,
    length: int,
    supervision: str = "full",
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if length not in map(int, config["lengths"]):
        raise ValueError(f"Length {length} is not configured")
    if int(config["heads"]) != 12:
        raise ValueError(
            "The one-bit encrypted update requires exactly 12 attention heads"
        )
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_bits = seed_bits_for_length(config, length)
    graph = fixed_graph(seed_bits, length, int(config["graph_seed"]))
    model = GoldreichToyTransformer(
        length,
        seed_bits,
        graph,
        int(config["key_dim_by_length"][str(length)]),
        int(config["gate_hidden_dim"]),
    ).to(device)
    train_data = load_data(data_root / f"n{length}" / "train.jsonl.gz")
    validation_data = load_data(data_root / f"n{length}" / "validation.jsonl.gz")
    test_data = load_data(data_root / f"n{length}" / "test.jsonl.gz")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    settings = supervision_settings(config, supervision)
    gate_stage = train_gates(model, config, device)
    if settings["attention_supervision"]:
        routing_stage = train_routing(model, config, device)
    else:
        routing_stage = {
            "enabled": False,
            "reason": "direct attention-position supervision removed",
            "initial_metrics": routing_metrics(model, device),
        }
    selection_n = min(
        int(config.get("selection_validation_n", 256)), len(validation_data["ids"])
    )
    selection_data = {
        key: value[:selection_n] if isinstance(value, (list, torch.Tensor)) else value
        for key, value in validation_data.items()
    }
    integration_stage = train_integration(
        model, train_data, selection_data, config, device, supervision, settings
    )
    final_gates = gate_metrics(model, device)
    final_routing = routing_metrics(model, device)
    validation_hard = evaluate(model, validation_data, device, hard=True)
    test_hard = evaluate(model, test_data, device, hard=True)
    soft_n = min(
        int(config["soft_validation_n_by_length"][str(length)]),
        len(validation_data["ids"]),
    )
    soft_data = {
        key: value[:soft_n] if isinstance(value, (list, torch.Tensor)) else value
        for key, value in validation_data.items()
    }
    validation_soft = evaluate(model, soft_data, device, hard=False)
    success = passes(
        test_hard,
        final_gates,
        final_routing,
        config["success_gate"],
        require_routing=bool(settings["attention_supervision"]),
    )
    result = {
        "schema_version": 2,
        "experiment_name": config["experiment_name"],
        "supervision": supervision,
        "supervision_settings": settings,
        "seed": config["seed"],
        "length": length,
        "seed_bits": seed_bits,
        "mask_bits": length,
        "device": str(device),
        "architecture": model.architecture_metadata(),
        "train_examples": len(train_data["ids"]),
        "validation_examples": len(validation_data["ids"]),
        "test_examples": len(test_data["ids"]),
        "seed_split": "all validation and test PRG seeds are unseen during training",
        "gate_stage": gate_stage,
        "routing_stage": routing_stage,
        "integration_stage": integration_stage,
        "final_gate_metrics": final_gates,
        "final_routing_metrics": final_routing,
        "validation_hard_attention": validation_hard,
        "validation_soft_attention": validation_soft,
        "test_hard_attention": test_hard,
        "passes_gate": bool(success),
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": model.architecture_metadata(),
            "config": config,
        },
        output_root / "model.pt",
    )
    (output_root / "metrics.json").write_text(
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
        default=Path("generated_data/parity_goldreich_toy_transformer/seed_0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/parity_goldreich_toy_transformer"),
    )
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--supervision", default="full")
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="Write negative ablation results without exiting nonzero",
    )
    args = parser.parse_args()
    result = train(
        args.config, args.data_root, args.output_root, args.length, args.supervision
    )
    print(json.dumps(result, indent=2))
    if not result["passes_gate"] and not args.allow_failed_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
