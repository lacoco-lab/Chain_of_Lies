#!/usr/bin/env python3
"""Train and evaluate the standard one-layer encrypted-PARITY Transformer."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_data import fixed_graph, seed_bits_for_length
from standard_model import COMPUTATION_HEADS, StandardGoldreichTransformer
from train import load_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_rows(data: dict[str, Any], rows: torch.Tensor) -> dict[str, Any]:
    return {
        key: (
            value[rows]
            if isinstance(value, torch.Tensor)
            else [value[int(index)] for index in rows]
        )
        for key, value in data.items()
    }


def cyclic_steps(
    iteration: int, batch_size: int, length: int, device: torch.device
) -> torch.Tensor:
    start = ((iteration - 1) * batch_size) % (length + 1)
    return (torch.arange(batch_size, device=device) + start).remainder(length + 1)


def targets_for_steps(
    batch: dict[str, Any], steps: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(len(steps))
    cpu_steps = steps.cpu()
    output = torch.empty(len(steps), dtype=torch.long)
    previous_masks = torch.zeros(len(steps), dtype=torch.long)
    current_masks = torch.zeros(len(steps), dtype=torch.long)
    ordinary = cpu_steps < batch["states"].shape[1]
    if ordinary.any():
        selected_rows = rows[ordinary]
        selected_steps = cpu_steps[ordinary]
        output[ordinary] = batch["states"][selected_rows, selected_steps].long()
        current_masks[ordinary] = batch["masks"][selected_rows, selected_steps].long()
    final = ~ordinary
    if final.any():
        output[final] = batch["answers"][final].long()
    noninitial = cpu_steps > 0
    if noninitial.any():
        selected_rows = rows[noninitial]
        selected_steps = cpu_steps[noninitial] - 1
        previous_masks[noninitial] = batch["masks"][
            selected_rows, selected_steps
        ].long()
    return output.to(device), previous_masks.to(device), current_masks.to(device)


def make_training_batch(
    data: dict[str, Any],
    model: StandardGoldreichTransformer,
    iteration: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    rows = torch.randint(len(data["ids"]), (batch_size,), generator=generator)
    batch = select_rows(data, rows)
    steps = cyclic_steps(iteration, batch_size, model.length, device)
    seeds = batch["seeds"].to(device)
    inputs = batch["inputs"].to(device)
    prefixes = batch["states"].to(device).long()
    token_ids, valid = model.build_training_sequence(seeds, inputs, prefixes, steps)
    output, previous_masks, current_masks = targets_for_steps(batch, steps, device)
    return token_ids, valid, steps, output, previous_masks, current_masks


def route_loss_and_metrics(
    model: StandardGoldreichTransformer,
    outputs: dict[str, torch.Tensor],
    steps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["attention_logits"]
    targets = model.routing_targets(steps)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    probabilities = outputs["attention"]
    target_mass = probabilities.gather(-1, targets[..., None]).squeeze(-1)
    predicted = probabilities.argmax(dim=-1)
    return loss, {
        "routing_accuracy": float((predicted == targets).float().mean().detach()),
        "minimum_target_attention": float(target_mass.min().detach()),
        "mean_target_attention": float(target_mass.mean().detach()),
    }


def route_parameters(model: StandardGoldreichTransformer) -> list[torch.nn.Parameter]:
    return [
        *model.token_embedding.parameters(),
        *model.position_embedding.parameters(),
        *model.q_proj.parameters(),
        *model.k_proj.parameters(),
    ]


def joint_optimizer(
    model: StandardGoldreichTransformer, config: dict[str, Any]
) -> torch.optim.Optimizer:
    routing = route_parameters(model)
    routing_ids = {id(parameter) for parameter in routing}
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in routing_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": routing,
                "lr": float(config["integration_routing_learning_rate"]),
            },
            {
                "params": task_parameters,
                "lr": float(config["integration_learning_rate"]),
            },
        ],
        weight_decay=0.0,
    )


def task_only_optimizer(
    model: StandardGoldreichTransformer, config: dict[str, Any]
) -> torch.optim.Optimizer:
    routing_ids = {id(parameter) for parameter in route_parameters(model)}
    parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in routing_ids
    ]
    return torch.optim.AdamW(
        parameters, lr=float(config["integration_learning_rate"]), weight_decay=0.0
    )


def goldreich_from_columns(bits: torch.Tensor) -> torch.Tensor:
    return bits[:, 0] ^ bits[:, 1] ^ bits[:, 2] ^ (bits[:, 3] & bits[:, 4])


def curriculum_selection(
    previous_masks: torch.Tensor, current_masks: torch.Tensor, phase: str
) -> torch.Tensor:
    if phase == "both_masks_zero":
        return (previous_masks == 0) & (current_masks == 0)
    if phase == "current_mask_zero":
        return current_masks == 0
    raise ValueError(f"Unknown curriculum phase: {phase}")


def mechanism_losses(
    model: StandardGoldreichTransformer,
    outputs: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    steps: torch.Tensor,
    previous_masks: torch.Tensor,
    current_masks: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    route_loss, route = route_loss_and_metrics(model, outputs, steps)
    source_targets = token_ids.gather(1, model.computation_targets(steps))
    value_loss = F.cross_entropy(
        outputs["routed_value_logits"].reshape(-1, 2), source_targets.reshape(-1)
    )
    previous_loss = F.cross_entropy(outputs["previous_mask_logits"], previous_masks)
    current_loss = F.cross_entropy(outputs["current_mask_logits"], current_masks)
    delta_targets = previous_masks ^ current_masks
    delta_loss = F.cross_entropy(outputs["mask_delta_logits"], delta_targets)
    loss = (
        float(config["attention_loss_weight"]) * route_loss
        + float(config["value_loss_weight"]) * value_loss
        + float(config["mask_loss_weight"]) * (previous_loss + current_loss)
        + float(config["mask_delta_loss_weight"]) * delta_loss
    )
    parts = {
        "route_loss": route_loss,
        "value_loss": value_loss,
        "previous_mask_loss": previous_loss,
        "current_mask_loss": current_loss,
        "mask_delta_loss": delta_loss,
        "source_targets": source_targets,
        "delta_targets": delta_targets,
    }
    return loss, parts, route


def train_routing(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        route_parameters(model),
        lr=float(config["route_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + model.length + model.model_dim
    )
    maximum = int(config["route_steps_by_length"][str(model.length)])
    minimum_steps = max(50, 2 * ((model.length + batch_size) // batch_size))
    history = []
    best = (-1.0, -1.0)
    for iteration in range(1, maximum + 1):
        batch = make_training_batch(
            data, model, iteration, batch_size, generator, device
        )
        token_ids, valid, steps, _, _, _ = batch
        optimizer.zero_grad(set_to_none=True)
        outputs = model(token_ids, valid, steps)
        loss, metrics = route_loss_and_metrics(model, outputs, steps)
        loss.backward()
        optimizer.step()
        score = (metrics["routing_accuracy"], metrics["minimum_target_attention"])
        if score > best:
            best = score
        if iteration == 1 or iteration % 50 == 0 or iteration == maximum:
            record = {"step": iteration, "loss": float(loss.detach()), **metrics}
            history.append(record)
            print(
                json.dumps({"event": "standard_route_training", **record}), flush=True
            )
            if (
                iteration >= minimum_steps
                and metrics["routing_accuracy"] == 1.0
                and metrics["minimum_target_attention"]
                >= float(config["route_pretraining_minimum_attention"])
            ):
                break
    return {"steps": iteration, "best_batch_score": best, "history": history}


def train_public_truth_table(
    model: StandardGoldreichTransformer,
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Supervise the single standard FFN on the complete 12-bit local mechanism."""
    optimizer = task_only_optimizer(model, config)
    maximum = int(config["truth_table_steps_by_length"][str(model.length)])
    history = []
    for iteration in range(1, maximum + 1):
        integers = (
            torch.arange(batch_size, device=device) + (iteration - 1) * batch_size
        ).remainder(1 << COMPUTATION_HEADS)
        shifts = torch.arange(COMPUTATION_HEADS - 1, -1, -1, device=device)
        bits = ((integers[:, None] >> shifts) & 1).long()
        steps = cyclic_steps(iteration, batch_size, model.length, device)
        previous_masks = goldreich_from_columns(bits[:, 2:7])
        current_masks = goldreich_from_columns(bits[:, 7:12])
        delta = previous_masks ^ current_masks
        output_targets = bits[:, 0] ^ bits[:, 1] ^ delta

        optimizer.zero_grad(set_to_none=True)
        outputs = model.synthesized_mechanism(bits, steps)
        value_loss = F.cross_entropy(
            outputs["routed_value_logits"].reshape(-1, 2), bits.reshape(-1)
        )
        previous_loss = F.cross_entropy(outputs["previous_mask_logits"], previous_masks)
        current_loss = F.cross_entropy(outputs["current_mask_logits"], current_masks)
        delta_loss = F.cross_entropy(outputs["mask_delta_logits"], delta)
        output_loss = F.cross_entropy(outputs["output_logits"], output_targets)
        loss = (
            output_loss
            + float(config["value_loss_weight"]) * value_loss
            + float(config["mask_loss_weight"]) * (previous_loss + current_loss)
            + float(config["mask_delta_loss_weight"]) * delta_loss
        )
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == maximum:
            record = {
                "step": iteration,
                "loss": float(loss.detach()),
                "output_accuracy": float(
                    (outputs["output_logits"].argmax(-1) == output_targets)
                    .float()
                    .mean()
                    .detach()
                ),
                "routed_value_accuracy": float(
                    (outputs["routed_value_logits"].argmax(-1) == bits)
                    .float()
                    .mean()
                    .detach()
                ),
                "previous_mask_accuracy": float(
                    (outputs["previous_mask_logits"].argmax(-1) == previous_masks)
                    .float()
                    .mean()
                    .detach()
                ),
                "current_mask_accuracy": float(
                    (outputs["current_mask_logits"].argmax(-1) == current_masks)
                    .float()
                    .mean()
                    .detach()
                ),
                "mask_delta_accuracy": float(
                    (outputs["mask_delta_logits"].argmax(-1) == delta)
                    .float()
                    .mean()
                    .detach()
                ),
            }
            history.append(record)
            print(
                json.dumps({"event": "standard_public_truth_table", **record}),
                flush=True,
            )
    return {
        "steps": maximum,
        "truth_table_rows": 1 << COMPUTATION_HEADS,
        "history": history,
    }


def train_mechanism(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Teach routed bit values, both public masks, and their XOR before the full update."""
    optimizer = joint_optimizer(model, config)
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 5_000 + model.length + model.model_dim
    )
    maximum = int(config["mechanism_steps_by_length"][str(model.length)])
    history = []
    for iteration in range(1, maximum + 1):
        batch = make_training_batch(
            data, model, iteration, batch_size, generator, device
        )
        token_ids, valid, steps, _, previous_masks, current_masks = batch
        optimizer.zero_grad(set_to_none=True)
        outputs = model(token_ids, valid, steps)
        loss, parts, route = mechanism_losses(
            model, outputs, token_ids, steps, previous_masks, current_masks, config
        )
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == maximum:
            record = {
                "step": iteration,
                "loss": float(loss.detach()),
                "routed_value_accuracy": float(
                    (
                        outputs["routed_value_logits"].argmax(-1)
                        == parts["source_targets"]
                    )
                    .float()
                    .mean()
                    .detach()
                ),
                "previous_mask_accuracy": float(
                    (outputs["previous_mask_logits"].argmax(-1) == previous_masks)
                    .float()
                    .mean()
                    .detach()
                ),
                "current_mask_accuracy": float(
                    (outputs["current_mask_logits"].argmax(-1) == current_masks)
                    .float()
                    .mean()
                    .detach()
                ),
                "mask_delta_accuracy": float(
                    (outputs["mask_delta_logits"].argmax(-1) == parts["delta_targets"])
                    .float()
                    .mean()
                    .detach()
                ),
                **route,
            }
            history.append(record)
            print(
                json.dumps({"event": "standard_mechanism_training", **record}),
                flush=True,
            )
    return {"steps": maximum, "history": history}


def train_composition_curriculum(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Teach the single MLP to add update terms gradually, retaining all mechanisms."""
    optimizer = joint_optimizer(model, config)
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 7_500 + model.length + model.model_dim
    )
    steps_per_phase = int(config["composition_steps_by_length"][str(model.length)])
    history = []
    global_step = 0
    for phase in ("both_masks_zero", "current_mask_zero"):
        for phase_step in range(1, steps_per_phase + 1):
            global_step += 1
            sampling_attempts = 0
            while True:
                sampling_attempts += 1
                batch = make_training_batch(
                    data,
                    model,
                    global_step + sampling_attempts - 1,
                    batch_size,
                    generator,
                    device,
                )
                token_ids, valid, steps, true_targets, previous_masks, current_masks = (
                    batch
                )
                selected = curriculum_selection(previous_masks, current_masks, phase)
                if bool(selected.any()):
                    break
            optimizer.zero_grad(set_to_none=True)
            outputs = model(token_ids, valid, steps)
            mechanism_loss, parts, route = mechanism_losses(
                model, outputs, token_ids, steps, previous_masks, current_masks, config
            )
            output_loss = F.cross_entropy(
                outputs["output_logits"][selected], true_targets[selected]
            )
            loss = output_loss + mechanism_loss
            loss.backward()
            optimizer.step()
            if (
                phase_step == 1
                or phase_step % 100 == 0
                or phase_step == steps_per_phase
            ):
                record = {
                    "phase": phase,
                    "phase_step": phase_step,
                    "sampling_attempts": sampling_attempts,
                    "eligible_examples": int(selected.sum()),
                    "loss": float(loss.detach()),
                    "output_accuracy": float(
                        (
                            outputs["output_logits"][selected].argmax(-1)
                            == true_targets[selected]
                        )
                        .float()
                        .mean()
                        .detach()
                    ),
                    "routed_value_accuracy": float(
                        (
                            outputs["routed_value_logits"].argmax(-1)
                            == parts["source_targets"]
                        )
                        .float()
                        .mean()
                        .detach()
                    ),
                    "previous_mask_accuracy": float(
                        (outputs["previous_mask_logits"].argmax(-1) == previous_masks)
                        .float()
                        .mean()
                        .detach()
                    ),
                    "current_mask_accuracy": float(
                        (outputs["current_mask_logits"].argmax(-1) == current_masks)
                        .float()
                        .mean()
                        .detach()
                    ),
                    "mask_delta_accuracy": float(
                        (
                            outputs["mask_delta_logits"].argmax(-1)
                            == parts["delta_targets"]
                        )
                        .float()
                        .mean()
                        .detach()
                    ),
                    **route,
                }
                history.append(record)
                print(
                    json.dumps({"event": "standard_composition_curriculum", **record}),
                    flush=True,
                )
    return {"steps_per_phase": steps_per_phase, "history": history}


def train_integration(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    # The full standard block co-adapts. A smaller LR plus route loss protects learned routing.
    optimizer = joint_optimizer(model, config)
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 10_000 + model.length + model.model_dim
    )
    maximum = int(config["integration_steps_by_length"][str(model.length)])
    history = []
    best_loss = float("inf")
    for iteration in range(1, maximum + 1):
        batch = make_training_batch(
            data, model, iteration, batch_size, generator, device
        )
        token_ids, valid, steps, targets, previous_masks, current_masks = batch
        optimizer.zero_grad(set_to_none=True)
        outputs = model(token_ids, valid, steps)
        output_loss = F.cross_entropy(outputs["output_logits"], targets)
        mechanism_loss, parts, route = mechanism_losses(
            model, outputs, token_ids, steps, previous_masks, current_masks, config
        )
        loss = output_loss + mechanism_loss
        loss.backward()
        optimizer.step()
        detached_loss = float(loss.detach())
        if detached_loss < best_loss:
            best_loss = detached_loss
        if iteration == 1 or iteration % 100 == 0 or iteration == maximum:
            record = {
                "step": iteration,
                "loss": detached_loss,
                "output_accuracy": float(
                    (outputs["output_logits"].argmax(-1) == targets)
                    .float()
                    .mean()
                    .detach()
                ),
                "previous_mask_accuracy": float(
                    (outputs["previous_mask_logits"].argmax(-1) == previous_masks)
                    .float()
                    .mean()
                    .detach()
                ),
                "current_mask_accuracy": float(
                    (outputs["current_mask_logits"].argmax(-1) == current_masks)
                    .float()
                    .mean()
                    .detach()
                ),
                "routed_value_accuracy": float(
                    (
                        outputs["routed_value_logits"].argmax(-1)
                        == parts["source_targets"]
                    )
                    .float()
                    .mean()
                    .detach()
                ),
                "mask_delta_accuracy": float(
                    (outputs["mask_delta_logits"].argmax(-1) == parts["delta_targets"])
                    .float()
                    .mean()
                    .detach()
                ),
                "route_loss": float(parts["route_loss"].detach()),
                **route,
            }
            history.append(record)
            print(
                json.dumps({"event": "standard_integration_training", **record}),
                flush=True,
            )
    return {"steps": maximum, "best_training_loss": best_loss, "history": history}


@torch.inference_mode()
def validate_gold_batch(
    model: StandardGoldreichTransformer,
    seeds: torch.Tensor,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    states: torch.Tensor,
    answers: torch.Tensor,
) -> None:
    graph = torch.tensor(model.graph, dtype=torch.long, device=seeds.device)
    chosen = seeds[:, graph]
    expected_masks = (
        chosen[..., 0]
        ^ chosen[..., 1]
        ^ chosen[..., 2]
        ^ (chosen[..., 3] & chosen[..., 4])
    ).long()
    plaintext = inputs.long().cumsum(dim=1).remainder(2)
    if not torch.equal(expected_masks, masks.long()):
        raise RuntimeError("Invalid masks in evaluation data")
    if not torch.equal(plaintext ^ expected_masks, states.long()):
        raise RuntimeError("Invalid encrypted states in evaluation data")
    if not torch.equal(plaintext[:, -1], answers.long()):
        raise RuntimeError("Invalid PARITY labels in evaluation data")


@torch.inference_mode()
def evaluate(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    total = len(data["ids"])
    bit_correct = trace_correct = answer_correct = joint_correct = 0
    route_correct = route_total = 0
    minimum_target_attention = 1.0
    per_step_correct = torch.zeros(model.length, dtype=torch.long)
    samples = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        seeds = data["seeds"][start:end].to(device)
        inputs = data["inputs"][start:end].to(device)
        masks = data["masks"][start:end].to(device)
        states = data["states"][start:end].to(device)
        answers = data["answers"][start:end].to(device)
        validate_gold_batch(model, seeds, inputs, masks, states, answers)
        cache = model.initialize_decode_cache(seeds, inputs)
        predictions = []
        for step in range(model.length + 1):
            outputs = model.decode_step(cache, step)
            target_positions = model.routing_targets(
                torch.full((end - start,), step, dtype=torch.long, device=device)
            )
            routed = outputs["attention"]
            target_mass = routed.gather(-1, target_positions[..., None]).squeeze(-1)
            route_correct += int((routed.argmax(-1) == target_positions).sum())
            route_total += target_positions.numel()
            minimum_target_attention = min(
                minimum_target_attention, float(target_mass.min())
            )
            predicted = outputs["output_logits"].argmax(-1)
            if step < model.length:
                predictions.append(predicted)
                model.append_prediction(cache, step, predicted)
            else:
                predicted_answer = predicted
        predicted_states = torch.stack(predictions, dim=1)
        correct = predicted_states == states
        exact = correct.all(dim=1)
        answer_ok = predicted_answer == answers
        bit_correct += int(correct.sum())
        per_step_correct += correct.sum(dim=0).cpu()
        trace_correct += int(exact.sum())
        answer_correct += int(answer_ok.sum())
        joint_correct += int((exact & answer_ok).sum())
        if len(samples) < 8:
            for offset in range(min(8 - len(samples), end - start)):
                samples.append(
                    {
                        "id": data["ids"][start + offset],
                        "expected_states": "".join(
                            map(str, states[offset].cpu().tolist())
                        ),
                        "predicted_states": "".join(
                            map(str, predicted_states[offset].cpu().tolist())
                        ),
                        "expected_answer": int(answers[offset]),
                        "predicted_answer": int(predicted_answer[offset]),
                    }
                )
    return {
        "n": total,
        "state_bit_accuracy": bit_correct / (total * model.length),
        "trace_exact": trace_correct / total,
        "final_parity_accuracy": answer_correct / total,
        "joint_exact": joint_correct / total,
        "routing_accuracy": route_correct / route_total,
        "minimum_target_attention": minimum_target_attention,
        "per_step_accuracy": [int(value) / total for value in per_step_correct],
        "samples": samples,
    }


def task_passes(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        metrics["state_bit_accuracy"] >= float(gate["minimum_state_bit_accuracy"])
        and metrics["trace_exact"] >= float(gate["minimum_trace_exact"])
        and metrics["final_parity_accuracy"]
        >= float(gate["minimum_final_parity_accuracy"])
        and metrics["joint_exact"] >= float(gate["minimum_joint_exact"])
    )


def scaled_batch(config: dict[str, Any], key: str, length: int, model_dim: int) -> int:
    base = int(config[key][str(length)])
    return base if model_dim == 128 else max(4, base // 2)


def train(
    config_path: Path, data_root: Path, output_root: Path, length: int, model_dim: int
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if length not in map(int, config["lengths"]):
        raise ValueError(f"Length {length} is not configured")
    if model_dim not in map(int, config["model_dims"]):
        raise ValueError(f"Width {model_dim} is not configured")
    initialization_seed = int(config["seed"]) + model_dim
    set_seed(initialization_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_bits = seed_bits_for_length(config, length)
    graph = fixed_graph(seed_bits, length, int(config["graph_seed"]))
    model = StandardGoldreichTransformer(
        length=length,
        seed_bits=seed_bits,
        graph=graph,
        model_dim=model_dim,
        heads=int(config["heads"]),
        token_dim=int(config["token_embedding_dim"]),
        ff_dim=model_dim * int(config["ff_multiplier"]),
    ).to(device)
    train_data = load_data(data_root / f"n{length}" / "train.jsonl.gz")
    validation_data = load_data(data_root / f"n{length}" / "validation.jsonl.gz")
    test_data = load_data(data_root / f"n{length}" / "test.jsonl.gz")
    expected = {"seeds": seed_bits, "inputs": length, "masks": length, "states": length}
    observed = {key: int(train_data[key].shape[1]) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"Training data dimensions mismatch: {observed} != {expected}"
        )
    batch_size = scaled_batch(config, "batch_size_128_by_length", length, model_dim)
    eval_batch = scaled_batch(
        config, "evaluation_batch_size_128_by_length", length, model_dim
    )
    started = time.monotonic()
    routing = train_routing(model, train_data, config, batch_size, device)
    truth_table = train_public_truth_table(model, config, batch_size, device)
    mechanism = train_mechanism(model, train_data, config, batch_size, device)
    composition = train_composition_curriculum(
        model, train_data, config, batch_size, device
    )
    integration = train_integration(model, train_data, config, batch_size, device)
    validation = evaluate(model, validation_data, device, eval_batch)
    test = evaluate(model, test_data, device, eval_batch)
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "parameter_initialization_seed": initialization_seed,
        "length": length,
        "seed_bits": seed_bits,
        "model_dim": model_dim,
        "device": str(device),
        "train_examples": len(train_data["ids"]),
        "validation_examples": len(validation_data["ids"]),
        "test_examples": len(test_data["ids"]),
        "seed_split": "all validation and test PRG seeds are unseen during training",
        "architecture": model.architecture_metadata(),
        "routing_training": routing,
        "public_truth_table_training": truth_table,
        "mechanism_training": mechanism,
        "composition_curriculum": composition,
        "integration_training": integration,
        "validation_softmax_attention": validation,
        "test_softmax_attention": test,
        "passes_gate": task_passes(test, config["success_gate"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
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
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument("--allow-failed-gate", action="store_true")
    args = parser.parse_args()
    result = train(
        args.config, args.data_root, args.output_root, args.length, args.model_dim
    )
    if not result["passes_gate"] and not args.allow_failed_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
