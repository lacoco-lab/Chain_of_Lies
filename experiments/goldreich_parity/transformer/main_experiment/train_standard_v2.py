#!/usr/bin/env python3
"""Reliable staged training for the agreed standard encrypted-PARITY Transformer.

The architecture is defined entirely by ``StandardGoldreichTransformer`` and is unchanged here.
All additional objects in this file are losses or evaluations used only during training.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_data import fixed_graph, seed_bits_for_length
from standard_model import COMPUTATION_HEADS, StandardGoldreichTransformer
from train import load_data
from train_standard import (
    evaluate,
    make_training_batch,
    route_loss_and_metrics,
    route_parameters,
    scaled_batch,
    select_rows,
    set_seed,
    task_passes,
    train_routing,
)


def public_targets(bits: torch.Tensor) -> tuple[torch.Tensor, ...]:
    previous = bits[:, 2] ^ bits[:, 3] ^ bits[:, 4] ^ (bits[:, 5] & bits[:, 6])
    current = bits[:, 7] ^ bits[:, 8] ^ bits[:, 9] ^ (bits[:, 10] & bits[:, 11])
    delta = previous ^ current
    output = bits[:, 0] ^ bits[:, 1] ^ delta
    return previous, current, delta, output


def random_public_batch(
    batch_size: int, length: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(2, (batch_size, COMPUTATION_HEADS), generator=generator)
    steps = torch.randint(length + 1, (batch_size,), generator=generator)
    return bits.to(device), steps.to(device)


def exhaustive_public_batches(
    model: StandardGoldreichTransformer,
    repeats: int,
    batch_size: int,
    device: torch.device,
):
    integers = torch.arange(1 << COMPUTATION_HEADS)
    shifts = torch.arange(COMPUTATION_HEADS - 1, -1, -1)
    base_bits = ((integers[:, None] >> shifts) & 1).long()
    for repeat in range(repeats):
        steps = (integers + repeat * 997).remainder(model.length + 1)
        for start in range(0, len(integers), batch_size):
            stop = start + batch_size
            yield base_bits[start:stop].to(device), steps[start:stop].to(device)


@torch.inference_mode()
def evaluate_public_mechanism(
    model: StandardGoldreichTransformer,
    repeats: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    correct = {"previous_mask": 0, "current_mask": 0, "mask_delta": 0, "output": 0}
    total = 0
    model.eval()
    for bits, steps in exhaustive_public_batches(model, repeats, batch_size, device):
        previous, current, delta, output = public_targets(bits)
        residual = model.canonical_residual(bits, steps)
        predictions = model.finish_from_residual(residual)
        for name, logits, target in (
            ("previous_mask", predictions["previous_mask_logits"], previous),
            ("current_mask", predictions["current_mask_logits"], current),
            ("mask_delta", predictions["mask_delta_logits"], delta),
            ("output", predictions["output_logits"], output),
        ):
            correct[name] += int((logits.argmax(-1) == target).sum())
        total += len(bits)
    return {f"{name}_accuracy": value / total for name, value in correct.items()} | {
        "examples": total
    }


def representation_parameters(
    model: StandardGoldreichTransformer,
) -> list[torch.nn.Parameter]:
    return [*model.v_proj.parameters(), *model.out_proj.parameters()]


def boolean_parameters(model: StandardGoldreichTransformer) -> list[torch.nn.Parameter]:
    return [
        *model.ff_in.parameters(),
        *model.ff_out.parameters(),
        *model.output_head.parameters(),
        *model.previous_mask_head.parameters(),
        *model.current_mask_head.parameters(),
        *model.mask_delta_head.parameters(),
    ]


@torch.inference_mode()
def evaluate_canonical_representation(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate routed-bit coordinates on held-out rows at every recurrent step."""
    correct = 0
    bit_squared_error = 0.0
    full_squared_error = 0.0
    bit_count = 0
    full_count = 0
    model.eval()
    for start in range(0, model.length + 1, batch_size):
        stop = min(start + batch_size, model.length + 1)
        steps = torch.arange(start, stop, device=device)
        rows = torch.arange(start, stop).remainder(len(data["ids"]))
        batch = select_rows(data, rows)
        seeds = batch["seeds"].to(device)
        inputs = batch["inputs"].to(device)
        prefixes = batch["states"].to(device).long()
        token_ids, valid = model.build_training_sequence(seeds, inputs, prefixes, steps)
        outputs = model(token_ids, valid, steps)
        source_bits = token_ids.gather(1, model.computation_targets(steps))
        target = model.canonical_residual(source_bits, steps)
        actual_bits = outputs["attention_residual"][:, :COMPUTATION_HEADS]
        target_bits = target[:, :COMPUTATION_HEADS]
        correct += int(((actual_bits > 0).long() == source_bits).sum())
        bit_squared_error += float((actual_bits - target_bits).square().sum())
        full_squared_error += float(
            (outputs["attention_residual"] - target).square().sum()
        )
        bit_count += actual_bits.numel()
        full_count += target.numel()
    return {
        "routed_bit_accuracy": correct / bit_count,
        "bit_mse": bit_squared_error / bit_count,
        "full_mse": full_squared_error / full_count,
        "evaluated_steps": model.length + 1,
    }


def train_canonical_representation(
    model: StandardGoldreichTransformer,
    data: dict[str, Any],
    validation_data: dict[str, Any],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Make the attention residual expose each routed bit in one fixed coordinate."""
    optimizer = torch.optim.AdamW(
        representation_parameters(model),
        lr=float(config["representation_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 20_000 + model.length + model.model_dim
    )
    maximum = int(config["representation_steps_by_length"][str(model.length)])
    threshold = float(config["representation_mse_gate"])
    interval = int(config["representation_gate_interval"])
    nuisance_weight = float(config["representation_nuisance_loss_weight"])
    history = []
    best_state = None
    best = (0.0, float("inf"))
    for iteration in range(1, maximum + 1):
        token_ids, valid, steps, _, _, _ = make_training_batch(
            data, model, iteration, batch_size, generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(token_ids, valid, steps)
        source_bits = token_ids.gather(1, model.computation_targets(steps))
        target = model.canonical_residual(source_bits, steps)
        bit_loss = F.mse_loss(
            outputs["attention_residual"][:, :COMPUTATION_HEADS],
            target[:, :COMPUTATION_HEADS],
        )
        nuisance_loss = F.mse_loss(
            outputs["attention_residual"][:, COMPUTATION_HEADS:],
            target[:, COMPUTATION_HEADS:],
        )
        loss = bit_loss + nuisance_weight * nuisance_loss
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % interval == 0 or iteration == maximum:
            validation = evaluate_canonical_representation(
                model, validation_data, max(1, batch_size), device
            )
            bit_accuracy = validation["routed_bit_accuracy"]
            mse = validation["bit_mse"]
            score = (bit_accuracy, -mse)
            if score > (best[0], -best[1]):
                best = (bit_accuracy, mse)
                best_state = copy.deepcopy(model.state_dict())
            _, route = route_loss_and_metrics(model, outputs, steps)
            record = {
                "step": iteration,
                "training_loss": float(loss.detach()),
                **validation,
                **route,
            }
            history.append(record)
            print(
                json.dumps({"event": "standard_v2_representation", **record}),
                flush=True,
            )
            if iteration >= 200 and bit_accuracy == 1.0 and mse <= threshold:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best[0] < 1.0 or best[1] > threshold:
        raise RuntimeError(
            f"Canonical representation gate failed: accuracy={best[0]}, mse={best[1]}"
        )
    return {
        "steps": iteration,
        "best_routed_bit_accuracy": best[0],
        "best_mse": best[1],
        "history": history,
    }


def _train_public_mechanism_attempt(
    model: StandardGoldreichTransformer,
    config: dict[str, Any],
    device: torch.device,
    attempt: int,
) -> dict[str, Any]:
    """Train and exhaustively gate the one ReLU FFN on the public local computation."""
    optimizer = torch.optim.AdamW(
        boolean_parameters(model),
        lr=float(config["boolean_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 30_000 + model.length + model.model_dim + attempt * 10_000
    )
    batch_size = int(config["boolean_batch_size"])
    eval_batch = int(config["boolean_evaluation_batch_size"])
    repeats = int(config["boolean_evaluation_repeats"])
    feature_steps = int(config["boolean_feature_steps"])
    maximum = int(config["boolean_output_max_steps"])
    history = []

    def supervised_residual(bits: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        residual = model.canonical_residual(bits, steps)
        noise_scale = float(config["boolean_nuisance_noise_std"])
        if noise_scale:
            residual = residual.clone()
            residual[:, COMPUTATION_HEADS:] += noise_scale * torch.randn_like(
                residual[:, COMPUTATION_HEADS:]
            )
        return residual

    # First expose both Goldreich predicates and their XOR in the shared FFN representation.
    for iteration in range(1, feature_steps + 1):
        bits, steps = random_public_batch(batch_size, model.length, generator, device)
        previous, current, delta, _ = public_targets(bits)
        predictions = model.finish_from_residual(supervised_residual(bits, steps))
        loss = (
            F.cross_entropy(predictions["previous_mask_logits"], previous)
            + F.cross_entropy(predictions["current_mask_logits"], current)
            + F.cross_entropy(predictions["mask_delta_logits"], delta)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    feature_gate = evaluate_public_mechanism(model, repeats, eval_batch, device)
    # The two Goldreich predicates are the supervised public features. Their XOR is reported as
    # a diagnostic, but making it a prerequisite recreates the very composition bottleneck that
    # this next stage is designed to solve.
    if (
        min(
            feature_gate[f"{name}_accuracy"]
            for name in ("previous_mask", "current_mask")
        )
        < 0.999
    ):
        raise RuntimeError(f"Public feature gate failed: {feature_gate}")

    # Then prioritize the actual encrypted update. A small replay loss retains public features.
    best_accuracy = -1.0
    best_state = None
    for iteration in range(1, maximum + 1):
        if iteration in set(map(int, config["boolean_lr_decay_steps"])):
            for group in optimizer.param_groups:
                group["lr"] *= float(config["boolean_lr_decay_factor"])
        bits, steps = random_public_batch(batch_size, model.length, generator, device)
        previous, current, _, output = public_targets(bits)
        predictions = model.finish_from_residual(supervised_residual(bits, steps))
        output_loss = F.cross_entropy(predictions["output_logits"], output)
        retention = F.cross_entropy(
            predictions["previous_mask_logits"], previous
        ) + F.cross_entropy(predictions["current_mask_logits"], current)
        loss = output_loss + float(config["boolean_retention_weight"]) * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % int(config["boolean_gate_interval"]) == 0:
            gate = evaluate_public_mechanism(model, repeats, eval_batch, device)
            record = {
                "step": iteration,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": float(loss.detach()),
                **gate,
            }
            history.append(record)
            print(
                json.dumps({"event": "standard_v2_public_gate", **record}), flush=True
            )
            if gate["output_accuracy"] > best_accuracy:
                best_accuracy = gate["output_accuracy"]
                best_state = copy.deepcopy(model.state_dict())
            if gate["output_accuracy"] == 1.0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_accuracy < 1.0:
        raise RuntimeError(f"Complete public update gate failed: best={best_accuracy}")
    return {
        "attempt": attempt,
        "feature_gate": feature_gate,
        "output_steps": iteration,
        "best_output_accuracy": best_accuracy,
        "history": history,
    }


def train_public_mechanism(
    model: StandardGoldreichTransformer, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Retry only the public optimizer from clean deterministic initializations."""
    representation_state = copy.deepcopy(model.state_dict())
    failures = []
    for attempt in range(int(config["boolean_restarts"])):
        model.load_state_dict(representation_state)
        if attempt:
            set_seed(int(config["seed"]) + 50_000 + model.model_dim + attempt)
            for module in (
                model.ff_in,
                model.ff_out,
                model.output_head,
                model.previous_mask_head,
                model.current_mask_head,
                model.mask_delta_head,
            ):
                module.reset_parameters()
        try:
            result = _train_public_mechanism_attempt(model, config, device, attempt)
            result["failed_attempts"] = failures
            return result
        except RuntimeError as error:
            failures.append({"attempt": attempt, "error": str(error)})
            print(
                json.dumps({"event": "standard_v2_public_restart", **failures[-1]}),
                flush=True,
            )
    raise RuntimeError(f"All public-mechanism attempts failed: {failures}")


def train_real_integration(
    model: StandardGoldreichTransformer,
    train_data: dict[str, Any],
    validation_data: dict[str, Any],
    config: dict[str, Any],
    batch_size: int,
    eval_batch: int,
    device: torch.device,
) -> dict[str, Any]:
    """Calibrate on genuine trajectories while replaying the exact public update."""
    optimizer = torch.optim.AdamW(
        boolean_parameters(model),
        lr=float(config["integration_learning_rate"]),
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 40_000 + model.length + model.model_dim
    )
    maximum = int(config["integration_steps_by_length"][str(model.length)])
    interval = int(config["integration_gate_interval"])
    subset_n = min(
        int(config["integration_validation_n_by_length"][str(model.length)]),
        len(validation_data["ids"]),
    )
    subset = select_rows(validation_data, torch.arange(subset_n))
    history = []
    best_score = (-1.0, -1.0, -1.0)
    best_state = copy.deepcopy(model.state_dict())

    for iteration in range(0, maximum + 1):
        if iteration % interval == 0:
            metrics = evaluate(model, subset, device, eval_batch)
            score = (
                metrics["joint_exact"],
                metrics["trace_exact"],
                metrics["state_bit_accuracy"],
            )
            record = {"step": iteration, **metrics}
            history.append(record)
            print(
                json.dumps({"event": "standard_v2_integration_gate", **record}),
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            if metrics["joint_exact"] == 1.0:
                break
        if iteration == maximum:
            break
        token_ids, valid, steps, targets, _, _ = make_training_batch(
            train_data, model, iteration + 1, batch_size, generator, device
        )
        outputs = model(token_ids, valid, steps)
        real_loss = F.cross_entropy(outputs["output_logits"], targets)
        public_bits, public_steps = random_public_batch(
            int(config["boolean_batch_size"]), model.length, generator, device
        )
        *_, public_output = public_targets(public_bits)
        public_predictions = model.finish_from_residual(
            model.canonical_residual(public_bits, public_steps)
        )
        replay_loss = F.cross_entropy(
            public_predictions["output_logits"], public_output
        )
        loss = (
            real_loss + float(config["integration_public_replay_weight"]) * replay_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_state)
    if best_score[0] < float(config["integration_minimum_joint_gate"]):
        raise RuntimeError(
            f"Real-trajectory integration gate failed: best={best_score}"
        )
    return {"steps": iteration, "best_score": best_score, "history": history}


def write_stage(
    output_root: Path,
    name: str,
    value: dict[str, Any],
    model: StandardGoldreichTransformer | None = None,
) -> None:
    """Persist both the stage report and the best model left by that stage."""
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stages"
    stage_root.mkdir(exist_ok=True)
    (stage_root / f"{name}.json").write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )
    if model is not None:
        checkpoint_root = output_root / "stage_checkpoints"
        checkpoint_root.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_root / f"{name}.pt")
    (output_root / "progress.json").write_text(
        json.dumps({"completed_stage": name, name: value}, indent=2) + "\n",
        encoding="utf-8",
    )


def run_stage(
    name: str, function, *, continue_after_gate_failure: bool
) -> tuple[dict[str, Any], str | None]:
    """Run a stage, optionally treating its optimization gate as diagnostic.

    Every stage restores or retains its best model state before raising a gate
    error.  Continuing therefore starts the next stage from that best state;
    it does not silently reset or bypass any computation.
    """
    try:
        value = function()
        return {"passed_stage_gate": True, **value}, None
    except RuntimeError as error:
        if not continue_after_gate_failure:
            raise
        warning = str(error)
        print(
            json.dumps(
                {
                    "event": "nonfatal_stage_gate",
                    "stage": name,
                    "warning": warning,
                }
            ),
            flush=True,
        )
        return {
            "passed_stage_gate": False,
            "continued_from_best_stage_state": True,
            "warning": warning,
        }, warning


def train(
    config_path: Path,
    data_root: Path,
    output_root: Path,
    length: int,
    model_dim: int,
    continue_after_stage_gate_failure: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if length not in map(int, config["lengths"]) or model_dim not in map(
        int, config["model_dims"]
    ):
        raise ValueError("Requested length or width is not configured")
    initialization_seed = int(config["seed"]) + model_dim
    set_seed(initialization_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_bits = seed_bits_for_length(config, length)
    model = StandardGoldreichTransformer(
        length=length,
        seed_bits=seed_bits,
        graph=fixed_graph(seed_bits, length, int(config["graph_seed"])),
        model_dim=model_dim,
        heads=int(config["heads"]),
        token_dim=int(config["token_embedding_dim"]),
        ff_dim=model_dim * int(config["ff_multiplier"]),
    ).to(device)
    train_data = load_data(data_root / f"n{length}" / "train.jsonl.gz")
    validation_data = load_data(data_root / f"n{length}" / "validation.jsonl.gz")
    test_data = load_data(data_root / f"n{length}" / "test.jsonl.gz")
    batch_size = scaled_batch(config, "batch_size_128_by_length", length, model_dim)
    eval_batch = scaled_batch(
        config, "evaluation_batch_size_128_by_length", length, model_dim
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    stage_gate_warnings: dict[str, str] = {}

    routing, warning = run_stage(
        "routing",
        lambda: train_routing(model, train_data, config, batch_size, device),
        continue_after_gate_failure=continue_after_stage_gate_failure,
    )
    if warning:
        stage_gate_warnings["routing"] = warning
    write_stage(output_root, "routing", routing, model)

    representation, warning = run_stage(
        "representation",
        lambda: train_canonical_representation(
            model, train_data, validation_data, config, batch_size, device
        ),
        continue_after_gate_failure=continue_after_stage_gate_failure,
    )
    if warning:
        stage_gate_warnings["representation"] = warning
    write_stage(output_root, "representation", representation, model)

    public, warning = run_stage(
        "public_mechanism",
        lambda: train_public_mechanism(model, config, device),
        continue_after_gate_failure=continue_after_stage_gate_failure,
    )
    if warning:
        stage_gate_warnings["public_mechanism"] = warning
    write_stage(output_root, "public_mechanism", public, model)

    integration, warning = run_stage(
        "integration",
        lambda: train_real_integration(
            model, train_data, validation_data, config, batch_size, eval_batch, device
        ),
        continue_after_gate_failure=continue_after_stage_gate_failure,
    )
    if warning:
        stage_gate_warnings["integration"] = warning
    write_stage(output_root, "integration", integration, model)
    validation = evaluate(model, validation_data, device, eval_batch)
    test = evaluate(model, test_data, device, eval_batch)
    result = {
        "schema_version": 2,
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
        "training_only_supervision": {
            "attention_routes": True,
            "canonical_routed_bit_coordinates": True,
            "public_predicates_and_complete_update": True,
            "used_at_inference": False,
        },
        "routing_training": routing,
        "representation_training": representation,
        "public_mechanism_training": public,
        "integration_training": integration,
        "stage_gates_are_diagnostic": continue_after_stage_gate_failure,
        "stage_gate_warnings": stage_gate_warnings,
        "validation_softmax_attention": validation,
        "test_softmax_attention": test,
        "passes_gate": task_passes(test, config["success_gate"]),
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
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument(
        "--continue-after-stage-gate-failure",
        action="store_true",
        help="Continue from each stage's best state and always perform final evaluation.",
    )
    parser.add_argument(
        "--allow-failed-final-gate",
        action="store_true",
        help="Exit successfully after writing valid final metrics below the success gate.",
    )
    args = parser.parse_args()
    result = train(
        args.config,
        args.data_root,
        args.output_root,
        args.length,
        args.model_dim,
        args.continue_after_stage_gate_failure,
    )
    if not result["passes_gate"] and not args.allow_failed_final_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
