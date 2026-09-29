#!/usr/bin/env python3
"""Supervision ablations for the final standard encrypted-PARITY Transformer."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1] / "main_experiment"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from generate_data import fixed_graph, seed_bits_for_length  # noqa: E402
from standard_model import COMPUTATION_HEADS, StandardGoldreichTransformer  # noqa: E402
from train import load_data  # noqa: E402
from train_standard import (
    evaluate,
    make_training_batch,
    route_loss_and_metrics,  # noqa: E402
    select_rows,
    set_seed,
    task_passes,
)
from train_standard_v2 import public_targets, train_public_mechanism  # noqa: E402

MODES = ("no_route", "end_to_end")


def all_parameters(model: StandardGoldreichTransformer) -> list[torch.nn.Parameter]:
    return list(model.parameters())


def losses(
    model: StandardGoldreichTransformer,
    outputs: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    steps: torch.Tensor,
    output_targets: torch.Tensor,
    previous_masks: torch.Tensor,
    current_masks: torch.Tensor,
    mode: str,
    config: dict[str, Any],
    include_output: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    output_loss = F.cross_entropy(outputs["output_logits"], output_targets)
    if mode == "end_to_end":
        return output_loss, {"output_ce": float(output_loss.detach())}

    source_bits = token_ids.gather(1, model.computation_targets(steps))
    canonical = model.canonical_residual(source_bits, steps)
    canonical_loss = F.mse_loss(
        outputs["attention_residual"][:, :COMPUTATION_HEADS],
        canonical[:, :COMPUTATION_HEADS],
    )
    value_loss = F.cross_entropy(
        outputs["routed_value_logits"].reshape(-1, 2), source_bits.reshape(-1)
    )
    previous_loss = F.cross_entropy(outputs["previous_mask_logits"], previous_masks)
    current_loss = F.cross_entropy(outputs["current_mask_logits"], current_masks)
    delta = previous_masks ^ current_masks
    delta_loss = F.cross_entropy(outputs["mask_delta_logits"], delta)
    total = (
        float(config["canonical_loss_weight"]) * canonical_loss
        + float(config["value_loss_weight"]) * value_loss
        + float(config["mask_loss_weight"]) * (previous_loss + current_loss)
        + float(config["mask_delta_loss_weight"]) * delta_loss
    )
    if include_output:
        total = total + output_loss
    return total, {
        "output_ce": float(output_loss.detach()),
        "canonical_mse": float(canonical_loss.detach()),
        "value_ce": float(value_loss.detach()),
        "previous_mask_ce": float(previous_loss.detach()),
        "current_mask_ce": float(current_loss.detach()),
        "mask_delta_ce": float(delta_loss.detach()),
    }


def train_phase(
    model: StandardGoldreichTransformer,
    train_data: dict[str, Any],
    validation_data: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    phase: str,
    maximum: int,
    iteration_offset: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train every model parameter without ever using a route-position loss."""
    optimizer = torch.optim.AdamW(
        all_parameters(model), lr=float(config["learning_rate"]), weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(
        int(config["seed"])
        + model.model_dim
        + model.length
        + (50_000 if phase == "intermediate" else 60_000)
    )
    batch_size = int(config["batch_size"])
    eval_batch = int(config["evaluation_batch_size"])
    subset_n = min(
        int(config["validation_subset_by_length"][str(model.length)]),
        len(validation_data["ids"]),
    )
    subset = select_rows(validation_data, torch.arange(subset_n))
    interval = int(config["evaluation_interval"])
    best_score = (-1.0, -1.0, -1.0)
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for local_iteration in range(1, maximum + 1):
        iteration = iteration_offset + local_iteration
        token_ids, valid, steps, targets, previous, current = make_training_batch(
            train_data, model, iteration, batch_size, generator, device
        )
        outputs = model(token_ids, valid, steps)
        loss, components = losses(
            model,
            outputs,
            token_ids,
            steps,
            targets,
            previous,
            current,
            mode,
            config,
            include_output=(phase == "integration"),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (
            local_iteration == 1
            or local_iteration % interval == 0
            or local_iteration == maximum
        ):
            validation = evaluate(model, subset, device, eval_batch)
            _, route = route_loss_and_metrics(model, outputs, steps)
            score = (
                validation["joint_exact"],
                validation["trace_exact"],
                validation["state_bit_accuracy"],
            )
            if phase == "integration" and score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            record = {
                "step": local_iteration,
                "loss": float(loss.detach()),
                **components,
                "routing_accuracy_diagnostic_only": route["routing_accuracy"],
                "minimum_target_attention_diagnostic_only": route[
                    "minimum_target_attention"
                ],
                "validation": validation,
            }
            history.append(record)
            print(json.dumps({"event": f"ablation_{phase}", **record}), flush=True)
            if phase == "integration" and validation["joint_exact"] == 1.0:
                break

    if phase == "integration":
        model.load_state_dict(best_state)
    else:
        # Intermediate targets have their own training loss; do not select this
        # stage using the deliberately untrained trajectory output.
        best_state = copy.deepcopy(model.state_dict())
    return {
        "steps": local_iteration,
        "best_validation_score": best_score,
        "history": history,
    }


def train(
    config_path: Path, data_root: Path, output_root: Path, length: int, mode: str
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if length not in map(int, config["lengths"]) or mode not in MODES:
        raise ValueError("Unsupported ablation mode or length")
    model_dim = int(config["model_dim"])
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
    data = {
        split: load_data(data_root / f"n{length}" / f"{split}.jsonl.gz")
        for split in ("train", "validation", "test")
    }
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    public_stage: dict[str, Any]
    if mode == "no_route":
        # Retain the full baseline's public Boolean-function supervision. This
        # stage sees local bit assignments but no sequence positions or routes.
        try:
            baseline_config = json.loads(
                (BASE / "standard_v2_config.json").read_text(encoding="utf-8")
            )
            public_config = {**baseline_config, **config}
            public_stage = {
                "completed": True,
                **train_public_mechanism(model, public_config, device),
            }
        except RuntimeError as error:
            public_stage = {"completed": False, "warning": str(error)}
        intermediate = train_phase(
            model,
            data["train"],
            data["validation"],
            config,
            mode,
            "intermediate",
            int(config["intermediate_steps_by_length"][str(length)]),
            0,
            device,
        )
        offset = intermediate["steps"]
    else:
        public_stage = {"completed": False, "reason": "removed by end-to-end ablation"}
        intermediate = {"completed": False, "reason": "all intermediate labels removed"}
        offset = 0

    integration = train_phase(
        model,
        data["train"],
        data["validation"],
        config,
        mode,
        "integration",
        int(config["integration_steps_by_length"][str(length)]),
        offset,
        device,
    )
    validation = evaluate(
        model, data["validation"], device, int(config["evaluation_batch_size"])
    )
    test = evaluate(model, data["test"], device, int(config["evaluation_batch_size"]))
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "mode": mode,
        "length": length,
        "seed": config["seed"],
        "parameter_initialization_seed": initialization_seed,
        "architecture": model.architecture_metadata(),
        "architecture_matches_full_baseline": True,
        "direct_route_position_loss_used": False,
        "all_parameters_trainable": True,
        "supervision": (
            {
                "next_token_and_answer": True,
                "canonical_routed_bits": True,
                "routed_value_bits": True,
                "previous_and_current_masks": True,
                "mask_delta": True,
                "public_boolean_truth_table": True,
            }
            if mode == "no_route"
            else {
                "next_token_and_answer": True,
                "canonical_routed_bits": False,
                "routed_value_bits": False,
                "previous_and_current_masks": False,
                "mask_delta": False,
                "public_boolean_truth_table": False,
            }
        ),
        "route_metrics_are_diagnostics_not_losses": True,
        "public_stage": public_stage,
        "intermediate_training": intermediate,
        "integration_training": integration,
        "validation_softmax_attention": validation,
        "test_softmax_attention": test,
        "passes_full_baseline_gate": task_passes(test, config["success_gate"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(
        {"model_state": model.state_dict(), "config": config, "mode": mode},
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
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    train(args.config, args.data_root, args.output_root, args.length, args.mode)


if __name__ == "__main__":
    main()
