#!/usr/bin/env python3
"""Evaluate a trained Goldreich PARITY model with ordinary softmax attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from generate_data import fixed_graph, seed_bits_for_length
from model import GoldreichToyTransformer, STATE_POSITION
from train import gate_metrics, load_data, routing_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location=device)


def build_model(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    length: int,
    device: torch.device,
) -> GoldreichToyTransformer:
    seed_bits = seed_bits_for_length(config, length)
    graph = fixed_graph(seed_bits, length, int(config["graph_seed"]))
    model = GoldreichToyTransformer(
        length=length,
        seed_bits=seed_bits,
        graph=graph,
        key_dim=int(config["key_dim_by_length"][str(length)]),
        gate_hidden_dim=int(config["gate_hidden_dim"]),
    ).to(device)
    expected = model.architecture_metadata()
    recorded = checkpoint.get("architecture")
    if recorded != expected:
        raise RuntimeError(
            "Checkpoint architecture does not match the requested configuration: "
            f"recorded={recorded}, expected={expected}"
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def validate_supplied_config(
    checkpoint_config: dict[str, Any], supplied: dict[str, Any], length: int
) -> None:
    """Compare only fields that determine this checkpoint's computation.

    Training-only defaults may be added to the repository config after a checkpoint was saved.
    Such additions do not change the model and must not invalidate evaluation.
    """
    mismatches = {}
    for key in (
        "seed",
        "graph_seed",
        "seed_length_exponent",
        "heads",
        "gate_hidden_dim",
    ):
        if checkpoint_config.get(key) != supplied.get(key):
            mismatches[key] = {
                "checkpoint": checkpoint_config.get(key),
                "supplied": supplied.get(key),
            }
    for key in ("seed_bits_by_length", "key_dim_by_length"):
        checkpoint_value = checkpoint_config.get(key, {}).get(str(length))
        supplied_value = supplied.get(key, {}).get(str(length))
        if checkpoint_value != supplied_value:
            mismatches[f"{key}[{length}]"] = {
                "checkpoint": checkpoint_value,
                "supplied": supplied_value,
            }
    if length not in map(int, checkpoint_config.get("lengths", [])):
        mismatches["checkpoint_lengths"] = {
            "checkpoint": checkpoint_config.get("lengths"),
            "required": length,
        }
    if length not in map(int, supplied.get("lengths", [])):
        mismatches["supplied_lengths"] = {
            "supplied": supplied.get("lengths"),
            "required": length,
        }
    if mismatches:
        raise RuntimeError(f"Checkpoint and supplied config disagree: {mismatches}")


@torch.inference_mode()
def attention_probabilities(model: GoldreichToyTransformer) -> torch.Tensor:
    """Return the exact softmax QK attention used by ``model.soft_step``."""
    steps = torch.arange(model.steps, device=model.keys.device)
    return model.attention_scores(steps).softmax(dim=-1)


@torch.inference_mode()
def validate_gold_batch(
    model: GoldreichToyTransformer,
    seeds: torch.Tensor,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    states: torch.Tensor,
    answers: torch.Tensor,
) -> None:
    """Recompute the construction, refusing to score malformed test rows."""
    graph = torch.tensor(model.graph, dtype=torch.long, device=seeds.device)
    chosen = seeds[:, graph]
    reconstructed_masks = (
        chosen[..., 0]
        ^ chosen[..., 1]
        ^ chosen[..., 2]
        ^ (chosen[..., 3] & chosen[..., 4])
    ).long()
    plaintext_states = inputs.long().cumsum(dim=1).remainder(2)
    reconstructed_states = plaintext_states ^ reconstructed_masks
    if not torch.equal(reconstructed_masks, masks.long()):
        raise RuntimeError("Test masks do not match the private seeds and public graph")
    if not torch.equal(reconstructed_states, states.long()):
        raise RuntimeError(
            "Test encrypted states do not match the specified construction"
        )
    if not torch.equal(plaintext_states[:, -1], answers.long()):
        raise RuntimeError("Test PARITY answers are invalid")


@torch.inference_mode()
def evaluate_softmax(
    model: GoldreichToyTransformer,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Full autoregressive evaluation with exact vectorized softmax attention.

    Attention weights depend only on public step/head/source indices. We compute them once, use
    one matrix multiplication per batch for fixed seed/input contributions, and add the recurrent
    state contribution step by step. This is algebraically identical to repeated ``soft_step``
    calls and uses no attention truncation or approximation.
    """
    probabilities = attention_probabilities(model)
    static_probabilities = probabilities.clone()
    static_probabilities[:, :, STATE_POSITION] = 0.0
    flat_static = static_probabilities.permute(2, 0, 1).reshape(model.sources, -1)
    state_weights = probabilities[:, :, STATE_POSITION]

    total = len(data["ids"])
    bit_correct = trace_correct = answer_correct = joint_correct = 0
    per_step_correct = torch.zeros(model.steps, dtype=torch.long)
    samples: list[dict[str, Any]] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        seeds = data["seeds"][start:end].to(device)
        inputs = data["inputs"][start:end].to(device)
        gold_masks = data["masks"][start:end].to(device)
        gold_states = data["states"][start:end].to(device)
        gold_answers = data["answers"][start:end].to(device)
        validate_gold_batch(model, seeds, inputs, gold_masks, gold_states, gold_answers)

        previous = torch.zeros(end - start, dtype=torch.long, device=device)
        sources = model.build_sources(seeds, inputs, previous).float()
        static_values = (sources @ flat_static).reshape(end - start, model.steps, -1)
        predictions = []
        final_mask = None
        for step in range(model.steps):
            bits = (
                static_values[:, step, :]
                + previous.float()[:, None] * state_weights[step]
            )
            previous_mask_logits = model.predicate_gate(bits[:, 2:7])
            current_mask_logits = model.predicate_gate(bits[:, 7:12])
            previous_mask = previous_mask_logits.softmax(dim=-1)[:, 1]
            current_mask = current_mask_logits.softmax(dim=-1)[:, 1]
            value = bits[:, 0]
            for operand in (bits[:, 1], previous_mask, current_mask):
                value, state_logits = model._xor_soft(value, operand)
            previous = state_logits.argmax(dim=-1)
            final_mask = current_mask_logits.argmax(dim=-1)
            predictions.append(previous)

        predicted_states = torch.stack(predictions, dim=1)
        if final_mask is None:
            raise RuntimeError("No recurrent steps were evaluated")
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


def passes(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    gate = config["success_gate"]
    return (
        metrics["state_bit_accuracy"] >= float(gate["minimum_state_bit_accuracy"])
        and metrics["trace_exact"] >= float(gate["minimum_trace_exact"])
        and metrics["final_parity_accuracy"]
        >= float(gate["minimum_final_parity_accuracy"])
        and metrics["joint_exact"] >= float(gate["minimum_joint_exact"])
    )


def default_batch_size(length: int, device: torch.device) -> int:
    if device.type == "cuda":
        return 1024 if length <= 256 else 256 if length <= 1024 else 128
    return 512 if length <= 256 else 128 if length <= 512 else 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.length not in map(int, config["lengths"]):
        raise ValueError(f"Length {args.length} is not present in {args.config}")
    if int(config["seed"]) != 0:
        raise ValueError("The reported experiment requires experimental seed 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.set_float32_matmul_precision("highest")
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("Checkpoint does not contain its training configuration")
    validate_supplied_config(checkpoint_config, config, args.length)
    model = build_model(checkpoint, checkpoint_config, args.length, device)
    original = json.loads(args.original_metrics.read_text(encoding="utf-8"))
    if int(original["length"]) != args.length or int(original["test_examples"]) != 8192:
        raise RuntimeError(
            "Original metrics do not identify the expected 8,192-example test set"
        )
    if (
        original.get("seed_split")
        != "all validation and test PRG seeds are unseen during training"
    ):
        raise RuntimeError(
            "Original run does not record the required disjoint seed split"
        )

    data = load_data(args.test_data)
    if len(data["ids"]) != int(original["test_examples"]):
        raise RuntimeError("Test row count differs from the original validated run")
    observed_widths = {
        "seed_bits": int(data["seeds"].shape[1]),
        "input_bits": int(data["inputs"].shape[1]),
        "masks": int(data["masks"].shape[1]),
        "encrypted_states": int(data["states"].shape[1]),
    }
    expected_widths = {
        "seed_bits": model.seed_bits,
        "input_bits": model.length,
        "masks": model.length,
        "encrypted_states": model.length,
    }
    if observed_widths != expected_widths:
        raise RuntimeError(
            "Test data dimensions do not match the checkpoint: "
            f"observed={observed_widths}, expected={expected_widths}"
        )
    batch_size = args.batch_size or default_batch_size(args.length, device)
    started = time.monotonic()
    softmax_metrics = evaluate_softmax(model, data, device, batch_size)
    result = {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_toy_transformer_full_softmax_test",
        "evaluation_only": True,
        "attention_mode": "ordinary_softmax",
        "autoregressive_decoding": "argmax encrypted state token fed to the next step",
        "vectorization": "exact algebraic reordering; no attention truncation or approximation",
        "experimental_seed": int(config["seed"]),
        "graph_seed": int(config["graph_seed"]),
        "length": args.length,
        "seed_bits": model.seed_bits,
        "test_data": str(args.test_data),
        "test_data_sha256": sha256_file(args.test_data),
        "test_seed_split": original["seed_split"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "batch_size": batch_size,
        "architecture": model.architecture_metadata(),
        "gate_metrics": gate_metrics(model, device),
        "routing_diagnostics": routing_metrics(model, device),
        "test_softmax_attention": softmax_metrics,
        "passes_original_task_gate": passes(softmax_metrics, config),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passes_original_task_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
