#!/usr/bin/env python3
"""Train a fresh adapter to emit a complete encrypted PARITY trajectory."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import training_utils as utils


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")


def trajectory_passes(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        metrics["trace_exact"] >= float(gate["minimum_trace_exact"])
        and metrics["trace_bit_accuracy"] >= float(gate["minimum_trace_bit_accuracy"])
        and metrics["final_parity_accuracy"]
        >= float(gate["minimum_final_parity_accuracy"])
        and metrics["strict_format"] >= float(gate["minimum_strict_format"])
    )


def prerequisites_pass(
    metrics: dict[str, dict[str, float]], gate: dict[str, float]
) -> bool:
    threshold = float(gate["minimum_prerequisite_accuracy"])
    return all(item["accuracy"] >= threshold for item in metrics.values())


def evaluate_transitions(
    model: Any, rows: list[dict[str, Any]], pad_id: int, device: Any, batch_size: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    by_position = {
        task: utils.evaluate_prerequisite(model, current, pad_id, device, batch_size)
        for task, current in sorted(grouped.items())
    }
    total = sum(item["n"] for item in by_position.values())
    correct = sum(item["n"] * item["accuracy"] for item in by_position.values())
    return {
        "overall": {"n": total, "accuracy": correct / total},
        "by_position": by_position,
    }


def transitions_pass(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return metrics["overall"]["accuracy"] >= float(
        gate["minimum_transition_accuracy"]
    ) and min(item["accuracy"] for item in metrics["by_position"].values()) >= float(
        gate["minimum_transition_position_accuracy"]
    )


def evaluate_prerequisites(
    model: Any,
    data: dict[str, list[dict[str, Any]]],
    tasks: list[str],
    n: int | None,
    pad_id: int,
    device: Any,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    return {
        task: utils.evaluate_prerequisite(
            model,
            data[task] if n is None else data[task][:n],
            pad_id,
            device,
            batch_size,
        )
        for task in tasks
    }


def evaluate_retention(
    model: Any,
    data: dict[str, list[dict[str, Any]]],
    tasks: list[str],
    config: dict[str, Any],
    pad_id: int,
    device: Any,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Evaluate cheaply, but confirm a borderline failure on the full held-out sets."""
    metrics = evaluate_prerequisites(
        model,
        data,
        tasks,
        int(config["prerequisite_evaluation_n"]),
        pad_id,
        device,
        batch_size,
    )
    threshold = float(config["success_gate"]["minimum_prerequisite_accuracy"])
    margin = float(config["retention_confirmation_margin"])
    minimum = min(item["accuracy"] for item in metrics.values())
    if threshold - margin <= minimum < threshold:
        return evaluate_prerequisites(
            model, data, tasks, None, pad_id, device, batch_size
        )
    return metrics


def run_stage(
    *,
    model: Any,
    tokenizer: Any,
    data: dict[str, list[dict[str, Any]]],
    transition_train: list[dict[str, Any]],
    transition_validation: list[dict[str, Any]],
    prerequisite_train: dict[str, list[dict[str, Any]]],
    prerequisite_validation: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    length: int,
    previous_length: int | None,
    epochs: int,
    learning_rate: float,
    prerequisite_replay_n: int,
    phase: str,
    pad_id: int,
    device: Any,
    rng: random.Random,
    history: list[dict[str, Any]],
    output_root: Path,
    started: float,
) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW

    batch_size = int(config["batch_size"])
    evaluation_batch_size = int(config["evaluation_batch_size"])
    tasks = list(config["prerequisite_tasks"])
    gate = config["success_gate"]
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=0.0,
    )
    best_score, best_state, best_record = -math.inf, None, None
    stage_history = []
    for epoch in range(1, epochs + 1):
        groups = [data[f"n{length}_train"]]
        groups.append(
            utils.sample(transition_train, int(config["transition_replay_n"]), rng)
        )
        for earlier in [
            value
            for value in map(int, config["lengths"])
            if value < length and f"n{value}_train" in data
        ]:
            groups.append(
                utils.sample(
                    data[f"n{earlier}_train"],
                    int(config["previous_length_replay_n"]),
                    rng,
                )
            )
        for task in tasks:
            groups.append(
                utils.sample(prerequisite_train[task], prerequisite_replay_n, rng)
            )
        model.train()
        totals = defaultdict(float)
        seen = 0
        for rows in utils.epoch_batches(groups, batch_size, rng):
            batch = utils.make_train_batch(rows, pad_id, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            loss = utils.weighted_causal_loss(
                outputs.logits, batch["input_ids"], batch["loss_weights"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            count = len(rows)
            totals["loss"] += count * float(loss.detach().cpu())
            seen += count

        current = utils.evaluate_trajectory(
            model,
            tokenizer,
            data[f"n{length}_validation"],
            pad_id,
            device,
            evaluation_batch_size,
        )
        previous = None
        if previous_length is not None:
            previous = utils.evaluate_trajectory(
                model,
                tokenizer,
                data[f"n{previous_length}_validation"],
                pad_id,
                device,
                evaluation_batch_size,
            )
        prerequisite_metrics = evaluate_retention(
            model,
            prerequisite_validation,
            tasks,
            config,
            pad_id,
            device,
            evaluation_batch_size,
        )
        retention_ok = prerequisites_pass(prerequisite_metrics, gate)
        transition_metrics = evaluate_transitions(
            model, transition_validation, pad_id, device, evaluation_batch_size
        )
        retention_ok &= transitions_pass(transition_metrics, gate)
        if previous is not None:
            retention_ok &= previous["trace_exact"] >= float(
                gate["minimum_trace_exact"]
            )
        # Select the strongest trajectory, even if it needs a later retention repair.
        # A large retention bonus previously discarded a 98.8%-exact trajectory in
        # favour of a retained but substantially weaker checkpoint.
        score = (
            current["trace_exact"]
            + 0.25 * current["final_parity_accuracy"]
            + 0.10 * current["trace_bit_accuracy"]
        )
        if previous is not None:
            score += 0.10 * previous["trace_exact"]
        record = {
            "length": length,
            "phase": phase,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "prerequisite_replay_n": prerequisite_replay_n,
            "train": {key: value / seen for key, value in totals.items()},
            "current": current,
            "previous_length": previous_length,
            "previous": previous,
            "prerequisites": prerequisite_metrics,
            "transitions": transition_metrics,
            "retention_ok": retention_ok,
            "selection_score": score,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        stage_history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_state = utils.clone_trainable(model)
            best_record = record
        (output_root / "progress.json").write_text(
            json.dumps({"current_length": length, "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            epoch >= int(config["minimum_epochs"])
            and trajectory_passes(current, gate)
            and retention_ok
        ):
            # A jointly passing checkpoint always takes precedence over a
            # marginally higher trajectory-only score from an earlier epoch.
            best_score = score
            best_state = utils.clone_trainable(model)
            best_record = record
            break
    if best_state is None or best_record is None:
        raise RuntimeError(f"No checkpoint selected for length {length}")
    utils.restore_trainable(model, best_state)
    return {
        "length": length,
        "best_epoch": best_record["epoch"],
        "best_score": best_score,
        "best_current": best_record["current"],
        "best_previous": best_record["previous"],
        "best_prerequisites": best_record["prerequisites"],
        "best_transitions": best_record["transitions"],
        "retention_ok": best_record["retention_ok"],
        "history": stage_history,
    }


def train(
    config_path: Path,
    data_root: Path,
    prerequisite_data_root: Path,
    output_root: Path,
    adapters: list[Path],
    selected_length: int | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)
    tokenizer, model, device = utils.load_frozen_stack(config, adapters)
    format_weight = float(config["format_loss_weight"])
    bit_weight = float(config["bit_loss_weight"])
    configured_lengths = list(map(int, config["lengths"]))
    if selected_length is not None and selected_length not in configured_lengths:
        raise ValueError(f"Length {selected_length} is not configured")
    # The theorem permits a different T_N for every N. Train each requested
    # length independently so curriculum interference cannot determine the result.
    lengths = [selected_length] if selected_length is not None else configured_lengths
    data = {
        f"n{length}_{split}": utils.load_rows(
            tokenizer,
            data_root / f"trajectory_n{length}_{split}.jsonl",
            format_weight,
            bit_weight,
        )
        for length in lengths
        for split in ("train", "validation")
    }
    transition_train = utils.load_rows(
        tokenizer, data_root / "transition_train.jsonl", format_weight, bit_weight
    )
    transition_validation = utils.load_rows(
        tokenizer, data_root / "transition_validation.jsonl", format_weight, bit_weight
    )
    prerequisite_data = {}
    for task in config["prerequisite_tasks"]:
        prerequisite_data[task] = utils.load_rows(
            tokenizer,
            prerequisite_data_root / f"{task}_validation.jsonl",
            format_weight,
            bit_weight,
        )
        prerequisite_data[f"{task}_train"] = utils.load_rows(
            tokenizer,
            prerequisite_data_root / f"{task}_train.jsonl",
            format_weight,
            bit_weight,
        )
    replay_data = {
        task: prerequisite_data[f"{task}_train"]
        for task in config["prerequisite_tasks"]
    }
    validation_data = {
        task: prerequisite_data[task] for task in config["prerequisite_tasks"]
    }
    pad_id = tokenizer.pad_token_id
    gate = config["success_gate"]
    eval_batch_size = int(config["evaluation_batch_size"])
    baseline = evaluate_prerequisites(
        model,
        validation_data,
        list(config["prerequisite_tasks"]),
        None,
        pad_id,
        device,
        eval_batch_size,
    )
    print(
        json.dumps({"event": "frozen_prerequisite_baseline", "metrics": baseline}),
        flush=True,
    )
    if not prerequisites_pass(baseline, gate):
        raise RuntimeError(f"Frozen single-step stack failed its gate: {baseline}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    history, stages = [], []
    baseline_transitions = evaluate_transitions(
        model, transition_validation, pad_id, device, eval_batch_size
    )
    print(
        json.dumps(
            {
                "event": "position_general_transition_baseline",
                "metrics": baseline_transitions,
            }
        ),
        flush=True,
    )
    if not transitions_pass(baseline_transitions, gate):
        raise RuntimeError(
            f"Established full-update adapter failed the position gate: {baseline_transitions}"
        )
    utils.save_adapter(
        model,
        output_root / "validated_start_adapter",
        {
            "experiment_name": config["experiment_name"],
            "stage": "validated_position_general_full_update",
            "rank": config["lora_rank"],
        },
    )
    for index, length in enumerate(lengths):
        previous = lengths[index - 1] if index else None
        result = run_stage(
            model=model,
            tokenizer=tokenizer,
            data=data,
            transition_train=transition_train,
            transition_validation=transition_validation,
            prerequisite_train=replay_data,
            prerequisite_validation=validation_data,
            config=config,
            length=length,
            previous_length=previous,
            epochs=int(config["epochs_per_length"]),
            learning_rate=float(config["learning_rate"]),
            phase="main",
            prerequisite_replay_n=int(config["prerequisite_replay_n"]),
            pad_id=pad_id,
            device=device,
            rng=random.Random(f"trajectory:{seed}:n{length}"),
            history=history,
            output_root=output_root,
            started=started,
        )
        stages.append(result)
        passed = (
            trajectory_passes(result["best_current"], gate) and result["retention_ok"]
        )
        candidate = result
        if not passed and not trajectory_passes(candidate["best_current"], gate):
            candidate = run_stage(
                model=model,
                tokenizer=tokenizer,
                data=data,
                transition_train=transition_train,
                transition_validation=transition_validation,
                prerequisite_train=replay_data,
                prerequisite_validation=validation_data,
                config=config,
                length=length,
                previous_length=previous,
                epochs=int(config["rescue_epochs"]),
                learning_rate=float(config["rescue_learning_rate"]),
                phase="rescue",
                prerequisite_replay_n=int(config["prerequisite_replay_n"]),
                pad_id=pad_id,
                device=device,
                rng=random.Random(f"trajectory:{seed}:rescue:n{length}"),
                history=history,
                output_root=output_root,
                started=started,
            )
            stages.append(candidate)
            passed = (
                trajectory_passes(candidate["best_current"], gate)
                and candidate["retention_ok"]
            )
        if (
            not passed
            and trajectory_passes(candidate["best_current"], gate)
            and not candidate["retention_ok"]
        ):
            candidate = run_stage(
                model=model,
                tokenizer=tokenizer,
                data=data,
                transition_train=transition_train,
                transition_validation=transition_validation,
                prerequisite_train=replay_data,
                prerequisite_validation=validation_data,
                config=config,
                length=length,
                previous_length=previous,
                epochs=int(config["retention_repair_epochs"]),
                learning_rate=float(config["retention_repair_learning_rate"]),
                prerequisite_replay_n=int(config["retention_repair_replay_n"]),
                phase="retention_repair",
                pad_id=pad_id,
                device=device,
                rng=random.Random(f"trajectory:{seed}:repair:n{length}"),
                history=history,
                output_root=output_root,
                started=started,
            )
            stages.append(candidate)
            passed = (
                trajectory_passes(candidate["best_current"], gate)
                and candidate["retention_ok"]
            )
        if not passed and length != lengths[-1]:
            raise RuntimeError(
                f"Length {length} failed; refusing to train a longer autoregressive trajectory"
            )
        if passed:
            utils.save_adapter(
                model,
                output_root / "checkpoints" / f"n{length}",
                {
                    "experiment_name": config["experiment_name"],
                    "stage": f"trajectory_n{length}",
                    "rank": config["lora_rank"],
                },
            )

    final_by_length = {
        str(length): utils.evaluate_trajectory(
            model,
            tokenizer,
            data[f"n{length}_validation"],
            pad_id,
            device,
            eval_batch_size,
        )
        for length in lengths
    }
    final_prerequisites = evaluate_prerequisites(
        model,
        validation_data,
        list(config["prerequisite_tasks"]),
        None,
        pad_id,
        device,
        eval_batch_size,
    )
    final_transitions = evaluate_transitions(
        model, transition_validation, pad_id, device, eval_batch_size
    )
    passes = all(trajectory_passes(item, gate) for item in final_by_length.values())
    passes &= prerequisites_pass(final_prerequisites, gate)
    passes &= transitions_pass(final_transitions, gate)
    adapter = output_root / "best_adapter"
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "architecture": (
            "frozen mechanism/positions stack + independent rank-32 trajectory adapter"
        ),
        "fixed_length": selected_length,
        "established_adapters": [str(path) for path in adapters],
        "baseline_prerequisites": baseline,
        "baseline_transitions": baseline_transitions,
        "stages": stages,
        "final_by_length": final_by_length,
        "final_prerequisites": final_prerequisites,
        "final_transitions": final_transitions,
        "passes_gate": bool(passes),
        "elapsed_seconds": time.monotonic() - started,
        "final_adapter": str(adapter),
    }
    utils.save_adapter(
        model,
        adapter,
        {
            "experiment_name": config["experiment_name"],
            "rank": config["lora_rank"],
            "passes_gate": bool(passes),
        },
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
        default=Path("generated_data/parity_goldreich_trajectory/seed_0"),
    )
    parser.add_argument(
        "--prerequisite-data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_final/seed_0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_trajectory"),
    )
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.config,
                args.data_root,
                args.prerequisite_data_root,
                args.output_root,
                args.adapter,
                args.length,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
