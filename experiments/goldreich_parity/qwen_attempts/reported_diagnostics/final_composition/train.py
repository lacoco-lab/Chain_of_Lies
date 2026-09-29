#!/usr/bin/env python3
"""Train the final position-composition or full-update adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import training_utils as utils
from generate_data import RANK32_CALIBRATION_DIGESTS

RANK32_SELECTION_WEIGHTS = {"local": 0.25, "supplied_update": 0.10}


def verify_rank32_calibration(data_root: Path) -> None:
    """Refuse to train if the model-facing calibration data changed in transit."""
    observed = {}
    for name in RANK32_CALIBRATION_DIGESTS:
        digest = hashlib.sha256()
        path = data_root / f"{name}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                digest.update(line.encode())
        observed[name] = digest.hexdigest()
    if observed != RANK32_CALIBRATION_DIGESTS:
        raise RuntimeError(f"Rank-32 calibration verification failed: {observed}")


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


def evaluate_tasks(
    model: Any,
    data: dict[str, list[dict]],
    tasks: list[str],
    pad_id: int,
    device: Any,
    batch_size: int,
    bit_tokens: tuple[int, int],
) -> dict[str, dict[str, float]]:
    return {
        task: utils.evaluate(
            model, data[f"{task}_validation"], pad_id, device, batch_size, bit_tokens
        )
        for task in tasks
    }


def retention_passes(
    evaluations: dict[str, dict[str, float]],
    retained_tasks: list[str],
    gate: dict[str, float],
) -> bool:
    thresholds = {
        "local": float(gate["minimum_local_retention"]),
        "supplied_update": float(gate["minimum_supplied_update_retention"]),
        "derive_previous": float(gate["minimum_composition_accuracy"]),
        "derive_current": float(gate["minimum_composition_accuracy"]),
    }
    default = float(gate["minimum_composition_accuracy"])
    return all(
        evaluations[task]["greedy_accuracy"] >= thresholds.get(task, default)
        for task in retained_tasks
    )


def run_stage(
    *,
    model: Any,
    data: dict[str, list[dict]],
    config: dict[str, Any],
    stage: str,
    replay_groups: list[tuple[str, int | None]],
    retained_tasks: list[str],
    epochs: int,
    pad_id: int,
    device: Any,
    bit_tokens: tuple[int, int],
    rng: random.Random,
    selection_weights: dict[str, float],
    started: float,
    history: list[dict],
    output_root: Path,
    use_attention_loss: bool = True,
) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW

    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["evaluation_batch_size"])
    validation = data[f"{stage}_validation"]
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    best_score, best_state, best_record = -math.inf, None, None
    stage_history = []
    evaluation_tasks = list(dict.fromkeys([stage, *retained_tasks]))

    for epoch in range(1, epochs + 1):
        groups = [data[f"{stage}_train"]]
        for name, count in replay_groups:
            source = data[name]
            groups.append(source if count is None else utils.sample(source, count, rng))
        model.train()
        totals = defaultdict(float)
        seen = 0
        for batch_rows in utils.epoch_batches(groups, batch_size, rng):
            batch = utils.make_batch(batch_rows, pad_id, device)
            needs_attention = use_attention_loss and any(
                row["attention_groups"] for row in batch_rows
            )
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
            auxiliary = torch.zeros((), device=device)
            if needs_attention:
                auxiliary, _ = utils.attention_loss(
                    outputs.attentions,
                    batch,
                    int(config["attention_layers"]),
                    int(config["attention_heads"]),
                    float(config["attention_coverage_weight"]),
                )
            loss = bit_ce + float(config["attention_loss_weight"]) * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            count = len(batch_rows)
            seen += count
            totals["loss"] += count * float(loss.detach().cpu())
            totals["bit_ce"] += count * float(bit_ce.detach().cpu())
            totals["attention_aux"] += count * float(auxiliary.detach().cpu())

        evaluations = evaluate_tasks(
            model, data, evaluation_tasks, pad_id, device, eval_batch_size, bit_tokens
        )
        attention = utils.attention_diagnostic(
            model,
            validation[: int(config["attention_evaluation_n"])],
            pad_id,
            device,
            min(batch_size, eval_batch_size),
            config,
        )
        retention_ok = retention_passes(
            evaluations, retained_tasks, config["success_gate"]
        )
        score = evaluations[stage]["greedy_accuracy"]
        score += sum(
            selection_weights[task] * evaluations[task]["greedy_accuracy"]
            for task in retained_tasks
        )
        score += 10.0 if retention_ok else 0.0
        record = {
            "stage": stage,
            "epoch": epoch,
            "train": {key: value / seen for key, value in totals.items()},
            "evaluations": evaluations,
            "validation_attention": attention,
            "retention_ok": retention_ok,
            "selection_weights": selection_weights,
            "selection_score": score,
            "attention_regularization_enabled": use_attention_loss,
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
            json.dumps({"current_stage": stage, "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            epoch >= int(config["minimum_epochs"])
            and evaluations[stage]["greedy_accuracy"]
            >= float(config["early_stop_accuracy"])
            and retention_ok
        ):
            break

    if best_state is None or best_record is None:
        raise RuntimeError(f"No checkpoint selected for {stage}")
    utils.restore_trainable(model, best_state)
    return {
        "stage": stage,
        "best_epoch": best_record["epoch"],
        "best_score": best_score,
        "best_evaluations": best_record["evaluations"],
        "history": stage_history,
    }


def train(
    config_path: Path,
    action: str,
    data_root: Path,
    output_root: Path,
    mechanism_adapter: Path,
    positions_adapter: Path | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_rank32_calibration(data_root)
    seed = int(config["seed"])
    set_seed(seed)
    established = [mechanism_adapter]
    if action == "full":
        if positions_adapter is None:
            raise ValueError("The full action requires the positions adapter")
        established.append(positions_adapter)
    tokenizer, model, device = utils.load_frozen_stack(config, established)
    data = utils.load_data(tokenizer, data_root)
    bit_tokens = utils.binary_tokens(data)
    pad_id = tokenizer.pad_token_id
    eval_batch_size = int(config["evaluation_batch_size"])
    gate = config["success_gate"]
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    prerequisite_tasks = ["derive_previous", "local", "supplied_update"]
    if action == "full":
        prerequisite_tasks = [
            "derive_previous",
            "derive_current",
            "local",
            "supplied_update",
        ]
    baseline = evaluate_tasks(
        model, data, prerequisite_tasks, pad_id, device, eval_batch_size, bit_tokens
    )
    retained_prerequisites = [
        task for task in prerequisite_tasks if task != "derive_previous"
    ]
    if action == "full":
        retained_prerequisites = prerequisite_tasks
    if not retention_passes(baseline, retained_prerequisites, gate):
        raise RuntimeError(
            f"Frozen prerequisite stack failed before {action} training: {baseline}"
        )

    replay = config["replay"]
    history, stages = [], []
    local_replay = [("local_replay", None)] * int(replay["local_copies"])
    supplied_replay = [("supplied_update_train", int(replay["supplied_update_n"]))]
    if action == "positions":
        stages.append(
            run_stage(
                model=model,
                data=data,
                config=config,
                stage="derive_previous",
                replay_groups=[*local_replay, *supplied_replay],
                retained_tasks=["local", "supplied_update"],
                epochs=int(config["position_epochs"]),
                pad_id=pad_id,
                device=device,
                bit_tokens=bit_tokens,
                rng=random.Random(f"{seed}:composition-rank"),
                selection_weights=RANK32_SELECTION_WEIGHTS,
                started=started,
                history=history,
                output_root=output_root,
                use_attention_loss=True,
            )
        )
        previous_now = evaluate_tasks(
            model,
            data,
            ["derive_previous"],
            pad_id,
            device,
            eval_batch_size,
            bit_tokens,
        )["derive_previous"]["greedy_accuracy"]
        if previous_now < float(gate["minimum_composition_accuracy"]):
            raise RuntimeError(
                f"Previous-mask composition failed before current-mask stage: {previous_now}"
            )
        stages.append(
            run_stage(
                model=model,
                data=data,
                config=config,
                stage="derive_current",
                replay_groups=[
                    ("derive_previous_train", int(replay["previous_position_n"])),
                    *local_replay,
                    *supplied_replay,
                ],
                retained_tasks=["derive_previous", "local", "supplied_update"],
                epochs=int(config["position_epochs"]),
                pad_id=pad_id,
                device=device,
                bit_tokens=bit_tokens,
                rng=random.Random(f"{seed}:goldreich-final:derive-current"),
                selection_weights={
                    "derive_previous": 0.10,
                    "local": 0.10,
                    "supplied_update": 0.10,
                },
                started=started,
                history=history,
                output_root=output_root,
                use_attention_loss=True,
            )
        )
        final_tasks = ["derive_previous", "derive_current", "local", "supplied_update"]
    elif action == "full":
        # The previous endpoint supplied only a final random-looking parity bit,
        # which gave no intermediate credit for discovering two masks simultaneously.
        # Directly learn their XOR, transfer it to the endpoint wording with no
        # public terms, and then add one public XOR term per stage.
        base_replay = [
            ("derive_previous_train", int(replay["full_previous_n"])),
            ("derive_current_train", int(replay["full_current_n"])),
            ("local_replay", int(replay["full_local_n"])),
            ("supplied_update_train", int(replay["full_supplied_n"])),
        ]
        curriculum_replay_n = int(replay["full_curriculum_n"])
        curriculum = [
            (
                "mask_delta",
                [],
                ["derive_previous", "derive_current", "local", "supplied_update"],
            ),
            (
                "full_zero",
                [("mask_delta_train", curriculum_replay_n)],
                [
                    "mask_delta",
                    "derive_previous",
                    "derive_current",
                    "local",
                    "supplied_update",
                ],
            ),
            (
                "full_one",
                [
                    ("mask_delta_train", curriculum_replay_n),
                    ("full_zero_train", curriculum_replay_n),
                ],
                [
                    "mask_delta",
                    "full_zero",
                    "derive_previous",
                    "derive_current",
                    "local",
                    "supplied_update",
                ],
            ),
            (
                "full_two",
                [
                    ("mask_delta_train", curriculum_replay_n),
                    ("full_one_train", curriculum_replay_n),
                ],
                [
                    "mask_delta",
                    "full_one",
                    "derive_previous",
                    "derive_current",
                    "local",
                    "supplied_update",
                ],
            ),
            (
                "full_update",
                [
                    ("mask_delta_train", curriculum_replay_n),
                    ("full_two_train", curriculum_replay_n),
                ],
                [
                    "mask_delta",
                    "full_two",
                    "derive_previous",
                    "derive_current",
                    "local",
                    "supplied_update",
                ],
            ),
        ]
        for stage_index, (stage, bridge_replay, retained) in enumerate(curriculum):
            epochs = (
                int(config["mask_delta_epochs"])
                if stage == "mask_delta"
                else (
                    int(config["full_update_epochs"])
                    if stage == "full_update"
                    else int(config["bridge_epochs"])
                )
            )
            weights = {task: 0.05 for task in retained}
            stage_result = run_stage(
                model=model,
                data=data,
                config=config,
                stage=stage,
                replay_groups=[*bridge_replay, *base_replay],
                retained_tasks=retained,
                epochs=epochs,
                pad_id=pad_id,
                device=device,
                bit_tokens=bit_tokens,
                rng=random.Random(
                    f"{seed}:goldreich-final:curriculum:{stage_index}:{stage}"
                ),
                selection_weights=weights,
                started=started,
                history=history,
                output_root=output_root,
                # The frozen position stack already attends correctly (>0.97).
                # Avoiding dense attention tensors makes this curriculum much faster.
                use_attention_loss=False,
            )
            stages.append(stage_result)
            stage_ok = stage_result["best_evaluations"][stage][
                "greedy_accuracy"
            ] >= float(gate["minimum_composition_accuracy"]) and retention_passes(
                stage_result["best_evaluations"], retained, gate
            )
            if not stage_ok:
                # A focused continuation is cheaper and more informative than
                # advancing an unlearned bridge into another chance-level stage.
                rescue = run_stage(
                    model=model,
                    data=data,
                    config=config,
                    stage=stage,
                    replay_groups=[*bridge_replay, *base_replay],
                    retained_tasks=retained,
                    epochs=int(config["curriculum_rescue_epochs"]),
                    pad_id=pad_id,
                    device=device,
                    bit_tokens=bit_tokens,
                    rng=random.Random(
                        f"{seed}:goldreich-final:curriculum-rescue:{stage_index}:{stage}"
                    ),
                    selection_weights=weights,
                    started=started,
                    history=history,
                    output_root=output_root,
                    use_attention_loss=False,
                )
                stages.append(rescue)
                stage_ok = rescue["best_evaluations"][stage][
                    "greedy_accuracy"
                ] >= float(gate["minimum_composition_accuracy"]) and retention_passes(
                    rescue["best_evaluations"], retained, gate
                )
            if not stage_ok and stage != "full_update":
                raise RuntimeError(
                    f"Curriculum stage {stage} did not reach the gated accuracy; "
                    "refusing to train a harder stage on a missing prerequisite"
                )
        final_tasks = [
            "full_update",
            "mask_delta",
            "full_zero",
            "full_one",
            "full_two",
            "derive_previous",
            "derive_current",
            "local",
            "supplied_update",
        ]
    else:
        raise ValueError(f"Unknown action: {action}")

    final = evaluate_tasks(
        model, data, final_tasks, pad_id, device, eval_batch_size, bit_tokens
    )
    required = ["derive_previous", "derive_current", "local", "supplied_update"]
    if action == "full":
        required.append("mask_delta")
    passes = retention_passes(final, required, gate)
    if "full_update" in final:
        passes &= final["full_update"]["greedy_accuracy"] >= float(
            gate["minimum_composition_accuracy"]
        )
    attention_task = "full_update" if action == "full" else "derive_current"
    final_attention = utils.attention_diagnostic(
        model,
        data[f"{attention_task}_validation"][: int(config["attention_evaluation_n"])],
        pad_id,
        device,
        min(int(config["batch_size"]), eval_batch_size),
        config,
    )
    adapter = output_root / "best_adapter"
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "action": action,
        "seed": seed,
        "rank": config["lora_rank"],
        "rank32_calibration_verified": True,
        "rank32_training_rng": f"{seed}:composition-rank",
        "established_adapters_merged_and_frozen": [str(path) for path in established],
        "trainable_parameters": trainable_parameters,
        "baseline": baseline,
        "stages": stages,
        "final_evaluations": final,
        "final_attention": final_attention,
        "passes_gate": bool(passes),
        "elapsed_seconds": time.monotonic() - started,
        "final_adapter": str(adapter),
    }
    utils.save_adapter(
        model,
        adapter,
        {
            "experiment_name": config["experiment_name"],
            "action": action,
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
    parser.add_argument("--action", choices=("positions", "full"), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_final/seed_0"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mechanism-adapter", type=Path, required=True)
    parser.add_argument("--positions-adapter", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.config,
                args.action,
                args.data_root,
                args.output_root,
                args.mechanism_adapter,
                args.positions_adapter,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
