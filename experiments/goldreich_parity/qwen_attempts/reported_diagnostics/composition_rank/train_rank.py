#!/usr/bin/env python3
"""Train a fresh composition LoRA after freezing the successful mechanism adapter."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.goldreich_parity.qwen_attempts.exploratory.encrypted_bridge import (
    train_bridge as common,
)  # noqa: E402

DATASET_NAMES = (
    "local_replay",
    "local_validation",
    "supplied_update_train",
    "supplied_update_validation",
    "derive_previous_train",
    "derive_previous_validation",
    "derive_current_validation",
)


def load_model(
    config: dict[str, Any], rank: int, mechanism_adapter: Path
) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = mechanism_adapter / "adapter_config.json"
    weights_path = mechanism_adapter / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing mechanism adapter or weights: {mechanism_adapter}"
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    # Merge the already learned one-mask and supplied-update behavior into frozen base weights.
    mechanism = PeftModel.from_pretrained(
        base, str(mechanism_adapter), is_trainable=False
    )
    merged = mechanism.merge_and_unload(safe_merge=True)
    merged.config.use_cache = False
    try:
        merged.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        merged.gradient_checkpointing_enable()
    merged.enable_input_require_grads()

    composition_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(merged, composition_config)
    model.to(torch.device("cuda"))
    return tokenizer, model, torch.device("cuda")


def load_data(tokenizer: Any, data_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            common.prepare_row(tokenizer, row)
            for row in common.read_jsonl(data_root / f"{name}.jsonl")
        ]
        for name in DATASET_NAMES
    }


def binary_tokens(data: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    for rows in data.values():
        for row in rows:
            previous = mapping.setdefault(row["gold_bit"], row["target_id"])
            if previous != row["target_id"]:
                raise ValueError("Context-dependent binary target token")
    if set(mapping) != {0, 1}:
        raise ValueError(f"Missing binary targets: {mapping}")
    return mapping[0], mapping[1]


def save_adapter(model: Any, directory: Path, metadata: dict[str, Any]) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)
    (directory / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def train(
    config_path: Path,
    rank: int,
    data_root: Path,
    output_root: Path,
    mechanism_adapter: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.optim import AdamW

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if rank not in list(map(int, config["ranks"])):
        raise ValueError(f"Rank {rank} is not configured")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    tokenizer, model, device = load_model(config, rank, mechanism_adapter)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("Only the fresh composition LoRA may be trainable")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    data = load_data(tokenizer, data_root)
    bit_tokens = binary_tokens(data)
    pad_id = tokenizer.pad_token_id
    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["evaluation_batch_size"])
    condition = str(config["condition"])
    train_rows = data[f"{condition}_train"]
    validation_rows = data[f"{condition}_validation"]
    gate = config["success_gate"]
    # Keep replay samples and batch ordering identical across ranks.
    rng = random.Random(f"{seed}:composition-rank")
    started = time.monotonic()

    baseline = {
        "composition": common.evaluate(
            model, validation_rows, pad_id, device, eval_batch_size, bit_tokens
        ),
        "local": common.evaluate(
            model, data["local_validation"], pad_id, device, eval_batch_size, bit_tokens
        ),
        "supplied_update": common.evaluate(
            model,
            data["supplied_update_validation"],
            pad_id,
            device,
            eval_batch_size,
            bit_tokens,
        ),
    }
    baseline_gate = bool(
        baseline["local"]["greedy_accuracy"] >= float(gate["minimum_local_retention"])
        and baseline["supplied_update"]["greedy_accuracy"]
        >= float(gate["minimum_supplied_update_retention"])
    )
    if not baseline_gate:
        raise RuntimeError(f"Merged mechanism failed its retention gate: {baseline}")

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    best_score, best_state, best_record = -math.inf, None, None
    history = []
    output_root.mkdir(parents=True, exist_ok=True)
    replay = config["replay"]

    for epoch in range(1, int(config["epochs"]) + 1):
        groups = [train_rows]
        groups.extend(data["local_replay"] for _ in range(int(replay["local_copies"])))
        groups.append(
            common.sample(
                data["supplied_update_train"], int(replay["supplied_update_n"]), rng
            )
        )
        model.train()
        totals = defaultdict(float)
        seen = 0
        for batch_rows in common.epoch_batches(groups, batch_size, rng):
            batch = common.make_batch(batch_rows, pad_id, device)
            needs_attention = any(row["attention_groups"] for row in batch_rows)
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
            attention_loss = torch.zeros((), device=device)
            if needs_attention:
                attention_loss, _ = common.attention_auxiliary_loss(
                    outputs.attentions,
                    batch,
                    int(config["attention_layers"]),
                    int(config["attention_heads"]),
                    float(config["attention_coverage_weight"]),
                )
            loss = bit_ce + float(config["attention_loss_weight"]) * attention_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            count = len(batch_rows)
            seen += count
            totals["loss"] += count * float(loss.detach().cpu())
            totals["bit_ce"] += count * float(bit_ce.detach().cpu())
            totals["attention_aux"] += count * float(attention_loss.detach().cpu())

        composition_eval = common.evaluate(
            model, validation_rows, pad_id, device, eval_batch_size, bit_tokens
        )
        local_eval = common.evaluate(
            model, data["local_validation"], pad_id, device, eval_batch_size, bit_tokens
        )
        supplied_eval = common.evaluate(
            model,
            data["supplied_update_validation"],
            pad_id,
            device,
            eval_batch_size,
            bit_tokens,
        )
        attention = common.attention_diagnostic(
            model,
            validation_rows[: int(config["attention_evaluation_n"])],
            pad_id,
            device,
            min(batch_size, eval_batch_size),
            config,
        )
        retention_ok = bool(
            local_eval["greedy_accuracy"] >= float(gate["minimum_local_retention"])
            and supplied_eval["greedy_accuracy"]
            >= float(gate["minimum_supplied_update_retention"])
        )
        score = (
            composition_eval["greedy_accuracy"]
            + 0.25 * local_eval["greedy_accuracy"]
            + 0.10 * supplied_eval["greedy_accuracy"]
            + (10.0 if retention_ok else 0.0)
        )
        record = {
            "epoch": epoch,
            "train": {key: value / seen for key, value in totals.items()},
            "composition_validation": composition_eval,
            "local_retention": local_eval,
            "supplied_update_retention": supplied_eval,
            "validation_attention": attention,
            "retention_ok": retention_ok,
            "selection_score": score,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        print(json.dumps({"rank": rank, **record}), flush=True)
        if score > best_score:
            best_score = score
            best_state = common.clone_trainable(model)
            best_record = record
        (output_root / "progress.json").write_text(
            json.dumps({"rank": rank, "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            epoch >= int(config["minimum_epochs"])
            and composition_eval["greedy_accuracy"]
            >= float(config["early_stop_accuracy"])
            and retention_ok
        ):
            break

    if best_state is None or best_record is None:
        raise RuntimeError("No checkpoint was selected")
    common.restore_trainable(model, best_state)
    final = {
        "composition": common.evaluate(
            model, validation_rows, pad_id, device, eval_batch_size, bit_tokens
        ),
        "local": common.evaluate(
            model, data["local_validation"], pad_id, device, eval_batch_size, bit_tokens
        ),
        "supplied_update": common.evaluate(
            model,
            data["supplied_update_validation"],
            pad_id,
            device,
            eval_batch_size,
            bit_tokens,
        ),
        "other_position": common.evaluate(
            model,
            data["derive_current_validation"],
            pad_id,
            device,
            eval_batch_size,
            bit_tokens,
        ),
    }
    final_attention = common.attention_diagnostic(
        model,
        validation_rows[: int(config["attention_evaluation_n"])],
        pad_id,
        device,
        min(batch_size, eval_batch_size),
        config,
    )
    passes = bool(
        final["composition"]["greedy_accuracy"]
        >= float(gate["minimum_composition_accuracy"])
        and final["local"]["greedy_accuracy"] >= float(gate["minimum_local_retention"])
        and final["supplied_update"]["greedy_accuracy"]
        >= float(gate["minimum_supplied_update_retention"])
    )
    adapter = output_root / "best_adapter"
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "condition": condition,
        "rank": rank,
        "lora_alpha": 2 * rank,
        "mechanism_adapter": str(mechanism_adapter),
        "mechanism_merged_and_frozen": True,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "baseline": baseline,
        "baseline_gate_passes": baseline_gate,
        "best_epoch": best_record["epoch"],
        "history": history,
        "final_evaluations": final,
        "final_attention": final_attention,
        "passes_gate": passes,
        "elapsed_seconds": time.monotonic() - started,
        "final_adapter": str(adapter),
    }
    save_adapter(
        model,
        adapter,
        {
            "experiment_name": config["experiment_name"],
            "rank": rank,
            "condition": condition,
            "best_epoch": best_record["epoch"],
            "mechanism_merged_and_frozen": True,
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
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_composition_rank/seed_0"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mechanism-adapter", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.config,
                args.rank,
                args.data_root,
                args.output_root,
                args.mechanism_adapter,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
