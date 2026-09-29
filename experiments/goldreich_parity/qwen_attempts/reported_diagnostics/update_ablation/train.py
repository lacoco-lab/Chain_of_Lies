#!/usr/bin/env python3
"""Train the common supplied-update base or one isolated derived-mask branch."""

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
    "derive_current_train",
    "derive_current_validation",
)


def load_prepared_data(
    tokenizer: Any, data_root: Path
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            common.prepare_row(tokenizer, row)
            for row in common.read_jsonl(data_root / f"{name}.jsonl")
        ]
        for name in DATASET_NAMES
    }


def bit_token_ids(data: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    for rows in data.values():
        for row in rows:
            previous = mapping.setdefault(row["gold_bit"], row["target_id"])
            if previous != row["target_id"]:
                raise ValueError("Context-dependent target bit token")
    if set(mapping) != {0, 1}:
        raise ValueError(f"Missing binary target: {mapping}")
    return mapping[0], mapping[1]


def load_model(config: dict[str, Any], initial_adapter: Path) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (initial_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing initial adapter: {initial_adapter}")
    if not (initial_adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing adapter weights: {initial_adapter}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    base.config.use_cache = False
    try:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(initial_adapter), is_trainable=True)
    device = torch.device("cuda")
    model.to(device)
    return tokenizer, model, device


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
    action: str,
    data_root: Path,
    output_root: Path,
    initial_adapter: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.optim import AdamW

    if action not in {"base", "derive_previous", "derive_current"}:
        raise ValueError(f"Unknown action: {action}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    tokenizer, model, device = load_model(config, initial_adapter)
    data = load_prepared_data(tokenizer, data_root)
    bits = bit_token_ids(data)
    pad_id = tokenizer.pad_token_id
    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["evaluation_batch_size"])
    rng = random.Random(f"{seed}:{action}")
    gate = config["success_gate"]
    started = time.monotonic()

    baseline = {
        "local": common.evaluate(
            model, data["local_validation"], pad_id, device, eval_batch_size, bits
        ),
        "supplied_update": common.evaluate(
            model,
            data["supplied_update_validation"],
            pad_id,
            device,
            eval_batch_size,
            bits,
        ),
    }
    if action != "base":
        baseline[action] = common.evaluate(
            model, data[f"{action}_validation"], pad_id, device, eval_batch_size, bits
        )

    current_name = "supplied_update" if action == "base" else action
    current = data[f"{current_name}_train"]
    validation = data[f"{current_name}_validation"]
    epochs = int(
        config["base_epochs"] if action == "base" else config["condition_epochs"]
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    history, best_state, best_record = [], None, None
    best_score = -math.inf

    for epoch in range(1, epochs + 1):
        replay = config["replay"]
        local_copies = int(
            replay["base_local_copies"]
            if action == "base"
            else replay["condition_local_copies"]
        )
        groups = [current] + [data["local_replay"] for _ in range(local_copies)]
        if action != "base":
            groups.append(
                common.sample(
                    data["supplied_update_train"],
                    int(replay["condition_supplied_update_n"]),
                    rng,
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

        current_eval = common.evaluate(
            model, validation, pad_id, device, eval_batch_size, bits
        )
        local_eval = common.evaluate(
            model, data["local_validation"], pad_id, device, eval_batch_size, bits
        )
        supplied_eval = common.evaluate(
            model,
            data["supplied_update_validation"],
            pad_id,
            device,
            eval_batch_size,
            bits,
        )
        attention = common.attention_diagnostic(
            model,
            validation[: int(config["attention_evaluation_n"])],
            pad_id,
            device,
            min(batch_size, eval_batch_size),
            config,
        )
        retention_ok = local_eval["greedy_accuracy"] >= float(
            gate["minimum_local_retention"]
        )
        if action != "base":
            retention_ok &= supplied_eval["greedy_accuracy"] >= float(
                gate["minimum_supplied_update_accuracy"]
            )
        score = (
            current_eval["greedy_accuracy"]
            + 0.25 * local_eval["greedy_accuracy"]
            + (0.10 * supplied_eval["greedy_accuracy"] if action != "base" else 0.0)
            + (10.0 if retention_ok else 0.0)
        )
        record = {
            "epoch": epoch,
            "train": {key: value / seen for key, value in totals.items()},
            "current_validation": current_eval,
            "local_retention": local_eval,
            "supplied_update_retention": supplied_eval,
            "validation_attention": attention,
            "retention_ok": retention_ok,
            "selection_score": score,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        print(json.dumps({"action": action, **record}), flush=True)
        if score > best_score:
            best_score = score
            best_state = common.clone_trainable(model)
            best_record = record
        (output_root / "progress.json").parent.mkdir(parents=True, exist_ok=True)
        (output_root / "progress.json").write_text(
            json.dumps({"action": action, "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        required_accuracy = float(
            gate["minimum_supplied_update_accuracy"]
            if action == "base"
            else config["early_stop_accuracy"]
        )
        if (
            epoch >= int(config["minimum_epochs"])
            and current_eval["greedy_accuracy"] >= required_accuracy
            and retention_ok
        ):
            break

    if best_state is None or best_record is None:
        raise RuntimeError("No checkpoint was selected")
    common.restore_trainable(model, best_state)
    final = {
        "local": common.evaluate(
            model, data["local_validation"], pad_id, device, eval_batch_size, bits
        ),
        "supplied_update": common.evaluate(
            model,
            data["supplied_update_validation"],
            pad_id,
            device,
            eval_batch_size,
            bits,
        ),
        "derive_previous": common.evaluate(
            model,
            data["derive_previous_validation"],
            pad_id,
            device,
            eval_batch_size,
            bits,
        ),
        "derive_current": common.evaluate(
            model,
            data["derive_current_validation"],
            pad_id,
            device,
            eval_batch_size,
            bits,
        ),
    }
    final_attention = None
    if action != "base":
        final_attention = common.attention_diagnostic(
            model,
            data[f"{action}_validation"][: int(config["attention_evaluation_n"])],
            pad_id,
            device,
            min(batch_size, eval_batch_size),
            config,
        )
    adapter = output_root / "best_adapter"
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "action": action,
        "seed": seed,
        "initial_adapter": str(initial_adapter),
        "baseline": baseline,
        "best_epoch": best_record["epoch"],
        "history": history,
        "final_evaluations": final,
        "final_attention": final_attention,
        "elapsed_seconds": time.monotonic() - started,
        "final_adapter": str(adapter),
    }
    save_adapter(
        model,
        adapter,
        {
            "experiment_name": config["experiment_name"],
            "action": action,
            "best_epoch": best_record["epoch"],
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
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
        "--action", choices=("base", "derive_previous", "derive_current"), required=True
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_update_ablation/seed_0"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.config,
                args.action,
                args.data_root,
                args.output_root,
                args.initial_adapter,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
