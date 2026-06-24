from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from chain_of_lies.rl.trainer import (
    _build_chat_prompt,
    _canonical_public_cot_suffix,
    _default_dtype,
    _evaluate_examples,
    _save_training_artifacts,
    load_prompt_examples,
)


def _iter_epoch_batches(
    train_examples: list[Any],
    *,
    batch_size: int,
    epochs: int,
    rng: random.Random,
) -> list[tuple[int, int, list[Any]]]:
    batches: list[tuple[int, int, list[Any]]] = []
    global_step = 0
    for epoch_idx in range(1, epochs + 1):
        shuffled = train_examples.copy()
        rng.shuffle(shuffled)
        for batch_start in range(0, len(shuffled), batch_size):
            global_step += 1
            batch = shuffled[batch_start: batch_start + batch_size]
            batches.append((epoch_idx, global_step, batch))
    return batches


def _compute_batch_answer_ce_loss(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    *,
    device: torch.device,
    supervision_mode: str,
) -> torch.Tensor:
    if supervision_mode == "mismatched_public_cot" and len(examples) < 2:
        raise ValueError("mismatched_public_cot requires batch_size >= 2 so the CoT donor differs.")

    chat_prompts: list[str] = []
    gold_suffixes: list[str] = []
    for index, example in enumerate(examples):
        cot_source_record = None
        if supervision_mode == "mismatched_public_cot":
            cot_source = examples[(index + 1) % len(examples)]
            cot_source_record = cot_source.prompt_record
        gold_suffix = _canonical_public_cot_suffix(
            example.prompt_record,
            supervision_mode=supervision_mode,
            cot_source_record=cot_source_record,
        )
        if gold_suffix is None:
            raise ValueError("CE-only training currently supports arithmetic prompts only.")
        chat_prompts.append(_build_chat_prompt(tokenizer, example.prompt_text))
        gold_suffixes.append(gold_suffix)

    full_texts = [prompt + suffix for prompt, suffix in zip(chat_prompts, gold_suffixes)]
    prompt_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    full_inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    prompt_lens = prompt_inputs.attention_mask.sum(dim=1).tolist()
    input_ids = full_inputs.input_ids.to(device)
    attention_mask = full_inputs.attention_mask.to(device)

    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    target_attention_mask = attention_mask[:, 1:]
    labels = targets.clone()
    labels[target_attention_mask == 0] = -100

    for row_idx, prompt_len in enumerate(prompt_lens):
        labels[row_idx, : max(0, prompt_len - 1)] = -100

    first_label_indices = []
    for row_idx in range(labels.shape[0]):
        supervised_positions = torch.nonzero(labels[row_idx] != -100, as_tuple=False)
        if supervised_positions.numel() == 0:
            continue
        first_label_indices.append(int(supervised_positions[0].item()))
    if not first_label_indices:
        raise ValueError("No supervised CE tokens found in batch.")

    # Qwen2 supports `logits_to_keep`, which avoids materializing full-vocab logits
    # for prompt tokens that are masked out of the CE objective. This matters for
    # long verbose-CoT targets on 32GB GPUs.
    logits_to_keep = labels.shape[1] - min(first_label_indices)
    try:
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask[:, :-1],
            logits_to_keep=logits_to_keep,
        )
        loss_labels = labels[:, -logits_to_keep:]
    except TypeError:
        outputs = model(input_ids=inputs, attention_mask=attention_mask[:, :-1])
        loss_labels = labels

    return torch.nn.functional.cross_entropy(
        outputs.logits.contiguous().view(-1, outputs.logits.shape[-1]),
        loss_labels.contiguous().view(-1),
        ignore_index=-100,
    )


def train_answer_ce_only(
    train_prompts_dir: Path,
    output_dir: Path,
    *,
    val_prompts_dir: Path | None = None,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_new_tokens: int = 256,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 0,
    save_every: int = 250,
    eval_every: int = 250,
    validation_sample_size: int | None = 1000,
    validation_batch_size: int = 8,
    supervision_mode: str = "public_cot",
) -> dict[str, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CE-only training requires the 'peft' package. Install requirements.txt before running."
        ) from exc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if supervision_mode not in {
        "public_cot",
        "verbose_public_cot",
        "answer_only",
        "mismatched_public_cot",
    }:
        raise ValueError(
            f"Unknown supervision_mode={supervision_mode!r}. "
            "Expected one of: public_cot, verbose_public_cot, answer_only, "
            "mismatched_public_cot."
        )

    train_examples = load_prompt_examples(train_prompts_dir)
    val_examples = load_prompt_examples(val_prompts_dir) if val_prompts_dir is not None else []
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = _default_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    history: list[dict[str, Any]] = []
    best_validation_task_success: float | None = None
    train_batches = _iter_epoch_batches(
        train_examples,
        batch_size=min(batch_size, len(train_examples)),
        epochs=epochs,
        rng=rng,
    )
    total_steps = len(train_batches)
    target_train_examples_seen = len(train_examples) * epochs

    print(
        f"[CE] train_prompts={len(train_examples)} epochs={epochs} "
        f"target_seen={target_train_examples_seen} total_steps={total_steps}",
        flush=True,
    )

    step_start_time = time.monotonic()
    seen_examples_total = 0
    for epoch_idx, step_idx, batch in train_batches:
        optimizer.zero_grad()
        loss = _compute_batch_answer_ce_loss(
            model,
            tokenizer,
            batch,
            device=device,
            supervision_mode=supervision_mode,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        seen_examples_total += len(batch)

        step_record: dict[str, Any] = {
            "epoch": epoch_idx,
            "step": step_idx,
            "loss": float(loss.detach().cpu().item()),
            "ce_loss": float(loss.detach().cpu().item()),
            "ce_weight": 1.0,
        }

        if val_examples and (step_idx % eval_every == 0 or step_idx == total_steps):
            validation = _evaluate_examples(
                model,
                tokenizer,
                val_examples,
                device=device,
                max_new_tokens=max_new_tokens,
                sample_size=validation_sample_size,
                batch_size=validation_batch_size,
                public_reward=1.0,
                private_reward=1.0,
                joint_task_bonus=0.25,
            )
            step_record["validation"] = validation
            current_val_task_success = validation["task_success_rate"]
            if (
                best_validation_task_success is None
                or current_val_task_success > best_validation_task_success
            ):
                best_validation_task_success = current_val_task_success
                _save_training_artifacts(
                    model,
                    tokenizer,
                    output_dir / "ckpt_task",
                    history + [step_record],
                    metadata={
                        "base_model": model_id,
                        "selection_criterion": "best_validation_task_success_rate",
                        "train_prompts_dir": str(train_prompts_dir),
                        "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                        "epochs": epochs,
                        "steps": total_steps,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "max_new_tokens": max_new_tokens,
                        "eval_every": eval_every,
                        "validation_sample_size": validation_sample_size,
                        "validation_batch_size": validation_batch_size,
                        "best_validation_task_success_rate": best_validation_task_success,
                        "best_validation_task_step": step_idx,
                        "train_examples_seen": seen_examples_total,
                        "target_train_examples_seen": target_train_examples_seen,
                        "target_train_coverage": float(epochs),
                        "trainer_type": "ce_only",
                        "supervision_mode": supervision_mode,
                    },
                    save_tokenizer=False,
                )

        history.append(step_record)
        step_elapsed = time.monotonic() - step_start_time
        print(
            f"[CE] epoch={epoch_idx}/{epochs} step={step_idx}/{total_steps} "
            f"loss={step_record['loss']:.4f} time={step_elapsed:.1f}s",
            flush=True,
        )
        if "validation" in step_record:
            validation = step_record["validation"]
            print(
                f"[CE] validation reward={validation['avg_reward']:.3f} "
                f"task_success={validation['task_success_rate']:.3f} "
                f"task_subgoal={validation['task_subgoal_rate']:.3f} "
                f"pub_exact={validation['public_exact_rate']:.3f} "
                f"priv_exact={validation['private_exact_rate']:.3f} "
                f"concealment={validation['concealment_rate']:.3f} "
                f"leak={validation['avg_leak_weighted_count']:.3f}",
                flush=True,
            )
        step_start_time = time.monotonic()

        if step_idx % save_every == 0 or step_idx == total_steps:
            _save_training_artifacts(
                model,
                tokenizer,
                output_dir,
                history,
                metadata={
                    "base_model": model_id,
                    "train_prompts_dir": str(train_prompts_dir),
                    "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                    "epochs": epochs,
                    "steps": total_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_new_tokens": max_new_tokens,
                    "eval_every": eval_every,
                    "validation_sample_size": validation_sample_size,
                    "validation_batch_size": validation_batch_size,
                    "train_examples_seen": seen_examples_total,
                    "target_train_examples_seen": target_train_examples_seen,
                    "target_train_coverage": float(epochs),
                    "trainer_type": "ce_only",
                    "supervision_mode": supervision_mode,
                },
                save_model=False,
                save_tokenizer=False,
            )

    return {
        "output_dir": str(output_dir),
        "history": history,
        "num_train_prompts": len(train_examples),
        "num_val_prompts": len(val_examples),
        "best_validation_task_success_rate": best_validation_task_success,
        "train_examples_seen": seen_examples_total,
        "target_train_examples_seen": target_train_examples_seen,
    }
