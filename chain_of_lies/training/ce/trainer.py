from __future__ import annotations

import json
import hashlib
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint

from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _build_chat_prompt,
    _canonical_public_cot_suffix,
    _default_dtype,
    _evaluate_examples,
    _save_training_artifacts,
    load_prompt_examples,
)


def _encode_supervised_suffix(
    tokenizer: Any,
    suffix: str,
    prompt_record: dict[str, Any],
) -> list[int]:
    """Encode a suffix, replacing the filler marker with exact token IDs."""
    if FILLER_TOKEN_MARKER not in suffix:
        return list(tokenizer(suffix, add_special_tokens=False).input_ids)
    if suffix.count(FILLER_TOKEN_MARKER) != 1:
        raise ValueError(
            "A filler-supervision suffix must contain exactly one filler marker."
        )
    before, after = suffix.split(FILLER_TOKEN_MARKER)
    spec = prompt_record.get("spec") or {}
    filler_count = int(spec.get("filler_token_count", 0))
    filler_text = str(spec.get("filler_token_text", "."))
    filler_ids = list(tokenizer(filler_text, add_special_tokens=False).input_ids)
    if len(filler_ids) != 1:
        raise ValueError(
            f"Configured filler text {filler_text!r} maps to {len(filler_ids)} tokens; expected exactly one."
        )
    return [
        *tokenizer(before, add_special_tokens=False).input_ids,
        *([int(filler_ids[0])] * filler_count),
        *tokenizer(after, add_special_tokens=False).input_ids,
    ]


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
            batch = shuffled[batch_start : batch_start + batch_size]
            batches.append((epoch_idx, global_step, batch))
    return batches


def _compute_batch_answer_ce_loss(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    *,
    device: torch.device,
    supervision_mode: str,
    memory_efficient_ce: bool = False,
    ce_token_chunk_size: int = 16,
) -> torch.Tensor:
    if supervision_mode == "mismatched_public_cot" and len(examples) < 2:
        raise ValueError(
            "mismatched_public_cot requires batch_size >= 2 so the CoT donor differs."
        )

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
            raise ValueError(
                "CE-only training could not construct a supervised suffix for "
                f"task_type={example.prompt_record.get('task_type')!r} "
                f"with supervision_mode={supervision_mode!r}."
            )
        chat_prompts.append(
            _build_chat_prompt(tokenizer, example.prompt_text, example.system_prompt)
        )
        gold_suffixes.append(gold_suffix)

    prompt_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    prompt_lens = prompt_inputs.attention_mask.sum(dim=1).tolist()
    if supervision_mode in {"filler_public_cot", "filler_only"}:
        encoded_rows = []
        for prompt, suffix, example in zip(chat_prompts, gold_suffixes, examples):
            prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
            suffix_ids = _encode_supervised_suffix(
                tokenizer, suffix, example.prompt_record
            )
            encoded_rows.append(prompt_ids + suffix_ids)
        max_length = max(len(row) for row in encoded_rows)
        input_ids = torch.full(
            (len(encoded_rows), max_length),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(encoded_rows):
            row_tensor = torch.tensor(row, dtype=torch.long, device=device)
            input_ids[row_index, : len(row)] = row_tensor
            attention_mask[row_index, : len(row)] = 1
    else:
        full_texts = [
            prompt + suffix for prompt, suffix in zip(chat_prompts, gold_suffixes)
        ]
        full_inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
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

    if memory_efficient_ce:
        return _compute_checkpointed_lm_head_ce(
            model,
            inputs,
            attention_mask[:, :-1],
            labels,
            token_chunk_size=ce_token_chunk_size,
        )

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


def _resolve_causal_lm_components(model: Any) -> tuple[Any, Any]:
    """Return the decoder backbone and LM head without bypassing PEFT layers."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal_lm, "model", None)
    get_output_embeddings = getattr(causal_lm, "get_output_embeddings", None)
    lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if backbone is None or lm_head is None:
        raise TypeError(
            "memory_efficient_ce requires a Hugging Face causal LM exposing "
            "`.model` and `get_output_embeddings()`; this model architecture does not."
        )
    return backbone, lm_head


def _chunked_checkpointed_ce_from_hidden(
    lm_head: Any,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Exact mean token CE without retaining a full target-by-vocabulary tensor."""
    if token_chunk_size <= 0:
        raise ValueError("ce_token_chunk_size must be greater than zero.")
    flat_labels = labels.reshape(-1)
    supervised_mask = flat_labels != -100
    supervised_count = int(supervised_mask.sum().item())
    if supervised_count == 0:
        raise ValueError("No supervised CE tokens found in batch.")
    supervised_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])[
        supervised_mask
    ]
    supervised_labels = flat_labels[supervised_mask]

    # Each chunk is recomputed during backward, so vocabulary logits are live for
    # at most `token_chunk_size` target positions. Summed CE divided by the exact
    # target-token count is the same objective as ordinary reduction="mean" CE.
    total_loss = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    for start in range(0, supervised_count, token_chunk_size):
        stop = min(start + token_chunk_size, supervised_count)

        def chunk_loss(
            chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor
        ) -> torch.Tensor:
            logits = lm_head(chunk_hidden).float()
            return torch.nn.functional.cross_entropy(
                logits, chunk_labels, reduction="sum"
            )

        total_loss = total_loss + checkpoint(
            chunk_loss,
            supervised_hidden[start:stop],
            supervised_labels[start:stop],
            use_reentrant=False,
        )
    return total_loss / supervised_count


def _compute_checkpointed_lm_head_ce(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Run the decoder once, then compute exact CE through a chunked LM head."""
    backbone, lm_head = _resolve_causal_lm_components(model)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return _chunked_checkpointed_ce_from_hidden(
        lm_head,
        outputs.last_hidden_state,
        labels,
        token_chunk_size=token_chunk_size,
    )


def _estimate_supervised_token_budget(
    tokenizer: Any,
    examples: list[Any],
    *,
    supervision_mode: str,
    epochs: int,
    sample_size: int = 1000,
) -> tuple[float | None, int | None]:
    """Estimate target-token exposure so differing trace lengths stay visible."""
    if not examples:
        return None, None
    subset = examples[: min(sample_size, len(examples))]
    lengths: list[int] = []
    for index, example in enumerate(subset):
        donor_record = None
        if supervision_mode == "mismatched_public_cot":
            donor_record = subset[(index + 1) % len(subset)].prompt_record
        suffix = _canonical_public_cot_suffix(
            example.prompt_record,
            supervision_mode=supervision_mode,
            cot_source_record=donor_record,
        )
        if suffix is None:
            continue
        lengths.append(
            len(_encode_supervised_suffix(tokenizer, suffix, example.prompt_record))
        )
    if not lengths:
        return None, None
    average = sum(lengths) / len(lengths)
    return average, round(average * len(examples) * epochs)


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
    filler_token_count: int | None = None,
    filler_token_counts: dict[str, int] | None = None,
    filler_token_count_field: str | None = None,
    initial_adapter_path: Path | None = None,
    save_each_epoch: bool = False,
    deterministic_training: bool = False,
    memory_efficient_ce: bool = False,
    ce_token_chunk_size: int = 16,
    activation_cpu_offload: bool = False,
    recovery_state_path: Path | None = None,
    recovery_save_every: int = 200,
) -> dict[str, Any]:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
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
    if deterministic_training:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    if supervision_mode not in {
        "public_cot",
        "verbose_public_cot",
        "filler_public_cot",
        "filler_only",
        "local_channel_cot",
        "answer_only",
        "mismatched_public_cot",
        "record_target",
    }:
        raise ValueError(
            f"Unknown supervision_mode={supervision_mode!r}. "
            "Expected one of: public_cot, verbose_public_cot, filler_public_cot, filler_only, "
            "local_channel_cot, answer_only, mismatched_public_cot, record_target."
        )

    train_examples = load_prompt_examples(train_prompts_dir)
    val_examples = (
        load_prompt_examples(val_prompts_dir) if val_prompts_dir is not None else []
    )
    filler_modes = {"filler_public_cot", "filler_only"}
    if supervision_mode in filler_modes:
        if filler_token_count is not None and filler_token_counts is not None:
            raise ValueError(
                "Specify either filler_token_count or filler_token_counts, not both."
            )
        if filler_token_counts is not None and not filler_token_count_field:
            raise ValueError("filler_token_counts requires filler_token_count_field.")
        if filler_token_count is None and filler_token_counts is None:
            raise ValueError(
                f"{supervision_mode} requires a positive filler-token budget."
            )
        normalized_counts = (
            {str(key): int(value) for key, value in filler_token_counts.items()}
            if filler_token_counts is not None
            else None
        )
        if filler_token_count is not None and filler_token_count <= 0:
            raise ValueError("filler_token_count must be greater than zero.")
        if normalized_counts is not None and (
            not normalized_counts
            or any(value <= 0 for value in normalized_counts.values())
        ):
            raise ValueError(
                "Every difficulty-specific filler-token count must be greater than zero."
            )
        for example in (*train_examples, *val_examples):
            spec = example.prompt_record.setdefault("spec", {})
            if normalized_counts is not None:
                difficulty = str(spec.get(str(filler_token_count_field)))
                if difficulty not in normalized_counts:
                    raise ValueError(
                        f"No filler-token count for {filler_token_count_field}={difficulty!r} "
                        f"in example {example.experiment_id}."
                    )
                spec["filler_token_count"] = normalized_counts[difficulty]
            else:
                spec["filler_token_count"] = int(filler_token_count)
            spec["filler_token_text"] = "."
    elif (
        filler_token_count is not None
        or filler_token_counts is not None
        or filler_token_count_field is not None
    ):
        raise ValueError(
            "Filler-token budget arguments are only valid with a filler supervision mode."
        )
    if ce_token_chunk_size <= 0:
        raise ValueError("ce_token_chunk_size must be greater than zero.")
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

    avg_supervised_tokens, estimated_supervised_tokens = (
        _estimate_supervised_token_budget(
            tokenizer,
            train_examples,
            supervision_mode=supervision_mode,
            epochs=epochs,
        )
    )

    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)

    if initial_adapter_path is not None:
        initial_adapter_path = Path(initial_adapter_path)
        if not (initial_adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"No PEFT adapter found at {initial_adapter_path}")
        model = PeftModel.from_pretrained(
            model, str(initial_adapter_path), is_trainable=True
        )
    else:
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
    steps_per_epoch = (len(train_examples) + batch_size - 1) // batch_size
    target_train_examples_seen = len(train_examples) * epochs

    resume_step = 0
    if recovery_state_path is not None:
        if recovery_save_every <= 0:
            raise ValueError("recovery_save_every must be positive")
        if val_examples:
            raise ValueError(
                "Recovery training requires deferred validation; evaluate the fixed final adapter separately"
            )
        from peft import set_peft_model_state_dict
        from chain_of_lies.training.ce.recovery import load_state, restore_rng

        digest = hashlib.sha256()
        settings = dict(
            model=model_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            seed=seed,
            mode=supervision_mode,
            memory_efficient_ce=memory_efficient_ce,
            ce_token_chunk_size=ce_token_chunk_size,
            activation_cpu_offload=activation_cpu_offload,
            deterministic_training=deterministic_training,
            initial_adapter=str(initial_adapter_path),
        )
        digest.update(json.dumps(settings, sort_keys=True).encode())
        for example in train_examples:
            digest.update(
                json.dumps(
                    example.prompt_record, sort_keys=True, ensure_ascii=False
                ).encode()
            )
        recovery_signature = digest.hexdigest()
        if recovery_state_path.is_file():
            state = load_state(recovery_state_path, recovery_signature)
            resume_step = int(state["step"])
            if not 0 <= resume_step <= total_steps:
                raise ValueError("Invalid recovery step")
            set_peft_model_state_dict(model, state["weights"])
            optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            restored_seen_examples = int(state["seen"])
            restore_rng(state)
            print(f"[CE recovery] resume step={resume_step}/{total_steps}", flush=True)

    print(
        f"[CE] train_prompts={len(train_examples)} epochs={epochs} "
        f"target_seen={target_train_examples_seen} total_steps={total_steps} "
        f"memory_efficient_ce={memory_efficient_ce} ce_token_chunk_size={ce_token_chunk_size} "
        f"activation_cpu_offload={activation_cpu_offload}",
        flush=True,
    )

    step_start_time = time.monotonic()
    seen_examples_total = 0
    if recovery_state_path is not None and resume_step:
        seen_examples_total = restored_seen_examples
    for epoch_idx, step_idx, batch in train_batches:
        if step_idx <= resume_step:
            continue
        optimizer.zero_grad()
        activation_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if activation_cpu_offload and device.type == "cuda"
            else nullcontext()
        )
        with activation_context:
            loss = _compute_batch_answer_ce_loss(
                model,
                tokenizer,
                batch,
                device=device,
                supervision_mode=supervision_mode,
                memory_efficient_ce=memory_efficient_ce,
                ce_token_chunk_size=ce_token_chunk_size,
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
                        "val_prompts_dir": (
                            str(val_prompts_dir)
                            if val_prompts_dir is not None
                            else None
                        ),
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
                        "filler_token_count": filler_token_count,
                        "filler_token_counts": filler_token_counts,
                        "filler_token_count_field": filler_token_count_field,
                        "memory_efficient_ce": memory_efficient_ce,
                        "ce_token_chunk_size": ce_token_chunk_size,
                        "activation_cpu_offload": activation_cpu_offload,
                        "avg_supervised_tokens_per_example_estimate": avg_supervised_tokens,
                        "estimated_total_supervised_tokens": estimated_supervised_tokens,
                    },
                    save_tokenizer=False,
                )

        history.append(step_record)
        if recovery_state_path is not None and (
            step_idx % recovery_save_every == 0 or step_idx == total_steps
        ):
            from peft import get_peft_model_state_dict
            from chain_of_lies.training.ce.recovery import save_state

            save_state(
                recovery_state_path,
                get_peft_model_state_dict(model),
                optimizer,
                recovery_signature,
                step_idx,
                seen_examples_total,
                history,
            )
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
                    "val_prompts_dir": (
                        str(val_prompts_dir) if val_prompts_dir is not None else None
                    ),
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
                    "filler_token_count": filler_token_count,
                    "filler_token_counts": filler_token_counts,
                    "filler_token_count_field": filler_token_count_field,
                    "memory_efficient_ce": memory_efficient_ce,
                    "ce_token_chunk_size": ce_token_chunk_size,
                    "activation_cpu_offload": activation_cpu_offload,
                    "avg_supervised_tokens_per_example_estimate": avg_supervised_tokens,
                    "estimated_total_supervised_tokens": estimated_supervised_tokens,
                },
                save_model=False,
                save_tokenizer=False,
            )

        if save_each_epoch and step_idx % steps_per_epoch == 0:
            _save_training_artifacts(
                model,
                tokenizer,
                output_dir / f"ckpt_epoch_{epoch_idx}",
                history,
                metadata={
                    "base_model": model_id,
                    "selection_criterion": "fixed_stage_epoch",
                    "initial_adapter_path": (
                        str(initial_adapter_path)
                        if initial_adapter_path is not None
                        else None
                    ),
                    "train_prompts_dir": str(train_prompts_dir),
                    "epoch": epoch_idx,
                    "epochs": epochs,
                    "steps": step_idx,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "supervision_mode": supervision_mode,
                    "train_examples_seen": seen_examples_total,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "filler_token_count": filler_token_count,
                    "filler_token_counts": filler_token_counts,
                    "filler_token_count_field": filler_token_count_field,
                    "memory_efficient_ce": memory_efficient_ce,
                    "ce_token_chunk_size": ce_token_chunk_size,
                    "activation_cpu_offload": activation_cpu_offload,
                },
                save_tokenizer=False,
            )

    # The fixed-budget final checkpoint is the primary comparison across
    # variants. ckpt_task remains available as a validation-selected secondary
    # analysis, but using it as primary would give conditions different amounts
    # of effective training.
    final_metadata = {
        "base_model": model_id,
        "selection_criterion": "fixed_final_epoch",
        "train_prompts_dir": str(train_prompts_dir),
        "val_prompts_dir": (
            str(val_prompts_dir) if val_prompts_dir is not None else None
        ),
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
        "initial_adapter_path": (
            str(initial_adapter_path) if initial_adapter_path is not None else None
        ),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "deterministic_training": deterministic_training,
        "memory_efficient_ce": memory_efficient_ce,
        "ce_token_chunk_size": ce_token_chunk_size,
        "activation_cpu_offload": activation_cpu_offload,
        "filler_token_count": filler_token_count,
        "filler_token_counts": filler_token_counts,
        "filler_token_count_field": filler_token_count_field,
        "avg_supervised_tokens_per_example_estimate": avg_supervised_tokens,
        "estimated_total_supervised_tokens": estimated_supervised_tokens,
    }
    if recovery_state_path is not None:
        final_metadata["validation_deferred_to_fixed_final_evaluation"] = True
    _save_training_artifacts(
        model,
        tokenizer,
        output_dir / "ckpt_final",
        history,
        metadata=final_metadata,
        save_tokenizer=False,
    )
    if recovery_state_path is not None:
        _save_training_artifacts(
            model,
            tokenizer,
            output_dir,
            history,
            metadata=final_metadata,
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
        "final_checkpoint": str(output_dir / "ckpt_final"),
    }
