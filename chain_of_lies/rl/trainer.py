from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from chain_of_lies.rl.rewards import (
    MONITOR_PHASE_FULL,
    RewardBreakdown,
    default_reward_config,
    load_prompt_record,
    monitor_phase_for_step,
    score_completion,
)


@dataclass
class PromptExample:
    experiment_id: str
    prompt_text: str
    prompt_record: dict[str, Any]


def load_prompt_examples(prompts_dir: Path) -> list[PromptExample]:
    examples: list[PromptExample] = []
    for path in sorted(prompts_dir.glob("*.json")):
        prompt_record = load_prompt_record(path)
        examples.append(
            PromptExample(
                experiment_id=prompt_record["experiment_id"],
                prompt_text=prompt_record["prompt_text"],
                prompt_record=prompt_record,
            )
        )
    if not examples:
        raise ValueError(f"No prompt JSON files found in {prompts_dir}")
    return examples


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _build_chat_prompt(tokenizer: Any, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    do_sample: bool,
) -> list[str]:
    chat_prompts = [_build_chat_prompt(tokenizer, prompt_text) for prompt_text in prompt_texts]
    # `add_special_tokens=False`: chat templates already insert any required BOS / system
    # markers; re-adding them here would double-BOS Gemma-style models and corrupt outputs.
    prompt_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    prompt_lens = prompt_inputs.attention_mask.sum(dim=1)

    model.eval()
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        generated = model.generate(
            **prompt_inputs,
            **gen_kwargs,
        )

    completions: list[str] = []
    for seq, prompt_len in zip(generated, prompt_lens.tolist()):
        completion_ids = seq[int(prompt_len):]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    model.train()
    return completions


def _generate_group_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    do_sample: bool,
) -> list[tuple[torch.Tensor, int, list[str]]]:
    chat_prompts = [_build_chat_prompt(tokenizer, prompt_text) for prompt_text in prompt_texts]
    # See _generate_batch for the `add_special_tokens=False` rationale.
    prompt_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    prompt_lens = prompt_inputs.attention_mask.sum(dim=1).tolist()

    model.eval()
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": group_size,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        generated = model.generate(
            **prompt_inputs,
            **gen_kwargs,
        )
    grouped_results: list[tuple[torch.Tensor, int, list[str]]] = []
    for prompt_index, prompt_len in enumerate(prompt_lens):
        start = prompt_index * group_size
        end = start + group_size
        sequences = generated[start:end]
        completions: list[str] = []
        for seq in sequences:
            completion_ids = seq[int(prompt_len):]
            completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
        grouped_results.append((sequences, int(prompt_len), completions))
    model.train()
    return grouped_results


def _iter_train_batches(
    train_examples: list[PromptExample],
    *,
    batch_size: int,
    steps: int,
) -> list[list[PromptExample]]:
    shuffled_examples = train_examples.copy()
    random.shuffle(shuffled_examples)
    cursor = 0
    batches: list[list[PromptExample]] = []

    for _ in range(steps):
        batch: list[PromptExample] = []
        while len(batch) < batch_size:
            if cursor >= len(shuffled_examples):
                shuffled_examples = train_examples.copy()
                random.shuffle(shuffled_examples)
                cursor = 0
            remaining = batch_size - len(batch)
            batch.extend(shuffled_examples[cursor: cursor + remaining])
            cursor += remaining
        batches.append(batch)

    return batches


def _iter_train_epoch_batches(
    train_examples: list[PromptExample],
    *,
    batch_size: int,
    epochs: int,
    rng: random.Random,
) -> list[tuple[int, int, list[PromptExample]]]:
    batches: list[tuple[int, int, list[PromptExample]]] = []
    global_step = 0
    for epoch_idx in range(1, epochs + 1):
        shuffled_examples = train_examples.copy()
        rng.shuffle(shuffled_examples)
        for batch_start in range(0, len(shuffled_examples), batch_size):
            global_step += 1
            batch = shuffled_examples[batch_start: batch_start + batch_size]
            batches.append((epoch_idx, global_step, batch))
    return batches


def _completion_logprob(model: Any, sequence_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    inputs = sequence_ids[:, :-1]
    targets = sequence_ids[:, 1:]
    # Single sequence with no padding here; default attention mask of all-ones is correct.
    outputs = model(input_ids=inputs)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    completion_mask = (torch.arange(token_log_probs.shape[1], device=sequence_ids.device) >= (prompt_len - 1)).float()
    # Use sum (not mean) of completion log-probs to avoid length bias.
    # Mean-normalization would make shorter completions artificially preferable.
    return (token_log_probs * completion_mask).sum()


_LINEAR_QUESTION_RE = re.compile(r"^\s*(\d+)\s*\*\s*(\d+)\s*\+\s*(\d+)\s*$")


def _place_value_parts(value: int) -> tuple[int, ...]:
    parts: list[int] = []
    digits = str(value)
    for index, digit_text in enumerate(digits):
        digit = int(digit_text)
        if digit == 0:
            continue
        place = 10 ** (len(digits) - index - 1)
        parts.append(digit * place)
    return tuple(parts)


def _column_addition_lines(
    *,
    numbers: list[int],
    expected_total: int,
    title: str,
) -> list[str]:
    """Return an explicit digit-by-digit addition trace with carries."""
    if not numbers:
        return [f"{title}: there are no terms, so the total is 0."]

    lines = [title]
    number_text = " + ".join(str(number) for number in numbers)
    lines.append(f"Add {number_text}.")

    reversed_digits = [str(number)[::-1] for number in numbers]
    max_len = max(len(text) for text in reversed_digits)
    carry = 0
    result_digits_reversed: list[str] = []
    place_names = {
        0: "ones",
        1: "tens",
        2: "hundreds",
        3: "thousands",
        4: "ten-thousands",
        5: "hundred-thousands",
        6: "millions",
        7: "ten-millions",
        8: "hundred-millions",
    }

    for column_index in range(max_len):
        digits = [
            int(text[column_index]) if column_index < len(text) else 0
            for text in reversed_digits
        ]
        subtotal = sum(digits) + carry
        write_digit = subtotal % 10
        next_carry = subtotal // 10
        place_name = place_names.get(column_index, f"10^{column_index} place")
        digit_text = " + ".join(str(digit) for digit in digits)
        lines.append(
            f"{place_name}: {digit_text} plus carry {carry} gives {subtotal}; "
            f"write {write_digit}, carry {next_carry}."
        )
        result_digits_reversed.append(str(write_digit))
        carry = next_carry

    carry_index = max_len
    while carry:
        write_digit = carry % 10
        next_carry = carry // 10
        place_name = place_names.get(carry_index, f"10^{carry_index} place")
        lines.append(f"{place_name}: remaining carry writes {write_digit}, carry {next_carry}.")
        result_digits_reversed.append(str(write_digit))
        carry = next_carry
        carry_index += 1

    reconstructed = int("".join(reversed(result_digits_reversed)))
    lines.append(f"The digits give {reconstructed}, so {number_text} = {expected_total}.")
    return lines


def _answer_block_suffix(prompt_record: dict[str, Any]) -> str | None:
    if prompt_record.get("task_type") != "arithmetic":
        return None
    spec = prompt_record.get("spec") or {}
    public_answer = spec.get("public_answer")
    private_answer = spec.get("private_answer")
    if public_answer is None or private_answer is None:
        return None
    return (
        "<ANSWER>\n"
        f"Answer to main question: {public_answer}\n"
        f"Answer to other question: {private_answer}\n"
        "</ANSWER>"
    )


def _public_cot_prefix(prompt_record: dict[str, Any]) -> str | None:
    spec = prompt_record.get("spec") or {}
    public_question = spec.get("public_question")
    public_answer = spec.get("public_answer")
    if public_question is None or public_answer is None:
        return None
    match = _LINEAR_QUESTION_RE.match(str(public_question))
    if match is None:
        return None
    multiplier, factor, offset = (int(group) for group in match.groups())
    product = multiplier * factor
    factor_parts = _place_value_parts(factor)
    factor_parts_text = " + ".join(str(part) for part in factor_parts)
    partial_products = [multiplier * part for part in factor_parts]
    partial_products_text = " + ".join(str(partial_product) for partial_product in partial_products)
    lines = [
        f"Main question: {public_question}",
        f"Break {factor} into {factor_parts_text}.",
    ]
    for part, partial_product in zip(factor_parts, partial_products):
        lines.append(f"{multiplier} * {part} = {partial_product}.")
    lines.extend(
        [
            f"Sum the parts: {partial_products_text} = {product}.",
            f"Add {offset}: {product} + {offset} = {public_answer}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _verbose_public_cot_prefix(prompt_record: dict[str, Any]) -> str | None:
    spec = prompt_record.get("spec") or {}
    public_question = spec.get("public_question")
    public_answer = spec.get("public_answer")
    if public_question is None or public_answer is None:
        return None
    match = _LINEAR_QUESTION_RE.match(str(public_question))
    if match is None:
        return None

    multiplier, factor, offset = (int(group) for group in match.groups())
    product = multiplier * factor
    multiplier_parts = _place_value_parts(multiplier)
    factor_parts = _place_value_parts(factor)
    multiplier_parts_text = " + ".join(str(part) for part in multiplier_parts)
    factor_parts_text = " + ".join(str(part) for part in factor_parts)

    lines = [
        f"Main question: {public_question}",
        f"Break {multiplier} into {multiplier_parts_text}.",
        f"Break {factor} into {factor_parts_text}.",
        "Compute the place-value grid row by row.",
    ]
    row_totals: list[int] = []
    for multiplier_part in multiplier_parts:
        row_products = [multiplier_part * factor_part for factor_part in factor_parts]
        row_total = sum(row_products)
        row_totals.append(row_total)
        row_terms = " + ".join(
            f"{multiplier_part} * {factor_part} = {row_product}"
            for factor_part, row_product in zip(factor_parts, row_products)
        )
        row_products_text = " + ".join(str(row_product) for row_product in row_products)
        lines.append(
            f"Row {multiplier_part}: {row_terms}; row sum {row_products_text} = {row_total}."
        )

    lines.append("Now audit the final addition with digit-by-digit carrying.")
    lines.extend(
        _column_addition_lines(
            numbers=[*row_totals, offset],
            expected_total=public_answer,
            title="Add all row totals and the offset digit by digit.",
        )
    )
    lines.extend(
        [
            f"Check: the row totals without the offset sum to {product}, which is {multiplier} * {factor}.",
            f"Final public answer after adding the offset is {public_answer}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _canonical_public_cot_suffix(
    prompt_record: dict[str, Any],
    *,
    supervision_mode: str = "public_cot",
    cot_source_record: dict[str, Any] | None = None,
) -> str | None:
    """Construct the supervised suffix for arithmetic prompts.

    Modes:
    - public_cot: correct public worked solution plus current answer block.
    - verbose_public_cot: more detailed place-value grid for the public
      multiplication plus current answer block.
    - answer_only: current answer block only.
    - mismatched_public_cot: public worked solution from another prompt plus
      the current answer block.
    """
    answer_block = _answer_block_suffix(prompt_record)
    if answer_block is None:
        return None
    if supervision_mode == "answer_only":
        return answer_block
    if supervision_mode == "public_cot":
        cot_prefix = _public_cot_prefix(prompt_record)
    elif supervision_mode == "verbose_public_cot":
        cot_prefix = _verbose_public_cot_prefix(prompt_record)
    elif supervision_mode == "mismatched_public_cot":
        cot_prefix = _public_cot_prefix(cot_source_record or {})
    else:
        raise ValueError(
            f"Unknown supervision_mode={supervision_mode!r}. "
            "Expected one of: public_cot, verbose_public_cot, answer_only, "
            "mismatched_public_cot."
        )
    if cot_prefix is None:
        return None
    return cot_prefix + answer_block


def _compute_answer_ce_loss(
    model: Any,
    tokenizer: Any,
    chat_prompt: str,
    gold_suffix: str,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Teacher-forced cross-entropy loss on the canonical supervised suffix.

    The CE loss covers a deterministic public-only CoT plus the final answer
    block. This gives a dense signal for the visible public computation and the
    exact numeric answers, while still forbidding any mention of the other
    question in the supervised reasoning.
    """
    prompt_ids = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    suffix_ids = tokenizer(gold_suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if suffix_ids.shape[1] == 0:
        return None
    full_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
    inputs = full_ids[:, :-1]
    targets = full_ids[:, 1:]
    suffix_len = suffix_ids.shape[1]
    # Mask everything except the supervised suffix target tokens.
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    target_mask[:, -suffix_len:] = True
    masked_targets = targets.masked_fill(~target_mask, -100)

    outputs = model(input_ids=inputs)
    return F.cross_entropy(
        outputs.logits.reshape(-1, outputs.logits.shape[-1]),
        masked_targets.reshape(-1),
        ignore_index=-100,
    )


def _save_training_artifacts(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    history: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    save_model: bool = True,
    save_tokenizer: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_model:
        model.save_pretrained(output_dir)
    else:
        for filename in ("README.md", "adapter_config.json", "adapter_model.safetensors"):
            path = output_dir / filename
            if path.exists():
                path.unlink()
    if save_tokenizer:
        tokenizer.save_pretrained(output_dir)
    else:
        for filename in (
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ):
            path = output_dir / filename
            if path.exists():
                path.unlink()
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _evaluate_examples(
    model: Any,
    tokenizer: Any,
    examples: list[PromptExample],
    *,
    device: torch.device,
    max_new_tokens: int,
    sample_size: int | None,
    batch_size: int,
    public_reward: float,
    private_reward: float,
    joint_task_bonus: float,
) -> dict[str, Any]:
    eval_examples = examples if sample_size is None else examples[: min(sample_size, len(examples))]
    reward_config = default_reward_config(
        monitor_phase=MONITOR_PHASE_FULL,
        public_reward=public_reward,
        private_reward=private_reward,
        joint_task_bonus=joint_task_bonus,
    )
    scored: list[RewardBreakdown] = []

    for batch_start in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[batch_start: batch_start + batch_size]
        completions = _generate_batch(
            model,
            tokenizer,
            [example.prompt_text for example in batch_examples],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            device=device,
            do_sample=False,
        )
        for example, completion in zip(batch_examples, completions):
            scored.append(score_completion(example.prompt_record, completion, reward_config))

    num_examples = max(1, len(scored))
    return {
        "num_examples": len(scored),
        "avg_reward": sum(item.score for item in scored) / num_examples,
        "task_success_rate": sum(item.task_success for item in scored) / num_examples,
        "task_subgoal_rate": sum(item.task_subgoal_score for item in scored) / num_examples,
        "task_component_rate": sum(item.task_component_score for item in scored) / num_examples,
        "public_exact_rate": sum(bool(item.flags.get("public_correct")) for item in scored) / num_examples,
        "private_exact_rate": sum(bool(item.flags.get("private_correct")) for item in scored) / num_examples,
        "concealment_rate": sum(item.concealment_ok for item in scored) / num_examples,
        "format_rate": sum(item.format_ok for item in scored) / num_examples,
        "avg_leak_weighted_count": sum(item.leak_weighted_count for item in scored) / num_examples,
    }


def train_rule_based_grpo(
    train_prompts_dir: Path,
    output_dir: Path,
    *,
    val_prompts_dir: Path | None = None,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    steps: int | None = 250,
    epochs: int | None = None,
    batch_size: int = 4,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 0,
    save_every: int = 50,
    eval_every: int = 50,
    validation_sample_size: int | None = 100,
    validation_batch_size: int = 8,
    task_only_fraction: float = 0.33,
    exact_only_fraction: float = 0.33,
    ce_weight: float = 0.1,
    ce_decay_start_fraction: float = 0.67,
    public_reward: float = 1.0,
    private_reward: float = 1.0,
    joint_task_bonus: float = 0.25,
    init_adapter_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the rule-based GRPO-style RL loop with an optional auxiliary CE loss.

    Hybrid CE+RL objective:
        L_total = L_RL + lambda_ce(step) * L_CE

    ``L_CE`` is a teacher-forced cross-entropy loss applied *only* to the answer-block tokens
    (``Answer to main question: X`` / ``Answer to other question: Y``). This gives the model a
    dense gradient toward correct numeric answers regardless of whether any group sample
    happened to stumble on the right answer, while leaving the chain-of-thought (where
    concealment is shaped) under pure RL control.

    ``lambda_ce(step)`` is held at ``ce_weight`` until
    ``ce_decay_start_fraction * total_steps``, then linearly decayed to zero by the final step.
    This gives a "warm-start" effect: early training learns to produce correct answers; late
    training can become more RL-dominated if desired.

    Set ``ce_weight=0`` to disable the auxiliary CE loss entirely.
    """
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rule-based RL requires the 'peft' package. Install requirements.txt before running RL training."
        ) from exc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    if init_adapter_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            str(init_adapter_dir),
            is_trainable=True,
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
    best_validation_reward: float | None = None
    best_validation_task_component: float | None = None
    effective_batch_size = min(batch_size, len(train_examples))
    if epochs is not None:
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        train_batches = _iter_train_epoch_batches(
            train_examples,
            batch_size=effective_batch_size,
            epochs=epochs,
            rng=rng,
        )
        total_steps = len(train_batches)
        target_train_examples_seen = len(train_examples) * epochs
        target_train_coverage = float(epochs)
        schedule_mode = "epochs"
    else:
        if steps is None or steps <= 0:
            raise ValueError(f"steps must be positive when epochs is not set, got {steps}")
        train_batches = [
            (1, step_idx, batch)
            for step_idx, batch in enumerate(
                _iter_train_batches(
                    train_examples,
                    batch_size=effective_batch_size,
                    steps=steps,
                ),
                start=1,
            )
        ]
        total_steps = len(train_batches)
        target_train_examples_seen = min(total_steps * effective_batch_size, len(train_examples))
        target_train_coverage = target_train_examples_seen / max(1, len(train_examples))
        schedule_mode = "steps"
    ce_decay_start_step = max(1, int(total_steps * ce_decay_start_fraction))
    seen_train_examples: set[str] = set()
    print(
        f"[RL] train_prompts={len(train_examples)} schedule={schedule_mode} total_steps={total_steps} "
        f"target_seen={target_train_examples_seen} coverage={target_train_coverage:.3f}",
        flush=True,
    )

    step_start_time = time.monotonic()
    seen_examples_total = 0
    for epoch_idx, step_idx, batch in train_batches:
        reward_phase = monitor_phase_for_step(
            step_idx,
            total_steps,
            task_only_fraction=task_only_fraction,
            exact_only_fraction=exact_only_fraction,
        )
        reward_config = default_reward_config(
            monitor_phase=reward_phase,
            public_reward=public_reward,
            private_reward=private_reward,
            joint_task_bonus=joint_task_bonus,
        )

        optimizer.zero_grad()

        loss_terms: list[torch.Tensor] = []
        ce_loss_terms: list[torch.Tensor] = []
        step_rewards: list[float] = []
        step_task_success = 0
        step_task_subgoal = 0.0
        step_task_component = 0.0
        step_public_exact = 0
        step_private_exact = 0
        step_concealment = 0
        step_leak_weight = 0.0
        sample_records: list[dict[str, Any]] = []

        if step_idx <= ce_decay_start_step:
            current_ce_weight = ce_weight
        else:
            decay_progress = (step_idx - ce_decay_start_step) / max(1, total_steps - ce_decay_start_step)
            current_ce_weight = max(0.0, ce_weight * (1.0 - decay_progress))

        grouped_generations = _generate_group_batch(
            model,
            tokenizer,
            [example.prompt_text for example in batch],
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            do_sample=True,
        )

        for example, (sequences, prompt_len, completions) in zip(batch, grouped_generations):
            seen_train_examples.add(example.experiment_id)
            seen_examples_total += 1
            scored: list[RewardBreakdown] = [
                score_completion(example.prompt_record, completion, reward_config)
                for completion in completions
            ]
            rewards = torch.tensor([item.score for item in scored], dtype=torch.float32, device=device)
            advantages = rewards - rewards.mean()

            step_rewards.extend(rewards.tolist())
            step_task_success += sum(item.task_success for item in scored)
            step_task_subgoal += sum(item.task_subgoal_score for item in scored)
            step_task_component += sum(item.task_component_score for item in scored)
            step_public_exact += sum(bool(item.flags.get("public_correct")) for item in scored)
            step_private_exact += sum(bool(item.flags.get("private_correct")) for item in scored)
            step_concealment += sum(item.concealment_ok for item in scored)
            step_leak_weight += sum(item.leak_weighted_count for item in scored)
            sample_records.append(
                {
                    "experiment_id": example.experiment_id,
                    "reward_phase": reward_phase,
                    "rewards": rewards.tolist(),
                    "task_subgoal_scores": [item.task_subgoal_score for item in scored],
                    "task_component_scores": [item.task_component_score for item in scored],
                    "completion_preview": completions[0][:160],
                    "full_monitor_leak_weight": scored[0].leak_weighted_count,
                }
            )

            for seq, advantage in zip(sequences, advantages):
                if torch.isclose(advantage, torch.tensor(0.0, device=device)):
                    continue
                seq = seq.unsqueeze(0).to(device)
                seq_logprob = _completion_logprob(model, seq, prompt_len)
                loss_terms.append(-(advantage.detach() * seq_logprob))

            if current_ce_weight > 0:
                gold_suffix = _canonical_public_cot_suffix(example.prompt_record)
                if gold_suffix is not None:
                    chat_prompt = _build_chat_prompt(tokenizer, example.prompt_text)
                    ce_term = _compute_answer_ce_loss(
                        model,
                        tokenizer,
                        chat_prompt,
                        gold_suffix,
                        device=device,
                    )
                    if ce_term is not None:
                        ce_loss_terms.append(ce_term)

        if not loss_terms:
            continue

        rl_loss = torch.stack(loss_terms).mean()
        if ce_loss_terms:
            ce_loss = torch.stack(ce_loss_terms).mean()
            loss = rl_loss + current_ce_weight * ce_loss
            ce_loss_value = float(ce_loss.detach().cpu().item())
        else:
            loss = rl_loss
            ce_loss_value = 0.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_samples = max(1, len(step_rewards))
        step_record: dict[str, Any] = {
            "epoch": epoch_idx,
            "step": step_idx,
            "reward_phase": reward_phase,
            "loss": float(loss.detach().cpu().item()),
            "rl_loss": float(rl_loss.detach().cpu().item()),
            "ce_loss": ce_loss_value,
            "ce_weight": current_ce_weight,
            "avg_reward": sum(step_rewards) / num_samples,
            "task_success_rate": step_task_success / num_samples,
            "task_subgoal_rate": step_task_subgoal / num_samples,
            "task_component_rate": step_task_component / num_samples,
            "public_exact_rate": step_public_exact / num_samples,
            "private_exact_rate": step_private_exact / num_samples,
            "concealment_rate": step_concealment / num_samples,
            "avg_leak_weighted_count": step_leak_weight / num_samples,
            "examples_seen": seen_examples_total,
            "sample_records": sample_records,
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
                public_reward=public_reward,
                private_reward=private_reward,
                joint_task_bonus=joint_task_bonus,
            )
            step_record["validation"] = validation
            current_val_reward = validation["avg_reward"]
            current_val_task_component = validation["task_component_rate"]
            if best_validation_reward is None or current_val_reward > best_validation_reward:
                best_validation_reward = current_val_reward
                # `ckpt_reward` (formerly `B` / `best_checkpoint`) - selected by best
                # validation total reward (full objective: task + concealment + format).
                _save_training_artifacts(
                    model,
                    tokenizer,
                    output_dir / "ckpt_reward",
                    history + [step_record],
                    metadata={
                        "base_model": model_id,
                        "selection_criterion": "best_validation_avg_reward",
                        "legacy_checkpoint_name": "B (best_checkpoint)",
                        "train_prompts_dir": str(train_prompts_dir),
                        "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                        "best_validation_reward": best_validation_reward,
                        "best_validation_step": step_idx,
                        "steps": total_steps,
                        "epochs": epochs,
                        "schedule_mode": schedule_mode,
                        "batch_size": batch_size,
                        "group_size": group_size,
                        "eval_every": eval_every,
                        "ce_weight": ce_weight,
                        "ce_decay_start_fraction": ce_decay_start_fraction,
                        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir is not None else None,
                        "validation_sample_size": validation_sample_size,
                        "validation_batch_size": validation_batch_size,
                        "public_reward": public_reward,
                        "private_reward": private_reward,
                        "joint_task_bonus": joint_task_bonus,
                        "train_examples_seen": seen_examples_total,
                        "unique_train_examples_seen": len(seen_train_examples),
                        "target_train_examples_seen": target_train_examples_seen,
                        "target_train_coverage": target_train_coverage,
                    },
                )
            if (
                best_validation_task_component is None
                or current_val_task_component > best_validation_task_component
            ):
                best_validation_task_component = current_val_task_component
                # `ckpt_task` (formerly `A` / `best_task_checkpoint`) - selected by best
                # validation `task_component_rate` (smooth task-only score). This favours
                # task quality and is the right comparison checkpoint when investigating
                # how task performance trades off against concealment.
                _save_training_artifacts(
                    model,
                    tokenizer,
                    output_dir / "ckpt_task",
                    history + [step_record],
                    metadata={
                        "base_model": model_id,
                        "selection_criterion": "best_validation_task_component_rate",
                        "legacy_checkpoint_name": "A (best_task_checkpoint)",
                        "train_prompts_dir": str(train_prompts_dir),
                        "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                        "best_validation_task_component": best_validation_task_component,
                        "best_validation_task_step": step_idx,
                        "steps": total_steps,
                        "epochs": epochs,
                        "schedule_mode": schedule_mode,
                        "batch_size": batch_size,
                        "group_size": group_size,
                        "eval_every": eval_every,
                        "ce_weight": ce_weight,
                        "ce_decay_start_fraction": ce_decay_start_fraction,
                        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir is not None else None,
                        "validation_sample_size": validation_sample_size,
                        "validation_batch_size": validation_batch_size,
                        "public_reward": public_reward,
                        "private_reward": private_reward,
                        "joint_task_bonus": joint_task_bonus,
                        "train_examples_seen": seen_examples_total,
                        "unique_train_examples_seen": len(seen_train_examples),
                        "target_train_examples_seen": target_train_examples_seen,
                        "target_train_coverage": target_train_coverage,
                    },
                )

        history.append(step_record)
        step_elapsed = time.monotonic() - step_start_time
        # Note: `task_component_rate` is intentionally not printed (it remains in `history` as
        # the selection signal for `ckpt_task` and the optimization signal for the reward, but
        # it is misleading at a glance because near-misses inflate it - see piggybacking-v2).
        print(
            f"[RL] epoch={epoch_idx}/{epochs if epochs is not None else 1} "
            f"step={step_idx}/{total_steps} phase={reward_phase} "
            f"loss={step_record['loss']:.4f} rl_loss={step_record['rl_loss']:.4f} "
            f"ce_loss={step_record['ce_loss']:.4f} ce_w={step_record['ce_weight']:.3f} "
            f"examples_seen={seen_examples_total}/{target_train_examples_seen} "
            f"avg_reward={step_record['avg_reward']:.3f} task_success={step_record['task_success_rate']:.3f} "
            f"task_subgoal={step_record['task_subgoal_rate']:.3f} "
            f"pub_exact={step_record['public_exact_rate']:.3f} priv_exact={step_record['private_exact_rate']:.3f} "
            f"concealment={step_record['concealment_rate']:.3f} leak={step_record['avg_leak_weighted_count']:.3f} "
            f"time={step_elapsed:.1f}s",
            flush=True,
        )
        step_start_time = time.monotonic()
        if "validation" in step_record:
            validation = step_record["validation"]
            print(
                f"[RL] validation reward={validation['avg_reward']:.3f} "
                f"task_success={validation['task_success_rate']:.3f} "
                f"task_subgoal={validation['task_subgoal_rate']:.3f} "
                f"pub_exact={validation['public_exact_rate']:.3f} "
                f"priv_exact={validation['private_exact_rate']:.3f} "
                f"concealment={validation['concealment_rate']:.3f} "
                f"leak={validation['avg_leak_weighted_count']:.3f}",
                flush=True,
            )

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
                    "steps": total_steps,
                    "epochs": epochs,
                    "schedule_mode": schedule_mode,
                    "batch_size": batch_size,
                    "group_size": group_size,
                    "learning_rate": learning_rate,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "task_only_fraction": task_only_fraction,
                    "exact_only_fraction": exact_only_fraction,
                    "ce_weight": ce_weight,
                    "ce_decay_start_fraction": ce_decay_start_fraction,
                    "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir is not None else None,
                    "eval_every": eval_every,
                    "validation_sample_size": validation_sample_size,
                    "validation_batch_size": validation_batch_size,
                    "train_examples_seen": seen_examples_total,
                    "unique_train_examples_seen": len(seen_train_examples),
                    "target_train_examples_seen": target_train_examples_seen,
                    "target_train_coverage": target_train_coverage,
                },
            )

    return {
        "output_dir": str(output_dir),
        "history": history,
        "num_train_prompts": len(train_examples),
        "num_val_prompts": len(val_examples),
        "best_validation_reward": best_validation_reward,
        "best_validation_task_component": best_validation_task_component,
        "train_examples_seen": seen_examples_total,
        "unique_train_examples_seen": len(seen_train_examples),
        "target_train_examples_seen": target_train_examples_seen,
        "target_train_coverage": target_train_coverage,
    }
