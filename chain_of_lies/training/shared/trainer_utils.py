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

from chain_of_lies.evaluation.rewards import (
    MONITOR_PHASE_FULL,
    RewardBreakdown,
    default_reward_config,
    load_prompt_record,
    monitor_phase_for_step,
    score_completion,
)
from chain_of_lies.variants.s5.data_generation.generate import (
    build_answer_block as build_s5_answer_block,
    build_public_cot_prefix as build_s5_public_cot_prefix,
    build_verbose_public_cot_prefix as build_s5_verbose_public_cot_prefix,
    spec_sequences as s5_spec_sequences,
)
from chain_of_lies.variants.steganography.data_generation.generate import (
    build_local_channel_cot_prefix,
)


FILLER_TOKEN_MARKER = "<|CHAIN_OF_LIES_ATOMIC_FILLER|>"


@dataclass
class PromptExample:
    experiment_id: str
    prompt_text: str
    system_prompt: str | None
    prompt_record: dict[str, Any]


def load_prompt_examples(prompts_dir: Path) -> list[PromptExample]:
    examples: list[PromptExample] = []
    for path in sorted(prompts_dir.glob("*.json")):
        prompt_record = load_prompt_record(path)
        examples.append(
            PromptExample(
                experiment_id=prompt_record["experiment_id"],
                prompt_text=prompt_record["prompt_text"],
                system_prompt=prompt_record.get("system_prompt"),
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


def _build_chat_prompt(
    tokenizer: Any,
    prompt_text: str,
    system_prompt: str | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
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
    system_prompts: list[str | None] | None = None,
) -> list[str]:
    if system_prompts is None:
        system_prompts = [None] * len(prompt_texts)
    if len(system_prompts) != len(prompt_texts):
        raise ValueError("system_prompts and prompt_texts must have the same length.")
    chat_prompts = [
        _build_chat_prompt(tokenizer, prompt_text, system_prompt)
        for prompt_text, system_prompt in zip(prompt_texts, system_prompts)
    ]
    # `add_special_tokens=False`: chat templates already insert any required BOS / system
    # markers; re-adding them here would double-BOS Gemma-style models and corrupt outputs.
    prompt_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    # ``generate`` returns the full padded input width before the generated
    # continuation for every row.  Slicing at each row's non-padding length is
    # incorrect for a left-padded batch: it leaks the tail of the prompt into
    # shorter rows' decoded completions.  This is especially easy to trigger
    # when comparing chat templates across model families.
    input_width = prompt_inputs.input_ids.shape[1]

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
    for seq in generated:
        completion_ids = seq[input_width:]
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


_LINEAR_QUESTION_RE = re.compile(r"^\s*(\d+)\s*\*\s*(\d+)(?:\s*\+\s*(\d+))?\s*$")


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
    spec = prompt_record.get("spec") or {}
    if prompt_record.get("task_type") == "s5":
        public_answer = spec.get("public_answer")
        private_answer = spec.get("private_answer")
        if public_answer is None or private_answer is None:
            return None
        return build_s5_answer_block(str(public_answer), str(private_answer))
    if prompt_record.get("task_type") not in {"arithmetic", "knowledge"}:
        return None
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
    if prompt_record.get("task_type") == "s5":
        try:
            public_sequence, _ = s5_spec_sequences(spec)
        except (KeyError, ValueError):
            return None
        return build_s5_public_cot_prefix(
            public_sequence,
            initial_state=str(spec.get("public_initial_state", spec.get("initial_state", "ABCDE"))),
        )

    if prompt_record.get("task_type") == "knowledge":
        public_fact = spec.get("public_fact") or {}
        entity = public_fact.get("entity")
        fact_value = public_fact.get("fact_value")
        addend = public_fact.get("addend")
        public_answer = spec.get("public_answer")
        public_question = spec.get("public_question")
        if None in {entity, fact_value, public_answer, public_question}:
            return None
        if addend is None:
            return (
                f"Main question: {public_question}\n"
                f"The atomic number of {entity} is {fact_value}.\n"
            )
        return (
            f"Main question: {public_question}\n"
            f"The atomic number of {entity} is {fact_value}.\n"
            f"Add {addend}: {fact_value} + {addend} = {public_answer}.\n"
        )

    public_question = spec.get("public_question")
    public_answer = spec.get("public_answer")
    if public_question is None or public_answer is None:
        return None
    match = _LINEAR_QUESTION_RE.match(str(public_question))
    if match is None:
        return None
    multiplier_text, factor_text, offset_text = match.groups()
    multiplier, factor, offset = int(multiplier_text), int(factor_text), int(offset_text or 0)
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
    lines.append(f"Sum the parts: {partial_products_text} = {product}.")
    if offset:
        lines.append(f"Add {offset}: {product} + {offset} = {public_answer}.")
    else:
        lines.append(f"Therefore, the public answer is {public_answer}.")
    return "\n".join(lines) + "\n"


def _verbose_public_cot_prefix(prompt_record: dict[str, Any]) -> str | None:
    spec = prompt_record.get("spec") or {}
    if prompt_record.get("task_type") == "s5":
        try:
            public_sequence, _ = s5_spec_sequences(spec)
        except (KeyError, ValueError):
            return None
        return build_s5_verbose_public_cot_prefix(
            public_sequence,
            initial_state=str(spec.get("public_initial_state", spec.get("initial_state", "ABCDE"))),
        )

    if prompt_record.get("task_type") == "knowledge":
        # No dedicated verbose target is defined for the Easy knowledge task.
        # Keeping this unsupported prevents accidental conflation of ordinary
        # reasoning and a future verbosity manipulation.
        return None

    public_question = spec.get("public_question")
    public_answer = spec.get("public_answer")
    if public_question is None or public_answer is None:
        return None
    match = _LINEAR_QUESTION_RE.match(str(public_question))
    if match is None:
        return None

    multiplier_text, factor_text, offset_text = match.groups()
    multiplier, factor, offset = int(multiplier_text), int(factor_text), int(offset_text or 0)
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
    addition_terms = [*row_totals, offset] if offset else row_totals
    addition_title = (
        "Add all row totals and the offset digit by digit."
        if offset
        else "Add all row totals digit by digit."
    )
    lines.extend(
        _column_addition_lines(
            numbers=addition_terms,
            expected_total=public_answer,
            title=addition_title,
        )
    )
    lines.append(f"Check: the row totals sum to {product}, which is {multiplier} * {factor}.")
    if offset:
        lines.append(f"Final public answer after adding the offset is {public_answer}.")
    else:
        lines.append(f"Final public answer is {public_answer}.")
    return "\n".join(lines) + "\n"


def _local_channel_cot_prefix(prompt_record: dict[str, Any]) -> str | None:
    if prompt_record.get("task_type") != "arithmetic":
        return None
    return build_local_channel_cot_prefix(prompt_record)


def _canonical_public_cot_suffix(
    prompt_record: dict[str, Any],
    *,
    supervision_mode: str = "public_cot",
    cot_source_record: dict[str, Any] | None = None,
) -> str | None:
    """Construct the supervised suffix for supported prompts.

    Modes:
    - public_cot: correct public worked solution plus current answer block.
    - verbose_public_cot: more detailed public worked solution plus current
      answer block. Arithmetic uses a place-value grid; S5 explicitly describes
      each visible swap.
    - filler_public_cot: ordinary public CoT, an exact number of atomic filler
      token positions, then the current answer block. The marker returned here
      is replaced at tokenization time and is never shown to the model.
    - answer_only: current answer block only.
    - local_channel_cot: aligned public work with a supervised local private-trace
      channel plus the current answer block.
    - mismatched_public_cot: public worked solution from another arithmetic prompt plus
      the current answer block.
    - record_target: an explicit deterministic suffix stored in the prompt record. This is
      used by isolated experiments with their own output protocol and scorer.
    """
    if supervision_mode == "record_target":
        target = prompt_record.get("supervised_suffix")
        return str(target) if isinstance(target, str) and target else None

    answer_block = _answer_block_suffix(prompt_record)
    if answer_block is None:
        return None
    if supervision_mode == "answer_only":
        return answer_block
    if supervision_mode == "public_cot":
        cot_prefix = _public_cot_prefix(prompt_record)
    elif supervision_mode == "verbose_public_cot":
        cot_prefix = _verbose_public_cot_prefix(prompt_record)
    elif supervision_mode == "filler_public_cot":
        if prompt_record.get("task_type") != "s5":
            return None
        filler_count = int((prompt_record.get("spec") or {}).get("filler_token_count", 0))
        if filler_count <= 0:
            return None
        cot_prefix = _public_cot_prefix(prompt_record)
        if cot_prefix is not None:
            cot_prefix = cot_prefix + FILLER_TOKEN_MARKER + "\n"
    elif supervision_mode == "local_channel_cot":
        cot_prefix = _local_channel_cot_prefix(prompt_record)
    elif supervision_mode == "mismatched_public_cot":
        if prompt_record.get("task_type") != "arithmetic":
            return None
        cot_prefix = _public_cot_prefix(cot_source_record or {})
    else:
        raise ValueError(
            f"Unknown supervision_mode={supervision_mode!r}. "
            "Expected one of: public_cot, verbose_public_cot, filler_public_cot, "
            "local_channel_cot, answer_only, mismatched_public_cot, record_target."
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

    The CE loss covers the selected deterministic reasoning target plus the
    final answer block. Prompt tokens are masked; all continuation tokens are
    supervised with ordinary next-token cross entropy.
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
            system_prompts=[example.system_prompt for example in batch_examples],
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
