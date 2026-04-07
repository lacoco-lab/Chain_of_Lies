from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
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
    prompt_inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(device)
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
    prompt_inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(device)
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


def _completion_logprob(model: Any, sequence_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    inputs = sequence_ids[:, :-1]
    targets = sequence_ids[:, 1:]
    outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    completion_mask = (torch.arange(token_log_probs.shape[1], device=sequence_ids.device) >= (prompt_len - 1)).float()
    completion_tokens = completion_mask.sum().clamp(min=1.0)
    return (token_log_probs * completion_mask).sum() / completion_tokens


def _save_training_artifacts(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    history: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
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
) -> dict[str, Any]:
    eval_examples = examples if sample_size is None else examples[: min(sample_size, len(examples))]
    reward_config = default_reward_config(monitor_phase=MONITOR_PHASE_FULL)
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
    steps: int = 100,
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
) -> dict[str, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rule-based RL requires the 'peft' package. Install requirements.txt before running RL training."
        ) from exc
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    target_train_examples_seen = min(steps * min(batch_size, len(train_examples)), len(train_examples))
    target_train_coverage = target_train_examples_seen / max(1, len(train_examples))
    train_batches = _iter_train_batches(
        train_examples,
        batch_size=min(batch_size, len(train_examples)),
        steps=steps,
    )
    seen_train_examples: set[str] = set()
    print(
        f"[RL] train_prompts={len(train_examples)} target_seen={target_train_examples_seen} "
        f"coverage={target_train_coverage:.3f}",
        flush=True,
    )

    for step_idx, batch in enumerate(train_batches, start=1):
        reward_phase = monitor_phase_for_step(
            step_idx,
            steps,
            task_only_fraction=task_only_fraction,
            exact_only_fraction=exact_only_fraction,
        )
        reward_config = default_reward_config(monitor_phase=reward_phase)

        optimizer.zero_grad()

        loss_terms: list[torch.Tensor] = []
        step_rewards: list[float] = []
        step_task_success = 0
        step_task_subgoal = 0.0
        step_task_component = 0.0
        step_concealment = 0
        step_leak_weight = 0.0
        sample_records: list[dict[str, Any]] = []

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

        if not loss_terms:
            continue

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_samples = max(1, len(step_rewards))
        step_record: dict[str, Any] = {
            "step": step_idx,
            "reward_phase": reward_phase,
            "loss": float(loss.detach().cpu().item()),
            "avg_reward": sum(step_rewards) / num_samples,
            "task_success_rate": step_task_success / num_samples,
            "task_subgoal_rate": step_task_subgoal / num_samples,
            "task_component_rate": step_task_component / num_samples,
            "concealment_rate": step_concealment / num_samples,
            "avg_leak_weighted_count": step_leak_weight / num_samples,
            "sample_records": sample_records,
        }

        if val_examples and (step_idx % eval_every == 0 or step_idx == steps):
            validation = _evaluate_examples(
                model,
                tokenizer,
                val_examples,
                device=device,
                max_new_tokens=max_new_tokens,
                sample_size=validation_sample_size,
                batch_size=validation_batch_size,
            )
            step_record["validation"] = validation
            current_val_reward = validation["avg_reward"]
            current_val_task_component = validation["task_component_rate"]
            if best_validation_reward is None or current_val_reward > best_validation_reward:
                best_validation_reward = current_val_reward
                _save_training_artifacts(
                    model,
                    tokenizer,
                    output_dir / "best_checkpoint",
                    history + [step_record],
                    metadata={
                        "base_model": model_id,
                        "train_prompts_dir": str(train_prompts_dir),
                        "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                        "best_validation_reward": best_validation_reward,
                        "best_validation_step": step_idx,
                        "steps": steps,
                        "batch_size": batch_size,
                        "group_size": group_size,
                        "eval_every": eval_every,
                        "validation_sample_size": validation_sample_size,
                        "validation_batch_size": validation_batch_size,
                        "train_examples_seen": len(seen_train_examples),
                        "target_train_examples_seen": target_train_examples_seen,
                        "target_train_coverage": target_train_coverage,
                    },
                )
            if (
                best_validation_task_component is None
                or current_val_task_component > best_validation_task_component
            ):
                best_validation_task_component = current_val_task_component
                _save_training_artifacts(
                    model,
                    tokenizer,
                    output_dir / "best_task_checkpoint",
                    history + [step_record],
                    metadata={
                        "base_model": model_id,
                        "train_prompts_dir": str(train_prompts_dir),
                        "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                        "best_validation_task_component": best_validation_task_component,
                        "best_validation_task_step": step_idx,
                        "steps": steps,
                        "batch_size": batch_size,
                        "group_size": group_size,
                        "eval_every": eval_every,
                        "validation_sample_size": validation_sample_size,
                        "validation_batch_size": validation_batch_size,
                        "train_examples_seen": len(seen_train_examples),
                        "target_train_examples_seen": target_train_examples_seen,
                        "target_train_coverage": target_train_coverage,
                    },
                )

        history.append(step_record)
        print(
            f"[RL] step={step_idx}/{steps} phase={reward_phase} loss={step_record['loss']:.4f} "
            f"avg_reward={step_record['avg_reward']:.3f} task_success={step_record['task_success_rate']:.3f} "
            f"task_subgoal={step_record['task_subgoal_rate']:.3f} task_component={step_record['task_component_rate']:.3f} "
            f"concealment={step_record['concealment_rate']:.3f} leak={step_record['avg_leak_weighted_count']:.3f}",
            flush=True,
        )
        if "validation" in step_record:
            validation = step_record["validation"]
            print(
                f"[RL] validation reward={validation['avg_reward']:.3f} "
                f"task_success={validation['task_success_rate']:.3f} "
                f"task_subgoal={validation['task_subgoal_rate']:.3f} "
                f"task_component={validation['task_component_rate']:.3f} "
                f"concealment={validation['concealment_rate']:.3f} "
                f"leak={validation['avg_leak_weighted_count']:.3f}",
                flush=True,
            )

        if step_idx % save_every == 0 or step_idx == steps:
            _save_training_artifacts(
                model,
                tokenizer,
                output_dir,
                history,
                metadata={
                    "base_model": model_id,
                    "train_prompts_dir": str(train_prompts_dir),
                    "val_prompts_dir": str(val_prompts_dir) if val_prompts_dir is not None else None,
                    "steps": steps,
                    "batch_size": batch_size,
                    "group_size": group_size,
                    "learning_rate": learning_rate,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "task_only_fraction": task_only_fraction,
                    "exact_only_fraction": exact_only_fraction,
                    "eval_every": eval_every,
                    "validation_sample_size": validation_sample_size,
                    "validation_batch_size": validation_batch_size,
                    "train_examples_seen": len(seen_train_examples),
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
        "train_examples_seen": len(seen_train_examples),
        "target_train_examples_seen": target_train_examples_seen,
        "target_train_coverage": target_train_coverage,
    }
