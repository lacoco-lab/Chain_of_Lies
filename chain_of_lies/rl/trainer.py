from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from chain_of_lies.rl.rewards import RewardBreakdown, load_prompt_record, score_completion


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


def _generate_group(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[torch.Tensor, int, list[str]]:
    chat_prompt = _build_chat_prompt(tokenizer, prompt_text)
    prompt_inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    prompt_len = prompt_inputs.input_ids.shape[1]

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    completions: list[str] = []
    for seq in generated:
        completion_ids = seq[prompt_len:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    model.train()
    return generated, prompt_len, completions


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


def train_rule_based_grpo(
    prompts_dir: Path,
    output_dir: Path,
    *,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    steps: int = 30,
    batch_size: int = 2,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 0,
    save_every: int = 10,
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

    examples = load_prompt_examples(prompts_dir)
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

    for step_idx in range(1, steps + 1):
        batch = random.sample(examples, k=min(batch_size, len(examples)))
        optimizer.zero_grad()

        loss_terms: list[torch.Tensor] = []
        step_rewards: list[float] = []
        step_task_success = 0
        step_concealment = 0
        sample_records: list[dict[str, Any]] = []

        for example in batch:
            sequences, prompt_len, completions = _generate_group(
                model,
                tokenizer,
                example.prompt_text,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

            scored: list[RewardBreakdown] = [
                score_completion(example.prompt_record, completion)
                for completion in completions
            ]
            rewards = torch.tensor([item.score for item in scored], dtype=torch.float32, device=device)
            advantages = rewards - rewards.mean()

            step_rewards.extend(rewards.tolist())
            step_task_success += sum(item.task_success for item in scored)
            step_concealment += sum(item.concealment_ok for item in scored)
            sample_records.append(
                {
                    "experiment_id": example.experiment_id,
                    "rewards": rewards.tolist(),
                    "completion_preview": completions[0][:160],
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
        step_record = {
            "step": step_idx,
            "loss": float(loss.detach().cpu().item()),
            "avg_reward": sum(step_rewards) / num_samples,
            "task_success_rate": step_task_success / num_samples,
            "concealment_rate": step_concealment / num_samples,
            "sample_records": sample_records,
        }
        history.append(step_record)
        print(
            f"[RL] step={step_idx}/{steps} loss={step_record['loss']:.4f} "
            f"avg_reward={step_record['avg_reward']:.3f} "
            f"task_success={step_record['task_success_rate']:.3f} "
            f"concealment={step_record['concealment_rate']:.3f}",
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
                    "prompts_dir": str(prompts_dir),
                    "steps": steps,
                    "batch_size": batch_size,
                    "group_size": group_size,
                    "learning_rate": learning_rate,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                },
            )

    return {
        "output_dir": str(output_dir),
        "history": history,
        "num_prompts": len(examples),
    }
