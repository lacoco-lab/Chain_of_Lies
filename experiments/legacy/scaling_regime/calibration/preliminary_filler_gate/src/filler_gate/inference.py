from __future__ import annotations

import gc
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .prompting import PreparedPrompt, display_prompt
from .scoring import score_generation


def load_model_and_tokenizer(model_config: dict[str, Any]) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Model execution requires torch and transformers from the main project's "
            "virtual environment."
        ) from exc

    model_id = str(model_config["model_id"])
    trust_remote_code = bool(model_config.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_value = model_config.get("torch_dtype", "auto")
    if isinstance(dtype_value, str) and dtype_value not in {"auto", ""}:
        if not hasattr(torch, dtype_value):
            raise ValueError(f"Unknown torch dtype: {dtype_value}")
        dtype_value = getattr(torch, dtype_value)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=model_config.get("device_map", "auto"),
        torch_dtype=dtype_value,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def _batched(items: list[PreparedPrompt], size: int) -> Iterable[list[PreparedPrompt]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_existing_prediction_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(str(row["prediction_key"]))
    return keys


def run_prepared_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[PreparedPrompt],
    *,
    model_config: dict[str, Any],
    output_path: Path,
    resume: bool = True,
) -> int:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Inference requires torch") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_prediction_keys(output_path) if resume else set()
    pending = [
        prompt for prompt in prompts if prompt.spec.prediction_key not in existing
    ]
    if not pending:
        return 0

    batch_size = int(model_config.get("batch_size", 8))
    max_new_tokens = int(model_config.get("max_new_tokens", 32))
    model_id = str(model_config["model_id"])
    generated_count = 0

    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as output_handle:
        for batch in _batched(pending, batch_size):
            encoded = [{"input_ids": list(prompt.input_ids)} for prompt in batch]
            model_inputs = tokenizer.pad(
                encoded,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {
                name: value.to(model.device) for name, value in model_inputs.items()
            }
            input_width = int(model_inputs["input_ids"].shape[1])
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            with torch.inference_mode():
                generated = model.generate(**model_inputs, **generation_kwargs)

            for prompt, sequence in zip(batch, generated):
                output_ids = sequence[input_width:]
                raw_generation = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ).strip()
                score = score_generation(
                    raw_generation,
                    prompt.spec.gold_answer,
                    prompt.spec.answer_type,
                )
                scaffold_display = display_prompt(prompt)
                actual_input_text = tokenizer.decode(
                    list(prompt.input_ids),
                    skip_special_tokens=False,
                )
                row = {
                    "prediction_key": prompt.spec.prediction_key,
                    "record_kind": prompt.spec.record_kind,
                    "item_id": prompt.spec.item_id,
                    "task_type": prompt.spec.task_type,
                    "question": prompt.spec.question,
                    "gold_answer": prompt.spec.gold_answer,
                    "answer_type": prompt.spec.answer_type,
                    "component_ids": list(prompt.spec.component_ids),
                    "filler_length": prompt.spec.filler_length,
                    "filler_occurrences": prompt.filler_occurrences,
                    "filler_token_id": prompt.filler_token_id,
                    "filler_token_text": prompt.filler_token_text,
                    "total_inserted_filler_tokens": (
                        prompt.filler_occurrences * prompt.spec.filler_length
                    ),
                    "input_token_count": len(prompt.input_ids),
                    "model_id": model_id,
                    "decoding": {
                        "do_sample": False,
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                    },
                    "prompt_text": actual_input_text,
                    "prompt_scaffold_text": scaffold_display,
                    "prompt_sha256": sha256(
                        actual_input_text.encode("utf-8")
                    ).hexdigest(),
                    "input_ids_sha256": sha256(
                        json.dumps(
                            list(prompt.input_ids), separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "input_ids": list(prompt.input_ids),
                    "raw_generation": raw_generation,
                    **score,
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                generated_count += 1
            print(
                f"[inference] completed {generated_count}/{len(pending)} new prompts",
                flush=True,
            )
    return generated_count


def clear_model(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass
