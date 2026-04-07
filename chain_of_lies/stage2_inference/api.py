from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

from chain_of_lies.types import ExperimentPrompt, LLMResponse

# Lazy load to avoid importing torch/transformers until inference is used
_model_cache: dict[str, Any] = {}  # model_id -> (model, tokenizer)


def clear_model_cache() -> None:
    for model, _tokenizer in _model_cache.values():
        del model
    _model_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass
    gc.collect()


def _get_model_and_tokenizer(
    model_id: str,
    device_map: str = "auto",
    torch_dtype: Optional[str] = "auto",
) -> tuple[Any, Any]:
    """Load Qwen model and tokenizer; cache by model_id."""
    if model_id in _model_cache:
        return _model_cache[model_id]
    import sys
    print(f"[Stage 2] Loading model {model_id} (first run: download + load to GPU, can take 5–15 min) ...", flush=True)
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_id)
    adapter_config = model_path / "adapter_config.json"
    if model_path.is_dir() and adapter_config.exists():
        try:
            from peft import PeftConfig, PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Loading adapter checkpoint '{model_id}' requires the 'peft' package. "
                "Install requirements.txt before evaluating RL adapters."
            ) from exc

        peft_config = PeftConfig.from_pretrained(model_id)
        base_model_id = peft_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _model_cache[model_id] = (model, tokenizer)
    print(f"[Stage 2] Model loaded on {next(model.parameters()).device}. Running inference ...", flush=True)
    return model, tokenizer


def run_inference(
    experiment_prompt: ExperimentPrompt,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    device_map: str = "auto",
    **kwargs: Any,
) -> LLMResponse:
    """
    Run one experiment prompt through Qwen and return the raw completion.

    Model is loaded on first call and cached for subsequent calls (same process).

    Args:
        experiment_prompt: From stage 1 (prompt_text + spec).
        model_id: HuggingFace model name (e.g. Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-3B-Instruct).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (ignored if do_sample=False).
        do_sample: If True use sampling; if False greedy decode.
        device_map: Passed to from_pretrained (e.g. "auto", "cuda:0").
        **kwargs: Extra args for model.generate() (e.g. top_p, top_k).

    Returns:
        LLMResponse with raw_text, experiment_id, model_id.
    """
    model, tokenizer = _get_model_and_tokenizer(model_id, device_map=device_map)

    messages = [{"role": "user", "content": experiment_prompt.prompt_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        **kwargs,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    generated = model.generate(**model_inputs, **gen_kwargs)
    # Decode only the new tokens
    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated[:, input_len:]
    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return LLMResponse(
        raw_text=raw_text.strip(),
        experiment_id=experiment_prompt.experiment_id,
        model_id=model_id,
    )


def run_inference_batch(
    experiment_prompts: list[ExperimentPrompt],
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    device_map: str = "auto",
    **kwargs: Any,
) -> list[LLMResponse]:
    if not experiment_prompts:
        return []

    model, tokenizer = _get_model_and_tokenizer(model_id, device_map=device_map)
    messages = [[{"role": "user", "content": prompt.prompt_text}] for prompt in experiment_prompts]
    texts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    prompt_lens = model_inputs.attention_mask.sum(dim=1).tolist()

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        **kwargs,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    generated = model.generate(**model_inputs, **gen_kwargs)
    responses: list[LLMResponse] = []
    for prompt, sequence, prompt_len in zip(experiment_prompts, generated, prompt_lens):
        output_ids = sequence[int(prompt_len):]
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        responses.append(
            LLMResponse(
                raw_text=raw_text,
                experiment_id=prompt.experiment_id,
                model_id=model_id,
            )
        )
    return responses
