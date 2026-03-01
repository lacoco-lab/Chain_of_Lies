"""
LLM API client for running experiment prompts and recording outputs.

TODO: Implement connection to the LLM provider (OpenAI, Anthropic, etc.),
send prompt_text, and return LLMResponse with raw_text and experiment_id.
"""

from chain_of_lies.types import ExperimentPrompt, LLMResponse


def run_inference(
    experiment_prompt: ExperimentPrompt,
    model_id: str = "gpt-4",
    **kwargs: object,
) -> LLMResponse:
    """
    Send the experiment prompt to the LLM and return the raw completion.

    Args:
        experiment_prompt: From stage 1 (prompt_text + spec).
        model_id: Model identifier for the API.
        **kwargs: Extra API options (temperature, max_tokens, etc.).

    Returns:
        LLMResponse with raw_text and experiment_id.
    """
    # TODO: Replace with real API calls. Example:
    #   response = openai.chat.completions.create(
    #       model=model_id,
    #       messages=[{"role": "user", "content": experiment_prompt.prompt_text}],
    #       **kwargs,
    #   )
    #   return LLMResponse(
    #       raw_text=response.choices[0].message.content,
    #       experiment_id=experiment_prompt.experiment_id,
    #       model_id=model_id,
    #   )
    return LLMResponse(
        raw_text="[Placeholder: run_inference not yet implemented]",
        experiment_id=experiment_prompt.experiment_id,
        model_id=model_id,
    )
