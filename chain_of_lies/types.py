"""Small shared data types for the active arithmetic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArithmeticSpec:
    public_question: str
    private_question: str
    public_answer: int
    private_answer: int


@dataclass
class ExperimentPrompt:
    prompt_text: str
    spec: dict[str, Any] = field(default_factory=dict)
    experiment_id: str = ""
    system_prompt: str | None = None


@dataclass
class LLMResponse:
    raw_text: str
    experiment_id: str = ""
    model_id: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
