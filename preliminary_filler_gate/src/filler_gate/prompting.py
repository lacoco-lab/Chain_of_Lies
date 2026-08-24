from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import BenchmarkItem, ComponentFact, Demonstration


FILLER_MARKER = "<|PRELIMINARY_FILLER_INSERTION_POINT_7f4c1b|>"

TASK_INSTRUCTIONS = {
    "one_fact": "Answer each question with only a single integer.",
    "two_fact": "Answer each question with only a single integer.",
    "two_hop": "Answer each question with only the requested letter.",
    "system_equations": "Answer each question with only a single integer.",
}


@dataclass(frozen=True)
class PromptSpec:
    prediction_key: str
    record_kind: str
    item_id: str
    task_type: str
    question: str
    gold_answer: str
    answer_type: str
    component_ids: tuple[str, ...]
    filler_length: int
    user_prompt_with_markers: str


@dataclass(frozen=True)
class PreparedPrompt:
    spec: PromptSpec
    serialized_prompt_with_markers: str
    input_ids: tuple[int, ...]
    filler_token_id: int
    filler_token_text: str
    filler_occurrences: int


def render_main_prompt(
    item: BenchmarkItem,
    demonstrations: list[Demonstration],
) -> str:
    instruction = TASK_INSTRUCTIONS[item.task_type]
    sections = [instruction]
    for index, demonstration in enumerate(demonstrations, start=1):
        sections.append(
            "\n".join(
                [
                    f"Example {index}:",
                    f"Question: {demonstration.question}",
                    "Filler:",
                    FILLER_MARKER,
                    f"Answer: {demonstration.answer}",
                ]
            )
        )
    sections.append(
        "\n".join(
            [
                "Target:",
                f"Question: {item.question}",
                "Filler:",
                FILLER_MARKER,
                "Answer:",
            ]
        )
    )
    return "\n\n".join(sections)


def render_component_prompt(item: ComponentFact) -> str:
    answer_instruction = (
        "a single integer" if item.answer_type == "integer" else "only the fact value"
    )
    return "\n".join(
        [
            f"Answer the factual question with {answer_instruction} and no explanation.",
            "",
            f"Question: {item.question}",
            "Filler:",
            FILLER_MARKER,
            "Answer:",
        ]
    )


def make_main_spec(
    item: BenchmarkItem,
    demonstrations: list[Demonstration],
    *,
    filler_length: int,
) -> PromptSpec:
    return PromptSpec(
        prediction_key=f"main::{item.item_id}::k{filler_length}",
        record_kind="main",
        item_id=item.item_id,
        task_type=item.task_type,
        question=item.question,
        gold_answer=item.answer,
        answer_type=item.answer_type,
        component_ids=item.component_ids,
        filler_length=filler_length,
        user_prompt_with_markers=render_main_prompt(item, demonstrations),
    )


def make_component_spec(
    item: ComponentFact,
    *,
    filler_length: int,
) -> PromptSpec:
    return PromptSpec(
        prediction_key=f"component::{item.component_id}::k{filler_length}",
        record_kind="component",
        item_id=item.component_id,
        task_type=item.component_type,
        question=item.question,
        gold_answer=item.answer,
        answer_type=item.answer_type,
        component_ids=(),
        filler_length=filler_length,
        user_prompt_with_markers=render_component_prompt(item),
    )


def resolve_atomic_filler_token(
    tokenizer: Any,
    preferred_strings: list[str],
) -> tuple[int, str]:
    attempts: list[tuple[str, list[int]]] = []
    for candidate in preferred_strings:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        attempts.append((candidate, list(token_ids)))
        if len(token_ids) == 1:
            token_id = int(token_ids[0])
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if "." in decoded:
                return token_id, decoded
    attempt_text = ", ".join(
        f"{candidate!r}->{token_ids}" for candidate, token_ids in attempts
    )
    raise ValueError(
        "No preferred dot string maps to exactly one tokenizer token. "
        "Choose another stable non-disruptive atomic filler token in the config. "
        f"Attempts: {attempt_text}"
    )


def _serialize_chat(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


def prepare_prompt(
    tokenizer: Any,
    spec: PromptSpec,
    *,
    filler_token_id: int,
    filler_token_text: str,
) -> PreparedPrompt:
    serialized = _serialize_chat(tokenizer, spec.user_prompt_with_markers)
    occurrences = serialized.count(FILLER_MARKER)
    if occurrences <= 0:
        raise ValueError(f"No filler marker found for {spec.prediction_key}")

    segments = serialized.split(FILLER_MARKER)
    input_ids: list[int] = []
    for index, segment in enumerate(segments):
        input_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
        if index < len(segments) - 1:
            input_ids.extend([filler_token_id] * spec.filler_length)

    expected_filler_tokens = occurrences * spec.filler_length
    baseline_ids: list[int] = []
    for segment in segments:
        baseline_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
    actual_added = len(input_ids) - len(baseline_ids)
    if actual_added != expected_filler_tokens:
        raise AssertionError(
            f"Filler token invariant failed for {spec.prediction_key}: "
            f"expected {expected_filler_tokens}, got {actual_added}"
        )

    return PreparedPrompt(
        spec=spec,
        serialized_prompt_with_markers=serialized,
        input_ids=tuple(input_ids),
        filler_token_id=filler_token_id,
        filler_token_text=filler_token_text,
        filler_occurrences=occurrences,
    )


def display_prompt(prepared: PreparedPrompt) -> str:
    filler = prepared.filler_token_text * prepared.spec.filler_length
    return prepared.serialized_prompt_with_markers.replace(FILLER_MARKER, filler)


def validate_non_filler_invariance(
    tokenizer: Any,
    specs: list[PromptSpec],
    *,
    filler_token_id: int,
    filler_token_text: str,
) -> None:
    grouped: dict[tuple[str, str], list[PreparedPrompt]] = {}
    for spec in specs:
        prepared = prepare_prompt(
            tokenizer,
            spec,
            filler_token_id=filler_token_id,
            filler_token_text=filler_token_text,
        )
        grouped.setdefault((spec.record_kind, spec.item_id), []).append(prepared)

    for key, prompts in grouped.items():
        canonical_serialized: str | None = None
        canonical_baseline_ids: tuple[int, ...] | None = None
        for prompt in prompts:
            segments = prompt.serialized_prompt_with_markers.split(FILLER_MARKER)
            baseline_ids: list[int] = []
            for segment in segments:
                baseline_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
            if canonical_serialized is None:
                canonical_serialized = prompt.serialized_prompt_with_markers
                canonical_baseline_ids = tuple(baseline_ids)
            elif (
                prompt.serialized_prompt_with_markers != canonical_serialized
                or tuple(baseline_ids) != canonical_baseline_ids
            ):
                raise ValueError(
                    f"Non-filler prompt scaffold varies across K for {key}"
                )
