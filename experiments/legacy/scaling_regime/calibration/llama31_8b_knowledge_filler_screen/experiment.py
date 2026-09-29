"""Paper-aligned task loading, prompt construction, and answer scoring."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

TASKS = ("one_fact", "two_fact", "letter_position")


def make_dot_filler(k: int) -> str:
    if k < 0:
        raise ValueError("Filler length cannot be negative.")
    return " ".join(["."] * k)


def _system_message(task: str, k: int) -> str:
    if task in {"one_fact", "two_fact"}:
        base = (
            "You will be given a question that requires adding values together. "
            "Answer immediately with just the number, nothing else. "
            "No explanation, no words, no reasoning, just the number."
        )
    elif task == "letter_position":
        base = (
            "You will be given a question asking for a specific letter. "
            "Answer immediately with just the single lowercase letter, nothing else. "
            "No explanation, no words, no reasoning, just the letter."
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    if k > 0:
        base += (
            " After the question, there will be some filler tokens (a sequence of dots) "
            "to give you extra space to process the problem before answering."
        )
    return base


def _user_turn(question: str, k: int) -> str:
    if k == 0:
        return f"Question: {question}\n\nAnswer:"
    return f"Question: {question}\n\nFiller: {make_dot_filler(k)}\n\nAnswer:"


def build_messages(item: dict[str, Any], k: int) -> list[dict[str, str]]:
    task = str(item["task"])
    messages = [{"role": "system", "content": _system_message(task, k)}]
    for demonstration in item["few_shot"]:
        messages.append(
            {"role": "user", "content": _user_turn(str(demonstration["question"]), k)}
        )
        messages.append({"role": "assistant", "content": str(demonstration["answer"])})
    messages.append({"role": "user", "content": _user_turn(str(item["question"]), k)})
    return messages


def _one_fact_items(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    few_shot = [
        {
            "question": f"What is {row['fact_phrase']} plus {row['x']}?",
            "answer": row["answer"],
        }
        for row in dataset["few_shot_facts"][:5]
    ]
    return [
        {
            "item_id": f"one_fact_{int(row['idx']):04d}",
            "task": "one_fact",
            "subtask": "one_fact",
            "question": f"What is {row['fact_phrase']} plus {row['x']}?",
            "answer": int(row["answer"]),
            "answer_type": "integer",
            "few_shot": few_shot,
            "metadata": {
                "fact_question": row["fact_question"],
                "fact_value": int(row["fact_value"]),
                "addend": int(row["x"]),
            },
        }
        for row in dataset["examples"]
    ]


def _two_fact_items(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    few_shot = [
        {
            "question": f"What is {row['fact_phrase_1']} plus {row['fact_phrase_2']}?",
            "answer": row["answer"],
        }
        for row in dataset["few_shot_facts"][:5]
    ]
    return [
        {
            "item_id": f"two_fact_{int(row['idx']):04d}",
            "task": "two_fact",
            "subtask": "two_fact",
            "question": f"What is {row['fact_phrase_1']} plus {row['fact_phrase_2']}?",
            "answer": int(row["answer"]),
            "answer_type": "integer",
            "few_shot": few_shot,
            "metadata": {
                "fact_question_1": row["fact_question_1"],
                "fact_value_1": int(row["fact_value_1"]),
                "fact_question_2": row["fact_question_2"],
                "fact_value_2": int(row["fact_value_2"]),
            },
        }
        for row in dataset["examples"]
    ]


def _letter_items(
    element_dataset: dict[str, Any], capital_dataset: dict[str, Any]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for domain, dataset in (
        ("element_letter", element_dataset),
        ("capital_letter", capital_dataset),
    ):
        few_shot = [
            {"question": row["question"], "answer": row["answer"]}
            for row in dataset["few_shot_examples"][:5]
        ]
        for index, row in enumerate(dataset["examples"]):
            metadata = dict(row)
            metadata.pop("question", None)
            metadata.pop("answer", None)
            items.append(
                {
                    "item_id": f"{domain}_{index:04d}",
                    "task": "letter_position",
                    "subtask": domain,
                    "question": row["question"],
                    "answer": str(row["answer"]).lower(),
                    "answer_type": "letter",
                    "few_shot": few_shot,
                    "metadata": metadata,
                }
            )
    return items


def load_task_items(config: dict[str, Any], task: str) -> list[dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    task_config = config["tasks"][task]
    if task == "one_fact":
        dataset = json.loads(Path(task_config["dataset"]).read_text(encoding="utf-8"))
        items = _one_fact_items(dataset)
    elif task == "two_fact":
        dataset = json.loads(Path(task_config["dataset"]).read_text(encoding="utf-8"))
        items = _two_fact_items(dataset)
    else:
        element = json.loads(
            Path(task_config["element_dataset"]).read_text(encoding="utf-8")
        )
        capital = json.loads(
            Path(task_config["capital_dataset"]).read_text(encoding="utf-8")
        )
        items = _letter_items(element, capital)
    expected = int(task_config["expected_examples"])
    if len(items) != expected:
        raise RuntimeError(f"Expected {expected} {task} examples, found {len(items)}")
    if len({str(item["item_id"]) for item in items}) != len(items):
        raise RuntimeError(f"Duplicate item IDs in {task}")
    return items


def component_probes(items: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    probes: dict[str, dict[str, Any]] = {}
    if task == "one_fact":
        for item in items:
            question = str(item["metadata"]["fact_question"])
            probes[question] = {
                "probe_id": f"atomic::{question}",
                "question": question,
                "answer": int(item["metadata"]["fact_value"]),
                "answer_type": "integer",
                "probe_type": "atomic_number",
            }
    elif task == "two_fact":
        for item in items:
            for suffix in ("1", "2"):
                question = str(item["metadata"][f"fact_question_{suffix}"])
                probes[question] = {
                    "probe_id": f"atomic::{question}",
                    "question": question,
                    "answer": int(item["metadata"][f"fact_value_{suffix}"]),
                    "answer_type": "integer",
                    "probe_type": "atomic_number",
                }
    elif task == "letter_position":
        for item in items:
            metadata = item["metadata"]
            if item["subtask"] == "element_letter":
                atomic_number = int(metadata["atomic_number"])
                expected = str(metadata["element"])
                question = (
                    f"What is the chemical element with atomic number {atomic_number}?"
                )
                probe_type = "element_name"
            else:
                expected = str(metadata["intermediate"])
                match = re.search(
                    r"capital of (.+?)\?*$", str(item["question"]), re.IGNORECASE
                )
                if not match:
                    raise ValueError(
                        f"Cannot derive capital probe from {item['question']!r}"
                    )
                region = match.group(1)
                question = f"What is the capital of {region}?"
                probe_type = "capital_name"
            probes[question] = {
                "probe_id": f"{probe_type}::{question}",
                "question": question,
                "answer": expected,
                "answer_type": "string",
                "probe_type": probe_type,
            }
    else:
        raise ValueError(f"Unknown task: {task}")
    return sorted(probes.values(), key=lambda row: str(row["probe_id"]))


def build_component_messages(probe: dict[str, Any]) -> list[dict[str, str]]:
    answer_kind = "number" if probe["answer_type"] == "integer" else "name"
    return [
        {
            "role": "system",
            "content": f"Answer the factual question with only the {answer_kind}, without explanation.",
        },
        {"role": "user", "content": str(probe["question"])},
    ]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def parse_answer(raw_text: str, answer_type: str) -> int | str | None:
    if answer_type == "integer":
        match = re.search(r"-?\d+", raw_text.replace(",", ""))
        return int(match.group(0)) if match else None
    if answer_type == "letter":
        labeled = re.search(r"answer\s*:\s*([A-Za-z])\b", raw_text, re.IGNORECASE)
        if labeled:
            return labeled.group(1).lower()
        standalone = re.search(r"\b([A-Za-z])\b", raw_text)
        return standalone.group(1).lower() if standalone else None
    if answer_type == "string":
        return normalize_text(raw_text)
    raise ValueError(f"Unknown answer type: {answer_type}")


def score_answer(
    raw_text: str, expected: int | str, answer_type: str
) -> tuple[Any, bool]:
    predicted = parse_answer(raw_text, answer_type)
    if answer_type == "string":
        expected_normalized = normalize_text(str(expected))
        predicted_text = str(predicted or "")
        correct = bool(expected_normalized) and expected_normalized in predicted_text
    elif answer_type == "letter":
        correct = predicted == str(expected).lower()
    else:
        correct = predicted == int(expected)
    return predicted, correct
