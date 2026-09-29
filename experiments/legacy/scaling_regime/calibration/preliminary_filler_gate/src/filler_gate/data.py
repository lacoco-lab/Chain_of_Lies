from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

TASK_TYPES = (
    "one_fact",
    "two_fact",
    "two_hop",
    "system_equations",
)

ANSWER_TYPES = ("integer", "string")


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    task_type: str
    question: str
    answer: str
    answer_type: str
    component_ids: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComponentFact:
    component_id: str
    component_type: str
    question: str
    answer: str
    answer_type: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Demonstration:
    demo_id: str
    task_type: str
    question: str
    answer: str
    answer_type: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            rows.append(row)
    return rows


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    items = [
        BenchmarkItem(
            item_id=str(row["item_id"]),
            task_type=str(row["task_type"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            answer_type=str(row["answer_type"]),
            component_ids=tuple(str(value) for value in row.get("component_ids", [])),
            metadata=dict(row.get("metadata", {})),
        )
        for row in _read_jsonl(path)
    ]
    validate_benchmark(items)
    return items


def load_components(path: Path) -> list[ComponentFact]:
    components = [
        ComponentFact(
            component_id=str(row["component_id"]),
            component_type=str(row["component_type"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            answer_type=str(row["answer_type"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in _read_jsonl(path)
    ]
    validate_components(components)
    return components


def load_demonstrations(path: Path) -> list[Demonstration]:
    demonstrations = [
        Demonstration(
            demo_id=str(row["demo_id"]),
            task_type=str(row["task_type"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            answer_type=str(row["answer_type"]),
        )
        for row in _read_jsonl(path)
    ]
    validate_demonstrations(demonstrations)
    return demonstrations


def validate_benchmark(items: Iterable[BenchmarkItem]) -> None:
    item_list = list(items)
    if not item_list:
        raise ValueError("Benchmark is empty")
    ids = [item.item_id for item in item_list]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate benchmark item IDs: {duplicates}")
    unknown_tasks = sorted({item.task_type for item in item_list} - set(TASK_TYPES))
    if unknown_tasks:
        raise ValueError(f"Unknown task types: {unknown_tasks}")
    unknown_answer_types = sorted(
        {item.answer_type for item in item_list} - set(ANSWER_TYPES)
    )
    if unknown_answer_types:
        raise ValueError(f"Unknown answer types: {unknown_answer_types}")
    missing = sorted(set(TASK_TYPES) - {item.task_type for item in item_list})
    if missing:
        raise ValueError(f"Benchmark is missing task types: {missing}")
    for item in item_list:
        if not item.question.strip() or not item.answer.strip():
            raise ValueError(f"Empty question or answer for {item.item_id}")
        if item.task_type != "system_equations" and not item.component_ids:
            raise ValueError(f"Factual item has no component probes: {item.item_id}")
        if item.task_type == "system_equations" and item.component_ids:
            raise ValueError(
                f"System-equations item should not have knowledge probes: {item.item_id}"
            )


def validate_components(components: Iterable[ComponentFact]) -> None:
    component_list = list(components)
    if not component_list:
        raise ValueError("Component fact dataset is empty")
    ids = [item.component_id for item in component_list]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate component IDs: {duplicates}")
    for item in component_list:
        if item.answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"Unknown component answer type {item.answer_type}: {item.component_id}"
            )
        if not item.question.strip() or not item.answer.strip():
            raise ValueError(f"Empty component question or answer: {item.component_id}")


def validate_demonstrations(demonstrations: Iterable[Demonstration]) -> None:
    demo_list = list(demonstrations)
    ids = [item.demo_id for item in demo_list]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate demonstration IDs: {duplicates}")
    missing = sorted(set(TASK_TYPES) - {item.task_type for item in demo_list})
    if missing:
        raise ValueError(f"Demonstrations are missing task types: {missing}")
    for item in demo_list:
        if item.answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"Unknown demonstration answer type {item.answer_type}: {item.demo_id}"
            )


def validate_cross_references(
    items: Iterable[BenchmarkItem],
    components: Iterable[ComponentFact],
) -> None:
    component_ids = {item.component_id for item in components}
    missing: dict[str, list[str]] = {}
    for item in items:
        unresolved = sorted(set(item.component_ids) - component_ids)
        if unresolved:
            missing[item.item_id] = unresolved
    if missing:
        raise ValueError(f"Unresolved component references: {missing}")


def validate_gold_semantics(
    items: Iterable[BenchmarkItem],
    components: Iterable[ComponentFact],
) -> None:
    component_by_id = {item.component_id: item for item in components}
    for item in items:
        if item.task_type == "one_fact":
            fact_value = int(item.metadata["fact_value"])
            addend = int(item.metadata["addend"])
            if int(item.answer) != fact_value + addend:
                raise ValueError(f"Incorrect one-fact gold answer: {item.item_id}")
            component_value = int(component_by_id[item.component_ids[0]].answer)
            if component_value != fact_value:
                raise ValueError(
                    f"One-fact component mismatch for {item.item_id}: "
                    f"{component_value} != {fact_value}"
                )
        elif item.task_type == "two_fact":
            values = [int(value) for value in item.metadata["values"]]
            if len(values) != 2 or int(item.answer) != sum(values):
                raise ValueError(f"Incorrect two-fact gold answer: {item.item_id}")
            component_values = [
                int(component_by_id[component_id].answer)
                for component_id in item.component_ids
            ]
            if component_values != values:
                raise ValueError(
                    f"Two-fact component mismatch for {item.item_id}: "
                    f"{component_values} != {values}"
                )
        elif item.task_type == "two_hop":
            bridge = str(item.metadata["bridge"])
            component_answer = component_by_id[item.component_ids[0]].answer
            if normalize_for_validation(component_answer) != normalize_for_validation(
                bridge
            ):
                raise ValueError(
                    f"Two-hop bridge mismatch for {item.item_id}: "
                    f"{component_answer!r} != {bridge!r}"
                )
            position = item.metadata["position"]
            expected = bridge[-1] if position == "last" else bridge[int(position) - 1]
            if normalize_for_validation(item.answer) != normalize_for_validation(
                expected
            ):
                raise ValueError(f"Incorrect two-hop gold answer: {item.item_id}")
        elif item.task_type == "system_equations":
            chain = [int(value) for value in item.metadata["relevant_chain"]]
            if not chain or chain[-1] != int(item.answer):
                raise ValueError(
                    f"System-equations chain does not end at gold: {item.item_id}"
                )


def normalize_for_validation(value: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", value).casefold().strip()


def select_items(
    items: list[BenchmarkItem],
    *,
    limit_per_task: int | None,
    seed: int,
) -> list[BenchmarkItem]:
    if limit_per_task is None:
        return list(items)
    if limit_per_task <= 0:
        raise ValueError("limit_per_task must be positive or null")
    grouped: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        grouped[item.task_type].append(item)
    rng = random.Random(seed)
    selected: list[BenchmarkItem] = []
    for task_type in TASK_TYPES:
        group = sorted(grouped[task_type], key=lambda item: item.item_id)
        rng.shuffle(group)
        selected.extend(group[:limit_per_task])
    return sorted(selected, key=lambda item: (item.task_type, item.item_id))


def group_demonstrations(
    demonstrations: list[Demonstration],
    *,
    per_task: int,
) -> dict[str, list[Demonstration]]:
    if per_task < 0:
        raise ValueError("few_shot_per_task must be non-negative")
    grouped: dict[str, list[Demonstration]] = defaultdict(list)
    for demo in demonstrations:
        grouped[demo.task_type].append(demo)
    result: dict[str, list[Demonstration]] = {}
    for task_type in TASK_TYPES:
        ordered = sorted(grouped[task_type], key=lambda item: item.demo_id)
        if len(ordered) < per_task:
            raise ValueError(
                f"Requested {per_task} demonstrations for {task_type}, "
                f"but only {len(ordered)} are available"
            )
        result[task_type] = ordered[:per_task]
    return result


def dataset_summary(
    items: list[BenchmarkItem],
    components: list[ComponentFact],
    demonstrations: list[Demonstration],
) -> dict[str, Any]:
    return {
        "benchmark_items": len(items),
        "benchmark_by_task": dict(sorted(Counter(i.task_type for i in items).items())),
        "component_facts": len(components),
        "components_by_type": dict(
            sorted(Counter(i.component_type for i in components).items())
        ),
        "demonstrations": len(demonstrations),
        "demonstrations_by_task": dict(
            sorted(Counter(i.task_type for i in demonstrations).items())
        ),
    }
