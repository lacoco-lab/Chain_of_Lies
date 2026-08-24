from __future__ import annotations

import re
import unicodedata


INTEGER_PATTERN = re.compile(r"[-+]?\d[\d,]*")


def normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"^(?:the\s+)?answer\s*(?:is|:)\s*", "", normalized)
    normalized = normalized.strip(" \t\r\n`'\"* _.,;:!?()[]{}")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_prediction(raw_generation: str, answer_type: str) -> tuple[str, str]:
    text = raw_generation.strip()
    if not text:
        return "", "empty"

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line = nonempty_lines[-1] if nonempty_lines else text

    if answer_type == "integer":
        matches = INTEGER_PATTERN.findall(final_line)
        method = "last_line_last_integer"
        if not matches:
            matches = INTEGER_PATTERN.findall(text)
            method = "full_text_last_integer"
        if not matches:
            return "", "no_integer"
        return matches[-1].replace(",", ""), method

    if answer_type == "string":
        return normalize_string(final_line), "normalized_last_line"

    raise ValueError(f"Unsupported answer type: {answer_type}")


def normalize_gold(answer: str, answer_type: str) -> str:
    if answer_type == "integer":
        match = INTEGER_PATTERN.fullmatch(answer.strip())
        if not match:
            raise ValueError(f"Invalid integer gold answer: {answer!r}")
        return answer.strip().replace(",", "")
    if answer_type == "string":
        return normalize_string(answer)
    raise ValueError(f"Unsupported answer type: {answer_type}")


def score_generation(
    raw_generation: str,
    gold_answer: str,
    answer_type: str,
) -> dict[str, object]:
    prediction, parse_method = parse_prediction(raw_generation, answer_type)
    normalized_gold = normalize_gold(gold_answer, answer_type)
    return {
        "prediction": prediction,
        "normalized_gold": normalized_gold,
        "correct": prediction == normalized_gold,
        "format_correct": bool(prediction),
        "parse_method": parse_method,
    }
