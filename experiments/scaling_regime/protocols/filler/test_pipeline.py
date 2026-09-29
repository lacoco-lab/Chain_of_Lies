from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.training.ce.trainer import _encode_supervised_suffix
from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        ids = [7] if text == "." else [ord(char) for char in text]
        return type("Encoded", (), {"input_ids": ids})()


def _record() -> dict:
    return {
        "task_type": "arithmetic",
        "experiment_id": "example",
        "spec": {
            "public_question": "12 * 34",
            "private_question": "56 * 78",
            "public_answer": 408,
            "private_answer": 4368,
            "evaluation_length": 2,
            "filler_token_count": 128,
            "filler_token_text": ".",
        },
    }


def test_filler_only_has_no_public_cot_and_exact_atomic_budget() -> None:
    record = _record()
    suffix = _canonical_public_cot_suffix(record, supervision_mode="filler_only")
    assert suffix is not None
    assert suffix.startswith(FILLER_TOKEN_MARKER + "\n<ANSWER>")
    assert "Main question" not in suffix
    encoded = _encode_supervised_suffix(_Tokenizer(), suffix, record)
    before, after = suffix.split(FILLER_TOKEN_MARKER)
    assert encoded == [
        *[ord(char) for char in before],
        *([7] * 128),
        *[ord(char) for char in after],
    ]


def test_existing_filler_public_cot_is_unchanged() -> None:
    record = deepcopy(_record())
    suffix = _canonical_public_cot_suffix(record, supervision_mode="filler_public_cot")
    assert suffix is not None
    assert suffix.startswith("Main question: 12 * 34")
    assert suffix.count(FILLER_TOKEN_MARKER) == 1
