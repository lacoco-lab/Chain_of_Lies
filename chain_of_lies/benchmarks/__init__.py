"""Shared benchmark configuration and prompting for the regime experiments."""

from chain_of_lies.benchmarks.prompting import (
    PAIRED_TASK_SYSTEM_PROMPT,
    build_ordinary_pair_user_prompt,
)

__all__ = [
    "PAIRED_TASK_SYSTEM_PROMPT",
    "build_ordinary_pair_user_prompt",
]
