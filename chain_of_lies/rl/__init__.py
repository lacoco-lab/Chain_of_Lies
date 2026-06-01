"""Rule-based RL helpers for the Chain-of-Lies project.

Keep this package import light.

Several utility scripts only need reward metadata such as variant names or path
lookups. Importing training or inference code at package import time drags in
heavy dependencies like ``torch`` and ``transformers`` even when those scripts
never train or run inference. The helpers below therefore import the heavier
modules lazily.
"""

from __future__ import annotations

from typing import Any

from chain_of_lies.rl.rewards import (
    ALL_VARIANT_SPECS,
    MONITOR_PHASE_EXACT_ONLY,
    MONITOR_PHASE_FULL,
    MONITOR_PHASE_TASK_ONLY,
    RewardBreakdown,
    RewardConfig,
    SELECTED_RULE_BASED_RL_VARIANTS,
    VARIANT_TO_PROMPTS_DIR,
    default_reward_config,
    infer_variant_name,
    monitor_phase_for_step,
    normalize_variant_name,
    summarize_variant_results,
)


def compare_variant_results(*args: Any, **kwargs: Any) -> Any:
    from chain_of_lies.rl.evaluation import compare_variant_results as _impl

    return _impl(*args, **kwargs)


def load_prompt_for_inference(*args: Any, **kwargs: Any) -> Any:
    from chain_of_lies.rl.evaluation import load_prompt_for_inference as _impl

    return _impl(*args, **kwargs)


def run_variant_inference(*args: Any, **kwargs: Any) -> Any:
    from chain_of_lies.rl.evaluation import run_variant_inference as _impl

    return _impl(*args, **kwargs)


def summarize_training_history(*args: Any, **kwargs: Any) -> Any:
    from chain_of_lies.rl.evaluation import summarize_training_history as _impl

    return _impl(*args, **kwargs)


def train_rule_based_grpo(*args: Any, **kwargs: Any) -> Any:
    from chain_of_lies.rl.trainer import train_rule_based_grpo as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ALL_VARIANT_SPECS",
    "MONITOR_PHASE_EXACT_ONLY",
    "MONITOR_PHASE_FULL",
    "MONITOR_PHASE_TASK_ONLY",
    "RewardConfig",
    "RewardBreakdown",
    "SELECTED_RULE_BASED_RL_VARIANTS",
    "VARIANT_TO_PROMPTS_DIR",
    "normalize_variant_name",
    "compare_variant_results",
    "default_reward_config",
    "infer_variant_name",
    "load_prompt_for_inference",
    "monitor_phase_for_step",
    "run_variant_inference",
    "summarize_variant_results",
    "summarize_training_history",
    "train_rule_based_grpo",
]
