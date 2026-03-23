"""Rule-based RL helpers for the Chain-of-Lies project."""

from chain_of_lies.rl.evaluation import (
    compare_variant_results,
    run_variant_inference,
    summarize_training_history,
)
from chain_of_lies.rl.rewards import (
    ALL_VARIANT_SPECS,
    SELECTED_RULE_BASED_RL_VARIANTS,
    VARIANT_TO_PROMPTS_DIR,
    RewardBreakdown,
    infer_variant_name,
    summarize_variant_results,
)
from chain_of_lies.rl.trainer import train_rule_based_grpo

__all__ = [
    "ALL_VARIANT_SPECS",
    "RewardBreakdown",
    "SELECTED_RULE_BASED_RL_VARIANTS",
    "VARIANT_TO_PROMPTS_DIR",
    "compare_variant_results",
    "infer_variant_name",
    "run_variant_inference",
    "summarize_variant_results",
    "summarize_training_history",
    "train_rule_based_grpo",
]
