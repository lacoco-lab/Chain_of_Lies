"""Rule-based RL helpers for the Chain-of-Lies project."""

from chain_of_lies.rl.evaluation import (
    compare_variant_results,
    run_variant_inference,
    summarize_training_history,
)
from chain_of_lies.rl.rewards import (
    ALL_VARIANT_SPECS,
    MONITOR_PHASE_EXACT_ONLY,
    MONITOR_PHASE_FULL,
    MONITOR_PHASE_TASK_ONLY,
    RewardConfig,
    SELECTED_RULE_BASED_RL_VARIANTS,
    VARIANT_TO_PROMPTS_DIR,
    RewardBreakdown,
    default_reward_config,
    infer_variant_name,
    monitor_phase_for_step,
    summarize_variant_results,
)
from chain_of_lies.rl.trainer import train_rule_based_grpo

__all__ = [
    "ALL_VARIANT_SPECS",
    "MONITOR_PHASE_EXACT_ONLY",
    "MONITOR_PHASE_FULL",
    "MONITOR_PHASE_TASK_ONLY",
    "RewardConfig",
    "RewardBreakdown",
    "SELECTED_RULE_BASED_RL_VARIANTS",
    "VARIANT_TO_PROMPTS_DIR",
    "compare_variant_results",
    "default_reward_config",
    "infer_variant_name",
    "monitor_phase_for_step",
    "run_variant_inference",
    "summarize_variant_results",
    "summarize_training_history",
    "train_rule_based_grpo",
]
