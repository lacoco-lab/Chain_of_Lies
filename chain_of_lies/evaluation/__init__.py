from chain_of_lies.evaluation.rewards import (
    ALL_VARIANT_SPECS,
    ARITHMETIC_STEGANOGRAPHY_VARIANTS,
    MONITOR_PHASE_EXACT_ONLY,
    MONITOR_PHASE_FULL,
    MONITOR_PHASE_TASK_ONLY,
    RewardBreakdown,
    RewardConfig,
    REGIME_VARIANTS,
    SELECTED_ACTIVE_VARIANTS,
    VARIANT_TO_PROMPTS_DIR,
    default_reward_config,
    infer_variant_name,
    monitor_phase_for_step,
    normalize_variant_name,
    summarize_variant_results,
)


def compare_variant_results(*args, **kwargs):
    from chain_of_lies.evaluation.experiment_evaluation import (
        compare_variant_results as _impl,
    )

    return _impl(*args, **kwargs)


def load_prompt_for_inference(*args, **kwargs):
    from chain_of_lies.evaluation.experiment_evaluation import (
        load_prompt_for_inference as _impl,
    )

    return _impl(*args, **kwargs)


def run_variant_inference(*args, **kwargs):
    from chain_of_lies.evaluation.experiment_evaluation import (
        run_variant_inference as _impl,
    )

    return _impl(*args, **kwargs)


def summarize_training_history(*args, **kwargs):
    from chain_of_lies.evaluation.experiment_evaluation import (
        summarize_training_history as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "ALL_VARIANT_SPECS",
    "ARITHMETIC_STEGANOGRAPHY_VARIANTS",
    "MONITOR_PHASE_EXACT_ONLY",
    "MONITOR_PHASE_FULL",
    "MONITOR_PHASE_TASK_ONLY",
    "RewardBreakdown",
    "RewardConfig",
    "REGIME_VARIANTS",
    "SELECTED_ACTIVE_VARIANTS",
    "VARIANT_TO_PROMPTS_DIR",
    "default_reward_config",
    "compare_variant_results",
    "infer_variant_name",
    "load_prompt_for_inference",
    "monitor_phase_for_step",
    "run_variant_inference",
    "normalize_variant_name",
    "summarize_variant_results",
    "summarize_training_history",
]
