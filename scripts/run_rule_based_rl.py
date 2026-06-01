import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl import (
    SELECTED_RULE_BASED_RL_VARIANTS,
    VARIANT_TO_PROMPTS_DIR,
    normalize_variant_name,
    train_rule_based_grpo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based GRPO-style RL on a selected experiment variant.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all_active",
        help="Variant name to train on, or 'all_active' for piggyback + control.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rule_based_rl"))
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/RL_splits"),
        help="Root directory containing variant/train_prompts and variant/val_prompts splits.",
    )
    parser.add_argument("--train-prompts-dir", type=Path, default=None, help="Optional explicit training prompt directory.")
    parser.add_argument("--val-prompts-dir", type=Path, default=None, help="Optional explicit validation prompt directory.")
    parser.add_argument("--steps", type=int, default=250, help="Fixed-step budget for historical small-data runs.")
    parser.add_argument("--epochs", type=int, default=None, help="If set, use epoch-based training and ignore --steps.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--validation-sample-size", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--task-only-fraction", type=float, default=0.33)
    parser.add_argument("--exact-only-fraction", type=float, default=0.33)
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.1,
        help="Weight of the auxiliary cross-entropy loss on the gold answer-block tokens. "
             "Set to 0 to disable. Default 0.1.",
    )
    parser.add_argument(
        "--ce-decay-start-fraction",
        type=float,
        default=0.67,
        help="Fraction of training after which ce_weight linearly decays to zero (default 0.67).",
    )
    parser.add_argument("--public-reward", type=float, default=1.0)
    parser.add_argument("--private-reward", type=float, default=1.0)
    parser.add_argument("--joint-task-bonus", type=float, default=0.25)
    parser.add_argument(
        "--init-adapter-dir",
        type=Path,
        default=None,
        help="Optional LoRA adapter checkpoint to warm-start RL from (for example a CE-only ckpt_task).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.variant == "all_active":
        variants = list(SELECTED_RULE_BASED_RL_VARIANTS)
    else:
        args.variant = normalize_variant_name(args.variant)
        if args.variant not in VARIANT_TO_PROMPTS_DIR:
            raise ValueError(f"Unknown variant '{args.variant}'. Known variants: {sorted(VARIANT_TO_PROMPTS_DIR)}")
        variants = [args.variant]

    for variant_name in variants:
        if args.train_prompts_dir is not None:
            train_prompts_dir = args.train_prompts_dir
            val_prompts_dir = args.val_prompts_dir
        else:
            split_variant_root = args.split_root / variant_name
            default_train_dir = split_variant_root / "train_prompts"
            default_val_dir = split_variant_root / "val_prompts"
            if default_train_dir.exists():
                train_prompts_dir = default_train_dir
                val_prompts_dir = default_val_dir if default_val_dir.exists() else None
            else:
                train_prompts_dir = Path(VARIANT_TO_PROMPTS_DIR[variant_name])
                val_prompts_dir = args.val_prompts_dir
        output_dir = args.output_root / variant_name
        print(
            f"[RL] training variant={variant_name} train={train_prompts_dir} val={val_prompts_dir} output={output_dir}",
            flush=True,
        )
        train_rule_based_grpo(
            train_prompts_dir=train_prompts_dir,
            output_dir=output_dir,
            val_prompts_dir=val_prompts_dir,
            model_id=args.model,
            steps=args.steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            save_every=args.save_every,
            eval_every=args.eval_every,
            validation_sample_size=args.validation_sample_size,
            validation_batch_size=args.validation_batch_size,
            task_only_fraction=args.task_only_fraction,
            exact_only_fraction=args.exact_only_fraction,
            ce_weight=args.ce_weight,
            ce_decay_start_fraction=args.ce_decay_start_fraction,
            public_reward=args.public_reward,
            private_reward=args.private_reward,
            joint_task_bonus=args.joint_task_bonus,
            init_adapter_dir=args.init_adapter_dir,
        )


if __name__ == "__main__":
    main()
