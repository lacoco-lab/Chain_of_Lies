import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl import SELECTED_RULE_BASED_RL_VARIANTS, VARIANT_TO_PROMPTS_DIR, train_rule_based_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based GRPO-style RL on a selected experiment variant.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all_selected",
        help="Variant name to train on, or 'all_selected' to run the current top variants sequentially.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rule_based_rl"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.variant == "all_selected":
        variants = list(SELECTED_RULE_BASED_RL_VARIANTS)
    else:
        if args.variant not in VARIANT_TO_PROMPTS_DIR:
            raise ValueError(f"Unknown variant '{args.variant}'. Known variants: {sorted(VARIANT_TO_PROMPTS_DIR)}")
        variants = [args.variant]

    for variant_name in variants:
        prompts_dir = Path(VARIANT_TO_PROMPTS_DIR[variant_name])
        output_dir = args.output_root / variant_name
        print(f"[RL] training variant={variant_name} prompts={prompts_dir} output={output_dir}", flush=True)
        train_rule_based_grpo(
            prompts_dir=prompts_dir,
            output_dir=output_dir,
            model_id=args.model,
            steps=args.steps,
            batch_size=args.batch_size,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            save_every=args.save_every,
        )


if __name__ == "__main__":
    main()
