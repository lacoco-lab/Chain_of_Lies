#!/usr/bin/env python3
from pathlib import Path

import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.ce_only import train_answer_ce_only
from chain_of_lies.rl import VARIANT_TO_PROMPTS_DIR, normalize_variant_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CE-only LoRA training on a selected experiment variant.")
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/ce_only"))
    parser.add_argument("--split-root", type=Path, default=Path("data/RL_splits"))
    parser.add_argument("--train-prompts-dir", type=Path, default=None)
    parser.add_argument("--val-prompts-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-sample-size", type=int, default=1000)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--expected-train-prompts", type=int, default=None)
    parser.add_argument("--expected-val-prompts", type=int, default=None)
    parser.add_argument(
        "--supervision-mode",
        type=str,
        default="public_cot",
        choices=[
            "public_cot",
            "verbose_public_cot",
            "answer_only",
            "mismatched_public_cot",
        ],
        help="Supervised suffix used during CE training.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.variant = normalize_variant_name(args.variant)
    if args.variant not in VARIANT_TO_PROMPTS_DIR:
        raise ValueError(f"Unknown variant '{args.variant}'. Known variants: {sorted(VARIANT_TO_PROMPTS_DIR)}")

    if args.train_prompts_dir is not None:
        train_prompts_dir = args.train_prompts_dir
        val_prompts_dir = args.val_prompts_dir
    else:
        split_variant_root = args.split_root / args.variant
        default_train_dir = split_variant_root / "train_prompts"
        default_val_dir = split_variant_root / "val_prompts"
        if default_train_dir.exists():
            train_prompts_dir = default_train_dir
            val_prompts_dir = default_val_dir if default_val_dir.exists() else None
        else:
            train_prompts_dir = Path(VARIANT_TO_PROMPTS_DIR[args.variant])
            val_prompts_dir = args.val_prompts_dir

    output_dir = args.output_root / args.variant
    train_count = len(list(train_prompts_dir.glob("*.json")))
    val_count = len(list(val_prompts_dir.glob("*.json"))) if val_prompts_dir is not None else 0
    if args.expected_train_prompts is not None and train_count != args.expected_train_prompts:
        raise ValueError(
            f"Expected {args.expected_train_prompts} train prompts in {train_prompts_dir}, "
            f"found {train_count}. Regenerate splits before training."
        )
    if args.expected_val_prompts is not None and val_count != args.expected_val_prompts:
        raise ValueError(
            f"Expected {args.expected_val_prompts} validation prompts in {val_prompts_dir}, "
            f"found {val_count}. Regenerate splits before training."
        )
    print(
        f"[CE] training variant={args.variant} train={train_prompts_dir} "
        f"val={val_prompts_dir} train_count={train_count} val_count={val_count} "
        f"output={output_dir} supervision_mode={args.supervision_mode}",
        flush=True,
    )
    train_answer_ce_only(
        train_prompts_dir=train_prompts_dir,
        output_dir=output_dir,
        val_prompts_dir=val_prompts_dir,
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        save_every=args.save_every,
        eval_every=args.eval_every,
        validation_sample_size=args.validation_sample_size,
        validation_batch_size=args.validation_batch_size,
        supervision_mode=args.supervision_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
