#!/usr/bin/env python3
from pathlib import Path

import argparse
import sys

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))

from chain_of_lies.evaluation import VARIANT_TO_PROMPTS_DIR, normalize_variant_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CE-only LoRA training on a selected experiment variant."
    )
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/ce_only"))
    parser.add_argument(
        "--split-root", type=Path, default=Path("generated_data/prompt_splits")
    )
    parser.add_argument("--train-prompts-dir", type=Path, default=None)
    parser.add_argument("--val-prompts-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--initial-adapter-path", type=Path, default=None)
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--deterministic-training", action="store_true")
    parser.add_argument("--recovery-state-path", type=Path, default=None)
    parser.add_argument("--recovery-save-every", type=int, default=200)
    parser.add_argument(
        "--memory-efficient-ce",
        action="store_true",
        help="Compute exact supervised-token CE with a checkpointed, chunked LM head.",
    )
    parser.add_argument(
        "--ce-token-chunk-size",
        type=int,
        default=16,
        help="Maximum supervised positions projected to vocabulary logits at once.",
    )
    parser.add_argument(
        "--activation-cpu-offload",
        action="store_true",
        help="Keep autograd-saved activations on CPU until backward to reduce peak GPU memory.",
    )
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
            "filler_public_cot",
            "filler_only",
            "local_channel_cot",
            "answer_only",
            "mismatched_public_cot",
            "record_target",
        ],
        help="Supervised suffix used during CE training.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filler-token-count",
        type=int,
        default=None,
        help="Exact number of atomic filler tokens for a filler supervision mode.",
    )
    parser.add_argument(
        "--filler-token-counts-json",
        type=str,
        default=None,
        help="JSON object mapping difficulty values to fixed atomic filler-token counts.",
    )
    parser.add_argument(
        "--filler-token-count-field",
        type=str,
        default=None,
        help="Prompt spec field used with --filler-token-counts-json.",
    )
    args = parser.parse_args()

    args.variant = normalize_variant_name(args.variant)
    if args.variant not in VARIANT_TO_PROMPTS_DIR and args.train_prompts_dir is None:
        raise ValueError(
            f"Unknown variant '{args.variant}'. Known variants: {sorted(VARIANT_TO_PROMPTS_DIR)}"
        )

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
    val_count = (
        len(list(val_prompts_dir.glob("*.json"))) if val_prompts_dir is not None else 0
    )
    if (
        args.expected_train_prompts is not None
        and train_count != args.expected_train_prompts
    ):
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
    from chain_of_lies.training.ce import train_answer_ce_only

    filler_token_counts = None
    if args.filler_token_counts_json is not None:
        import json

        decoded = json.loads(args.filler_token_counts_json)
        if not isinstance(decoded, dict):
            raise ValueError("--filler-token-counts-json must decode to a JSON object.")
        filler_token_counts = {str(key): int(value) for key, value in decoded.items()}

    train_answer_ce_only(
        train_prompts_dir=train_prompts_dir,
        output_dir=output_dir,
        val_prompts_dir=val_prompts_dir,
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_new_tokens=args.max_new_tokens,
        save_every=args.save_every,
        eval_every=args.eval_every,
        validation_sample_size=args.validation_sample_size,
        validation_batch_size=args.validation_batch_size,
        supervision_mode=args.supervision_mode,
        filler_token_count=args.filler_token_count,
        filler_token_counts=filler_token_counts,
        filler_token_count_field=args.filler_token_count_field,
        seed=args.seed,
        initial_adapter_path=args.initial_adapter_path,
        save_each_epoch=args.save_each_epoch,
        deterministic_training=args.deterministic_training,
        memory_efficient_ce=args.memory_efficient_ce,
        ce_token_chunk_size=args.ce_token_chunk_size,
        activation_cpu_offload=args.activation_cpu_offload,
        recovery_state_path=args.recovery_state_path,
        recovery_save_every=args.recovery_save_every,
    )


if __name__ == "__main__":
    main()
