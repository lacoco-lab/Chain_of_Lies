#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.config import get_default_paths
from chain_of_lies.stage1_arithmetic import (
    ARITHMETIC_VARIANT_DEFAULT,
    generate_arithmetic_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate arithmetic Chain-of-Lies prompts")
    parser.add_argument("--n", type=int, default=1, help="Number of experiments to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-summand", type=int, default=99, help="Max summand for easy expressions (default 99)")
    parser.add_argument(
        "--difficulty-variant",
        type=str,
        default=ARITHMETIC_VARIANT_DEFAULT,
        choices=["default", "public_hard", "private_hard", "both_hard"],
        help="default=both easy; public_hard=public harder; private_hard=private harder; both_hard=both harder",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: data/prompts)")
    args = parser.parse_args()

    paths = get_default_paths()
    paths.ensure_dirs()
    out_dir = args.out_dir or paths.prompts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        seed = (args.seed + i) if args.seed is not None else None
        prompt_text, experiment_id, spec = generate_arithmetic_prompt(
            max_summand=args.max_summand,
            seed=seed,
            experiment_id=f"arith_{i:04d}",
            difficulty_variant=args.difficulty_variant,
        )
        payload = {
            "task_type": "arithmetic",
            "experiment_id": experiment_id,
            "difficulty_variant": args.difficulty_variant,
            "prompt_text": prompt_text,
            "spec": {
                "public_question": spec.public_question,
                "private_question": spec.private_question,
                "public_answer": spec.public_answer,
                "private_answer": spec.private_answer,
            },
        }
        out_file = out_dir / f"{experiment_id}.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_file}")

    print(f"Generated {args.n} arithmetic prompt(s) in {out_dir}")


if __name__ == "__main__":
    main()
