#!/usr/bin/env python3
"""Run greedy inference for the standalone Goldreich parity experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.evaluation.experiment_evaluation import run_variant_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_variant_inference(
        prompts_dir=args.prompts_dir,
        responses_dir=args.responses_dir,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        resume=False,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
