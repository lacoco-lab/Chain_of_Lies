import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl import compare_variant_results, run_variant_inference, summarize_training_history


def _print_variant_report(report: dict) -> None:
    baseline = report["baseline"]
    rl = report["rl"]
    delta = report["delta"]

    print(report["variant_name"])
    print(
        "  baseline:"
        f" reward={baseline.get('avg_reward', 0.0):.2f}"
        f" success={baseline.get('task_success_rate', 0.0):.2f}"
        f" concealment={baseline.get('concealment_rate', 0.0):.2f}"
        f" format={baseline.get('format_rate', 0.0):.2f}"
        f" cot_words={baseline.get('avg_cot_words', 0.0):.1f}"
    )
    print(
        "  rl:"
        f" reward={rl.get('avg_reward', 0.0):.2f}"
        f" success={rl.get('task_success_rate', 0.0):.2f}"
        f" concealment={rl.get('concealment_rate', 0.0):.2f}"
        f" format={rl.get('format_rate', 0.0):.2f}"
        f" cot_words={rl.get('avg_cot_words', 0.0):.1f}"
    )
    print(
        "  delta:"
        f" reward={delta['avg_reward']:+.2f}"
        f" success={delta['task_success_rate']:+.2f}"
        f" concealment={delta['concealment_rate']:+.2f}"
        f" format={delta['format_rate']:+.2f}"
        f" cot_words={delta['avg_cot_words']:+.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based RL adapters against the original selected variants.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/rule_based_rl"),
        help="Directory containing one RL adapter subdirectory per variant.",
    )
    parser.add_argument(
        "--baseline-data-root",
        type=Path,
        default=Path("data/Without_self_eval"),
        help="Root directory of the original prompts and baseline responses.",
    )
    parser.add_argument(
        "--eval-responses-root",
        type=Path,
        default=Path("data/Without_self_eval/responses_rule_based_rl"),
        help="Where to write RL evaluation responses.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate RL responses even if JSONs already exist.")
    parser.add_argument("--skip-inference", action="store_true", help="Only compare already-generated RL responses.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON summary report.")
    args = parser.parse_args()

    if not args.artifacts_root.exists():
        raise ValueError(f"Artifacts root does not exist: {args.artifacts_root}")

    reports = []
    for adapter_dir in sorted(path for path in args.artifacts_root.iterdir() if path.is_dir()):
        metadata_path = adapter_dir / "training_metadata.json"
        if not metadata_path.exists():
            print(f"[Eval] skipping {adapter_dir.name}: no training_metadata.json", flush=True)
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompts_dir = Path(metadata["prompts_dir"])
        baseline_responses_dir = args.baseline_data_root / f"responses_{adapter_dir.name.removeprefix('graph_').removeprefix('arith_')}"
        # Use explicit mapping because the baseline directory names are not all derived the same way.
        baseline_lookup = {
            "graph_default": args.baseline_data_root / "responses_default",
            "graph_benign_malign": args.baseline_data_root / "responses_benign_malign",
            "arith_private_hard": args.baseline_data_root / "responses_arithmetic_private_hard",
            "arith_both_hard": args.baseline_data_root / "responses_arithmetic_both_hard",
        }
        baseline_responses_dir = baseline_lookup.get(adapter_dir.name, baseline_responses_dir)
        rl_responses_dir = args.eval_responses_root / adapter_dir.name

        if not args.skip_inference:
            run_variant_inference(
                prompts_dir=prompts_dir,
                responses_dir=rl_responses_dir,
                model_id=str(adapter_dir),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                resume=not args.no_resume,
            )

        report = compare_variant_results(
            prompts_dir=prompts_dir,
            baseline_responses_dir=baseline_responses_dir,
            rl_responses_dir=rl_responses_dir,
        )
        report["training"] = summarize_training_history(adapter_dir)
        reports.append(report)
        _print_variant_report(report)
        print()

    output = {
        "artifacts_root": str(args.artifacts_root),
        "baseline_data_root": str(args.baseline_data_root),
        "eval_responses_root": str(args.eval_responses_root),
        "reports": reports,
    }
    output_path = args.json_out or (args.artifacts_root / "evaluation_summary.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
