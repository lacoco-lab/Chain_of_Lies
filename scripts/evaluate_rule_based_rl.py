import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl import (
    SELECTED_RULE_BASED_RL_VARIANTS,
    compare_variant_results,
    run_variant_inference,
    summarize_training_history,
)


def _print_variant_report(report: dict) -> None:
    baseline = report["baseline"]
    rl = report["rl"]
    delta = report["delta"]

    print(report["variant_name"])
    print(
        "  baseline:"
        f" reward={baseline.get('avg_reward', 0.0):.2f}"
        f" success={baseline.get('task_success_rate', 0.0):.2f}"
        f" subgoal={baseline.get('task_subgoal_rate', 0.0):.2f}"
        f" task_component={baseline.get('task_component_rate', 0.0):.2f}"
        f" concealment={baseline.get('concealment_rate', 0.0):.2f}"
        f" format={baseline.get('format_rate', 0.0):.2f}"
        f" cot_words={baseline.get('avg_cot_words', 0.0):.1f}"
    )
    print(
        "  rl:"
        f" reward={rl.get('avg_reward', 0.0):.2f}"
        f" success={rl.get('task_success_rate', 0.0):.2f}"
        f" subgoal={rl.get('task_subgoal_rate', 0.0):.2f}"
        f" task_component={rl.get('task_component_rate', 0.0):.2f}"
        f" concealment={rl.get('concealment_rate', 0.0):.2f}"
        f" format={rl.get('format_rate', 0.0):.2f}"
        f" cot_words={rl.get('avg_cot_words', 0.0):.1f}"
    )
    print(
        "  delta:"
        f" reward={delta['avg_reward']:+.2f}"
        f" success={delta['task_success_rate']:+.2f}"
        f" subgoal={delta['task_subgoal_rate']:+.2f}"
        f" task_component={delta['task_component_rate']:+.2f}"
        f" concealment={delta['concealment_rate']:+.2f}"
        f" format={delta['format_rate']:+.2f}"
        f" cot_words={delta['avg_cot_words']:+.1f}"
    )


def _aggregate_reports(
    *,
    input_root: Path,
    artifacts_root: Path,
) -> None:
    if not input_root.exists():
        raise ValueError(f"Input root does not exist: {input_root}")

    for checkpoint_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        reports: list[dict] = []
        artifacts_root_value = None
        baseline_data_root = None
        eval_responses_root = None

        for summary_path in sorted(checkpoint_dir.glob("*.json")):
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            if artifacts_root_value is None:
                artifacts_root_value = data.get("artifacts_root")
                baseline_data_root = data.get("baseline_data_root")
                eval_responses_root = data.get("eval_responses_root")
            reports.extend(data.get("reports", []))

        if not reports:
            print(f"[aggregate] skip {checkpoint_dir.name}: no reports found", flush=True)
            continue

        reports.sort(key=lambda item: item.get("variant_name", ""))
        output = {
            "artifacts_root": artifacts_root_value or str(artifacts_root),
            "baseline_data_root": baseline_data_root,
            "eval_responses_root": eval_responses_root,
            "reports": reports,
        }
        output_path = artifacts_root / f"evaluation_summary_{checkpoint_dir.name}.json"
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"[aggregate] wrote {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based RL adapters against the current selected variants.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/rule_based_rl"),
        help="Directory containing one RL adapter subdirectory per variant.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all_selected",
        help="Variant name, comma-separated variants, 'all_selected', or 'all_artifacts'.",
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
        default=Path("data/rl_eval_responses"),
        help="Base directory where checkpoint-specific evaluation responses are written.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate RL responses even if JSONs already exist.")
    parser.add_argument("--skip-inference", action="store_true", help="Only compare already-generated RL responses.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "best_task", "latest"],
        help="Use the best validation reward checkpoint, best task checkpoint, or the latest saved adapter directory.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON summary report.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate completed per-variant summaries into evaluation_summary_<checkpoint>.json files.",
    )
    parser.add_argument(
        "--aggregate-input-root",
        type=Path,
        default=Path("artifacts/rule_based_rl/per_variant_eval"),
        help="Input directory for --aggregate-only.",
    )
    args = parser.parse_args()

    if args.aggregate_only:
        _aggregate_reports(
            input_root=args.aggregate_input_root,
            artifacts_root=args.artifacts_root,
        )
        return

    if not args.artifacts_root.exists():
        raise ValueError(f"Artifacts root does not exist: {args.artifacts_root}")

    eval_responses_root = args.eval_responses_root / args.checkpoint

    if args.variant == "all_selected":
        variant_names = list(SELECTED_RULE_BASED_RL_VARIANTS)
    elif args.variant == "all_artifacts":
        variant_names = None
    else:
        variant_names = [item.strip() for item in args.variant.split(",") if item.strip()]

    adapter_dirs = sorted(path for path in args.artifacts_root.iterdir() if path.is_dir())
    if variant_names is not None:
        adapter_lookup = {path.name: path for path in adapter_dirs}
        adapter_dirs = [adapter_lookup[name] for name in variant_names if name in adapter_lookup]
        missing_variants = [name for name in variant_names if name not in adapter_lookup]
        for name in missing_variants:
            print(f"[Eval] skipping {name}: no adapter directory in {args.artifacts_root}", flush=True)
    if not adapter_dirs:
        raise ValueError(f"No adapter directories selected under {args.artifacts_root}")

    reports = []
    for adapter_dir in adapter_dirs:
        metadata_path = adapter_dir / "training_metadata.json"
        if not metadata_path.exists():
            print(f"[Eval] skipping {adapter_dir.name}: no training_metadata.json", flush=True)
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompts_dir = Path(
            metadata.get("val_prompts_dir")
            or metadata.get("prompts_dir")
            or metadata.get("train_prompts_dir")
        )
        baseline_model_id = metadata["base_model"]
        if args.checkpoint == "best" and (adapter_dir / "best_checkpoint").exists():
            checkpoint_dir = adapter_dir / "best_checkpoint"
        elif args.checkpoint == "best_task" and (adapter_dir / "best_task_checkpoint").exists():
            checkpoint_dir = adapter_dir / "best_task_checkpoint"
        else:
            checkpoint_dir = adapter_dir
        baseline_responses_dir = eval_responses_root / "baseline" / adapter_dir.name
        rl_responses_dir = eval_responses_root / "rl" / adapter_dir.name

        if not args.skip_inference:
            run_variant_inference(
                prompts_dir=prompts_dir,
                responses_dir=baseline_responses_dir,
                model_id=baseline_model_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                resume=not args.no_resume,
                batch_size=args.inference_batch_size,
            )
            run_variant_inference(
                prompts_dir=prompts_dir,
                responses_dir=rl_responses_dir,
                model_id=str(checkpoint_dir),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                resume=not args.no_resume,
                batch_size=args.inference_batch_size,
            )

        report = compare_variant_results(
            prompts_dir=prompts_dir,
            baseline_responses_dir=baseline_responses_dir,
            rl_responses_dir=rl_responses_dir,
        )
        report["training"] = summarize_training_history(adapter_dir)
        report["evaluation_prompts_dir"] = str(prompts_dir)
        report["evaluated_checkpoint"] = str(checkpoint_dir)
        reports.append(report)
        _print_variant_report(report)
        print()

    output = {
        "artifacts_root": str(args.artifacts_root),
        "baseline_data_root": str(args.baseline_data_root),
        "eval_responses_root": str(eval_responses_root),
        "reports": reports,
    }
    output_path = args.json_out or (args.artifacts_root / f"evaluation_summary_{args.checkpoint}.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
