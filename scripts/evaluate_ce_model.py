import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.evaluation import (
    SELECTED_ACTIVE_VARIANTS,
    compare_variant_results,
    normalize_variant_name,
    run_variant_inference,
    summarize_training_history,
)


def _print_variant_report(report: dict) -> None:
    """Print a per-variant report.

    Reported task metrics:
      - `success`        : strict joint exact (task_success_rate)
      - `subgoal`        : per-side exact partial credit (task_subgoal_rate)
      - `pub_exact`      : public-side exact rate
      - `priv_exact`     : private-side exact rate

    `task_component_rate` is intentionally omitted from the printed report. It is still tracked
    inside the raw summary JSONs (and is used internally for `ckpt_task` selection), but it
    overstates true task quality whenever public/private answers are numerically close - which
    masks the dominant failure mode in piggybacking. Per-side exact rates make the same
    information directly readable.
    """
    baseline = report["baseline"]
    rl = report["rl"]
    delta = report["delta"]

    print(report["variant_name"])
    print(
        "  baseline:"
        f" reward={baseline.get('avg_reward', 0.0):.2f}"
        f" success={baseline.get('task_success_rate', 0.0):.2f}"
        f" subgoal={baseline.get('task_subgoal_rate', 0.0):.2f}"
        f" pub_exact={baseline.get('public_exact_rate', 0.0):.2f}"
        f" priv_exact={baseline.get('private_exact_rate', 0.0):.2f}"
        f" concealment={baseline.get('concealment_rate', 0.0):.2f}"
        f" format={baseline.get('format_rate', 0.0):.2f}"
        f" cot_words={baseline.get('avg_cot_words', 0.0):.1f}"
    )
    print(
        "  rl:"
        f" reward={rl.get('avg_reward', 0.0):.2f}"
        f" success={rl.get('task_success_rate', 0.0):.2f}"
        f" subgoal={rl.get('task_subgoal_rate', 0.0):.2f}"
        f" pub_exact={rl.get('public_exact_rate', 0.0):.2f}"
        f" priv_exact={rl.get('private_exact_rate', 0.0):.2f}"
        f" concealment={rl.get('concealment_rate', 0.0):.2f}"
        f" format={rl.get('format_rate', 0.0):.2f}"
        f" cot_words={rl.get('avg_cot_words', 0.0):.1f}"
    )
    print(
        "  delta:"
        f" reward={delta['avg_reward']:+.2f}"
        f" success={delta['task_success_rate']:+.2f}"
        f" subgoal={delta['task_subgoal_rate']:+.2f}"
        f" pub_exact={delta.get('public_exact_rate', 0.0):+.2f}"
        f" priv_exact={delta.get('private_exact_rate', 0.0):+.2f}"
        f" concealment={delta['concealment_rate']:+.2f}"
        f" format={delta['format_rate']:+.2f}"
        f" cot_words={delta['avg_cot_words']:+.1f}"
    )


def _print_compact_summary(reports: list[dict]) -> None:
    """Print a compact table of all variants for quick scanning."""
    header = (
        f"{'variant':<30} {'source':<10} {'success':>8} {'subgoal':>8} "
        f"{'pub_ex':>8} {'priv_ex':>8} {'conceal':>8} {'reward':>8}"
    )
    print("\n" + "=" * len(header))
    print("COMPACT SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for report in reports:
        name = report["variant_name"]
        for source_key, source_label in [("baseline", "baseline"), ("rl", "rl")]:
            data = report[source_key]
            print(
                f"{name:<30} {source_label:<10}"
                f" {data.get('task_success_rate', 0.0):>8.2f}"
                f" {data.get('task_subgoal_rate', 0.0):>8.2f}"
                f" {data.get('public_exact_rate', 0.0):>8.2f}"
                f" {data.get('private_exact_rate', 0.0):>8.2f}"
                f" {data.get('concealment_rate', 0.0):>8.2f}"
                f" {data.get('avg_reward', 0.0):>8.2f}"
            )
    print("=" * len(header))


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


def _assert_training_complete(adapter_dir: Path, *, require_full_run: bool = True) -> None:
    metadata_path = adapter_dir / "training_metadata.json"
    history_path = adapter_dir / "train_history.json"
    if not metadata_path.exists():
        raise ValueError(f"Missing training metadata: {metadata_path}")
    if not history_path.exists():
        raise ValueError(f"Missing training history: {history_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"Empty training history: {history_path}")

    expected_steps = int(metadata.get("steps", 0))
    last_step = int(history[-1].get("step", 0))
    train_examples_seen = int(metadata.get("train_examples_seen", 0))
    target_train_examples_seen = int(metadata.get("target_train_examples_seen", 0))
    selected_step = int(
        metadata.get("best_validation_task_step")
        or metadata.get("best_validation_step")
        or last_step
    )

    if last_step < selected_step:
        raise ValueError(
            f"Inconsistent checkpoint artifact for {adapter_dir.name}: "
            f"history stopped at step {last_step} before selected_step={selected_step}."
        )

    if train_examples_seen <= 0:
        raise ValueError(
            f"Incomplete training artifact for {adapter_dir.name}: "
            f"train_examples_seen={train_examples_seen}."
        )

    if not require_full_run:
        return

    if last_step < expected_steps:
        raise ValueError(
            f"Incomplete training artifact for {adapter_dir.name}: "
            f"history stopped at step {last_step} but expected {expected_steps}. "
            "Do not evaluate this checkpoint; reset and rerun training first."
        )
    if train_examples_seen < target_train_examples_seen:
        raise ValueError(
            f"Incomplete training artifact for {adapter_dir.name}: "
            f"train_examples_seen={train_examples_seen} < target_train_examples_seen={target_train_examples_seen}. "
            "Do not evaluate this checkpoint; reset and rerun training first."
        )


def _resolve_checkpoint_dir(adapter_dir: Path, canonical_checkpoint: str) -> Path:
    checkpoint_lookup = {
        "ckpt_reward": ("ckpt_reward", "best_checkpoint"),
        "ckpt_task": ("ckpt_task", "best_task_checkpoint"),
    }
    candidate_names = checkpoint_lookup.get(canonical_checkpoint)
    if candidate_names is None:
        return adapter_dir
    for name in candidate_names:
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained adapters against active arithmetic variants.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/ce_training"),
        help="Directory containing one CE adapter subdirectory per variant.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all_active",
        help="Variant name, comma-separated variants, 'all_active', or 'all_artifacts'.",
    )
    parser.add_argument(
        "--baseline-data-root",
        type=Path,
        default=Path("generated_data/legacy_without_self_eval"),
        help="Root directory of the original prompts and baseline responses.",
    )
    parser.add_argument(
        "--eval-responses-root",
        type=Path,
        default=Path("generated_data/eval_responses"),
        help="Base directory where checkpoint-specific evaluation responses are written.",
    )
    parser.add_argument(
        "--model-responses-subdir",
        type=str,
        default="rl",
        help="Subdirectory name used for the trained model responses under eval-responses-root.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding during held-out evaluation (equivalent to do_sample=False).",
    )
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate CE responses even if JSONs already exist.")
    parser.add_argument("--skip-inference", action="store_true", help="Only compare already-generated CE responses.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt_reward",
        # Accept both new and legacy names so older callers/CI keep working.
        # New: ckpt_reward (formerly B/best), ckpt_task (formerly A/best_task).
        choices=["ckpt_reward", "ckpt_task", "latest", "best", "best_task"],
        help=(
            "Which adapter checkpoint to evaluate. "
            "ckpt_reward (formerly 'best' / B): selected by best validation total reward. "
            "ckpt_task (formerly 'best_task' / A): selected by best validation task_component_rate. "
            "latest: the most recently saved adapter (no per-step selection)."
        ),
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
        default=Path("artifacts/ce_training/per_variant_eval"),
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

    # Normalize legacy checkpoint names to the new canonical names so a single
    # `--checkpoint best` call (e.g. from older shell scripts) keeps working.
    legacy_checkpoint_aliases = {"best": "ckpt_reward", "best_task": "ckpt_task"}
    canonical_checkpoint = legacy_checkpoint_aliases.get(args.checkpoint, args.checkpoint)
    eval_responses_root = args.eval_responses_root / canonical_checkpoint

    if args.variant == "all_active":
        variant_names = list(SELECTED_ACTIVE_VARIANTS)
    elif args.variant == "all_artifacts":
        variant_names = None
    else:
        variant_names = [normalize_variant_name(item.strip()) for item in args.variant.split(",") if item.strip()]

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
        checkpoint_dir = _resolve_checkpoint_dir(adapter_dir, canonical_checkpoint)

        metadata_path = checkpoint_dir / "training_metadata.json"
        if not metadata_path.exists():
            print(f"[Eval] skipping {adapter_dir.name}: no training_metadata.json", flush=True)
            continue

        try:
            _assert_training_complete(adapter_dir, require_full_run=True)
            metadata_source_dir = adapter_dir
        except ValueError as root_exc:
            if checkpoint_dir == adapter_dir:
                raise
            _assert_training_complete(checkpoint_dir, require_full_run=False)
            print(
                f"[Eval] warning: top-level artifact for {adapter_dir.name} looks incomplete; "
                f"using selected checkpoint artifact {checkpoint_dir.name} instead: {root_exc}",
                flush=True,
            )
            metadata_source_dir = checkpoint_dir

        metadata = json.loads((metadata_source_dir / "training_metadata.json").read_text(encoding="utf-8"))
        prompts_dir = Path(
            metadata.get("val_prompts_dir")
            or metadata.get("prompts_dir")
            or metadata.get("train_prompts_dir")
        )
        baseline_model_id = metadata["base_model"]
        baseline_responses_dir = eval_responses_root / "baseline" / adapter_dir.name
        rl_responses_dir = eval_responses_root / args.model_responses_subdir / adapter_dir.name

        if not args.skip_inference:
            run_variant_inference(
                prompts_dir=prompts_dir,
                responses_dir=baseline_responses_dir,
                model_id=baseline_model_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=not args.greedy if args.greedy else None,
                resume=not args.no_resume,
                batch_size=args.inference_batch_size,
            )
            run_variant_inference(
                prompts_dir=prompts_dir,
                responses_dir=rl_responses_dir,
                model_id=str(checkpoint_dir),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=not args.greedy if args.greedy else None,
                resume=not args.no_resume,
                batch_size=args.inference_batch_size,
            )

        report = compare_variant_results(
            prompts_dir=prompts_dir,
            baseline_responses_dir=baseline_responses_dir,
            rl_responses_dir=rl_responses_dir,
        )
        report["training"] = summarize_training_history(metadata_source_dir)
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
    output_path = args.json_out or (args.artifacts_root / f"evaluation_summary_{canonical_checkpoint}.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote evaluation summary to {output_path}")

    if len(reports) > 1:
        _print_compact_summary(reports)


if __name__ == "__main__":
    main()
