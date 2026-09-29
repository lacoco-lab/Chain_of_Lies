from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .analysis import (
    compute_metrics,
    maybe_write_plot,
    read_predictions,
    write_accuracy_csv,
    write_detailed_csv,
)
from .data import (
    dataset_summary,
    group_demonstrations,
    load_benchmark,
    load_components,
    load_demonstrations,
    select_items,
    validate_cross_references,
    validate_gold_semantics,
)
from .inference import clear_model, load_model_and_tokenizer, run_prepared_prompts
from .prompting import (
    make_component_spec,
    make_main_spec,
    prepare_prompt,
    resolve_atomic_filler_token,
    validate_non_filler_invariance,
)
from .reporting import write_report

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.json"


def _read_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    source_bytes = config_path.read_bytes()
    config = json.loads(source_bytes.decode("utf-8"))
    base = config_path.parent
    data_config = config["data"]
    for key in ("benchmark_path", "component_path", "demonstrations_path"):
        candidate = Path(data_config[key])
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        data_config[key] = str(candidate)
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (base / output_root).resolve()
    config["output_root"] = str(output_root)
    config["_config_path"] = str(config_path)
    config["_source_config_sha256"] = sha256(source_bytes).hexdigest()
    return config


def _load_all_data(config: dict[str, Any]):
    data_config = config["data"]
    items = load_benchmark(Path(data_config["benchmark_path"]))
    components = load_components(Path(data_config["component_path"]))
    demonstrations = load_demonstrations(Path(data_config["demonstrations_path"]))
    validate_cross_references(items, components)
    validate_gold_semantics(items, components)
    selected = select_items(
        items,
        limit_per_task=data_config.get("limit_per_task"),
        seed=int(config["seed"]),
    )
    demos_by_task = group_demonstrations(
        demonstrations,
        per_task=int(data_config.get("few_shot_per_task", 5)),
    )
    return selected, components, demonstrations, demos_by_task


def _config_hash(config: dict[str, Any]) -> str:
    source_hash = config.get("_source_config_sha256")
    if source_hash:
        return str(source_hash)
    rendered = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(rendered.encode("utf-8")).hexdigest()


def _make_specs(config, items, components, demos_by_task):
    filler_config = config["filler"]
    filler_lengths = [int(value) for value in filler_config["lengths"]]
    if not filler_lengths or 0 not in filler_lengths:
        raise ValueError("Filler lengths must include the K=0 baseline")
    if len(set(filler_lengths)) != len(filler_lengths):
        raise ValueError("Filler lengths must be unique")
    if any(value < 0 for value in filler_lengths):
        raise ValueError("Filler lengths must be non-negative")

    specs = []
    for item in items:
        for filler_length in filler_lengths:
            specs.append(
                make_main_spec(
                    item,
                    demos_by_task[item.task_type],
                    filler_length=filler_length,
                )
            )
    component_lengths = (
        filler_lengths
        if filler_config.get("evaluate_components_at_all_lengths", False)
        else [0]
    )
    for component in components:
        for filler_length in component_lengths:
            specs.append(make_component_spec(component, filler_length=filler_length))
    return specs


def _build_manifest(
    config: dict[str, Any],
    *,
    tokenizer: Any,
    filler_token_id: int,
    filler_token_text: str,
    summary: dict[str, Any],
    specs_count: int,
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    try:
        import transformers
    except ModuleNotFoundError:
        transformers = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "config_path": config["_config_path"],
        "config_sha256": _config_hash(config),
        "seed": int(config["seed"]),
        "model_id": config["model"]["model_id"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "filler_token_id": filler_token_id,
        "filler_token_text": filler_token_text,
        "filler_lengths": config["filler"]["lengths"],
        "prepared_prompt_count": specs_count,
        "dataset_summary": summary,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "decoding": {
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": config["model"]["max_new_tokens"],
        },
    }


def _write_analysis(
    run_dir: Path,
    config: dict[str, Any],
    items,
    components,
) -> dict[str, Any]:
    predictions = read_predictions(run_dir / "raw_predictions.jsonl")
    metrics = compute_metrics(
        predictions,
        items,
        components,
        analysis_config=config["analysis"],
        seed=int(config["seed"]),
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_detailed_csv(predictions, run_dir / "detailed_predictions.csv")
    write_accuracy_csv(metrics, run_dir / "accuracy_by_filler.csv")
    plot_error = maybe_write_plot(metrics, run_dir / "accuracy_by_filler.png")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if plot_error:
        manifest["plot_warning"] = plot_error
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    write_report(
        metrics,
        run_manifest=manifest,
        output_path=run_dir / "REPORT.md",
        plot_available=plot_error is None,
    )
    return metrics


def command_validate(config_path: Path) -> None:
    config = _read_config(config_path)
    items, components, demonstrations, demos_by_task = _load_all_data(config)
    specs = _make_specs(config, items, components, demos_by_task)
    summary = dataset_summary(items, components, demonstrations)
    print(json.dumps({"status": "ok", **summary, "prompt_specs": len(specs)}, indent=2))


def command_run(
    config_path: Path,
    *,
    output_dir: Path | None,
    no_resume: bool,
) -> None:
    config = _read_config(config_path)
    items, components, demonstrations, demos_by_task = _load_all_data(config)
    specs = _make_specs(config, items, components, demos_by_task)
    summary = dataset_summary(items, components, demonstrations)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(config["output_root"]) / f"{config['experiment_name']}_{timestamp}"
        )
    run_dir = output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_config_path = run_dir / "config_resolved.json"
    existing_predictions_path = run_dir / "raw_predictions.jsonl"
    if (
        not no_resume
        and existing_predictions_path.exists()
        and existing_config_path.exists()
    ):
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if _config_hash(existing_config) != _config_hash(config):
            raise ValueError(
                "Refusing to resume into a run directory created with a different "
                "configuration. Choose another --output-dir or use --no-resume "
                "to intentionally replace the predictions."
            )
    elif (
        not no_resume
        and existing_predictions_path.exists()
        and not existing_config_path.exists()
    ):
        raise ValueError(
            "Cannot safely resume: raw_predictions.jsonl exists but "
            "config_resolved.json is missing"
        )
    (run_dir / "config_resolved.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model, tokenizer = load_model_and_tokenizer(config["model"])
    filler_token_id, filler_token_text = resolve_atomic_filler_token(
        tokenizer,
        list(config["filler"]["preferred_atomic_strings"]),
    )
    validate_non_filler_invariance(
        tokenizer,
        specs,
        filler_token_id=filler_token_id,
        filler_token_text=filler_token_text,
    )
    prepared = [
        prepare_prompt(
            tokenizer,
            spec,
            filler_token_id=filler_token_id,
            filler_token_text=filler_token_text,
        )
        for spec in specs
    ]
    manifest = _build_manifest(
        config,
        tokenizer=tokenizer,
        filler_token_id=filler_token_id,
        filler_token_text=filler_token_text,
        summary=summary,
        specs_count=len(prepared),
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    run_prepared_prompts(
        model,
        tokenizer,
        prepared,
        model_config=config["model"],
        output_path=run_dir / "raw_predictions.jsonl",
        resume=not no_resume,
    )
    clear_model(model, tokenizer)
    metrics = _write_analysis(run_dir, config, items, components)
    print(f"[done] report: {run_dir / 'REPORT.md'}")
    print(
        "[done] knowledge gate:",
        metrics["knowledge_screening"]["configured_gate"]["status"],
    )
    print(
        "[done] filler gate:",
        metrics["filler_gate"]["configured_gate"]["status"],
    )


def command_report(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    config = json.loads((resolved / "config_resolved.json").read_text(encoding="utf-8"))
    items, components, _demonstrations, _demos_by_task = _load_all_data(config)
    metrics = _write_analysis(resolved, config, items, components)
    print(f"[done] regenerated {resolved / 'REPORT.md'}")
    print(
        json.dumps(
            {
                "knowledge_gate": metrics["knowledge_screening"]["configured_gate"][
                    "status"
                ],
                "filler_gate": metrics["filler_gate"]["configured_gate"]["status"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent Qwen knowledge screen and inference-time filler capability gate"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate data, cross-references, configuration, and prompt counts",
    )
    validate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    run_parser = subparsers.add_parser(
        "run",
        help="Run deterministic model inference and produce all reports",
    )
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit run directory; useful for resumable cluster jobs",
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite predictions instead of skipping completed prediction keys",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Regenerate metrics and report from an existing run",
    )
    report_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        command_validate(args.config)
    elif args.command == "run":
        command_run(
            args.config,
            output_dir=args.output_dir,
            no_resume=args.no_resume,
        )
    elif args.command == "report":
        command_report(args.run_dir)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
