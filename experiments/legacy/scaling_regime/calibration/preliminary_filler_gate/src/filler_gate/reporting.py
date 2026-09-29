from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _format_task_table(by_task: dict[str, Any]) -> list[str]:
    lines = [
        "| Task type | n | Accuracy | Format |",
        "|---|---:|---:|---:|",
    ]
    for task_type, row in sorted(by_task.items()):
        lines.append(
            f"| `{task_type}` | {row['n']} | {_pct(row['accuracy'])} | "
            f"{_pct(row['format_rate'])} |"
        )
    return lines


def build_report(
    metrics: dict[str, Any],
    *,
    run_manifest: dict[str, Any],
    plot_available: bool,
) -> str:
    knowledge = metrics["knowledge_screening"]
    filler = metrics["filler_gate"]
    lines = [
        "# Preliminary knowledge and filler gate",
        "",
        "This report covers inference-only evaluation. It contains no piggybacking, "
        "fine-tuning, or hidden-state analysis.",
        "",
        "## Run",
        "",
        f"- Model: `{run_manifest['model_id']}`",
        f"- Decoding: greedy (`do_sample=False`, temperature recorded as 0)",
        f"- Seed: `{run_manifest['seed']}`",
        f"- Atomic filler token ID: `{run_manifest['filler_token_id']}`",
        f"- Atomic filler token text: `{run_manifest['filler_token_text']!r}`",
        f"- Benchmark items: `{run_manifest['dataset_summary']['benchmark_items']}`",
        f"- Component probes: `{run_manifest['dataset_summary']['component_facts']}`",
        "",
        "## Goal 1: knowledge screening",
        "",
        f"Configured gate: **{knowledge['configured_gate']['status']}**.",
        "",
        "### Main-task baseline",
        "",
    ]
    lines.extend(_format_task_table(knowledge["main_baseline_by_task"]))
    main_overall = knowledge["main_baseline_overall"]
    component_overall = knowledge["component_facts_overall"]
    lines.extend(
        [
            "",
            f"Overall main-task baseline: **{_pct(main_overall['accuracy'])}** "
            f"({main_overall['n']} items; format {_pct(main_overall['format_rate'])}).",
            "",
            "### Individual component facts",
            "",
        ]
    )
    lines.extend(_format_task_table(knowledge["component_facts_by_type"]))
    known = knowledge["factual_main_when_all_components_known"]
    unknown = knowledge["factual_main_with_missing_or_incorrect_components"]
    lines.extend(
        [
            "",
            f"Overall component-fact accuracy: **{_pct(component_overall['accuracy'])}** "
            f"({component_overall['n']} probes; "
            f"format {_pct(component_overall['format_rate'])}).",
            "",
            "### Knowledge versus composition",
            "",
            f"- Factual main tasks with all component probes correct: "
            f"{known['n']} items, {_pct(known['accuracy'])} main-task accuracy.",
            f"- Factual main tasks with at least one missing/incorrect component: "
            f"{unknown['n']} items, {_pct(unknown['accuracy'])} main-task accuracy.",
            "",
            "Component-probe accuracy measures whether the model supplies the required "
            "facts in isolation. Main-task failures despite all components being known "
            "are evidence of composition/reasoning errors rather than missing knowledge.",
            "",
            "## Goal 2: inference-time filler capability",
            "",
            f"Configured gate: **{filler['configured_gate']['status']}**. "
            f"The preregistered comparison is K={filler['configured_gate']['primary_k']} "
            "versus K=0.",
            "",
            "| K | n | Accuracy | Δ vs K=0 | Wrong→right | Right→wrong | McNemar p | 95% paired CI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for filler_length in sorted(filler["accuracy_by_k"], key=int):
        result = filler["accuracy_by_k"][filler_length]
        paired = filler["paired_vs_k0"][filler_length]
        ci = paired["paired_bootstrap_ci"]
        lines.append(
            f"| {filler_length} | {result['n']} | {_pct(result['accuracy'])} | "
            f"{paired['accuracy_delta']:+.3f} | {paired['wrong_to_right']} | "
            f"{paired['right_to_wrong']} | {paired['mcnemar_exact_p']:.4f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "### Accuracy by task and filler length",
            "",
            "| K | Task | n | Accuracy | Format |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for filler_length in sorted(filler["accuracy_by_task_and_k"], key=int):
        for task_type, result in sorted(
            filler["accuracy_by_task_and_k"][filler_length].items()
        ):
            lines.append(
                f"| {filler_length} | `{task_type}` | {result['n']} | "
                f"{_pct(result['accuracy'])} | {_pct(result['format_rate'])} |"
            )
    if plot_available:
        lines.extend(
            [
                "",
                "![Accuracy by filler length](accuracy_by_filler.png)",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A positive point estimate at one filler length is not sufficient evidence "
            "of filler capability; use the paired gate and the full length curve.",
            "- Failure of this gate applies to this model, prompt scaffold, and benchmark. "
            "It is not evidence that filler computation is impossible in general.",
            "- Component-fact correctness is measured on separate prompts and therefore "
            "does not prove that the same fact was internally retrieved on the composite item.",
            "- Long filler may plateau or degrade. All tested lengths are reported.",
            "",
            "## Files",
            "",
            "- `raw_predictions.jsonl`: exact prompts, token IDs, generations, parses, and scores.",
            "- `detailed_predictions.csv`: compact item-level results.",
            "- `metrics.json`: complete aggregate and paired statistics.",
            "- `accuracy_by_filler.csv`: filler-length summary.",
            "- `run_manifest.json`: model, tokenizer, data, and configuration metadata.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    metrics: dict[str, Any],
    *,
    run_manifest: dict[str, Any],
    output_path: Path,
    plot_available: bool,
) -> None:
    output_path.write_text(
        build_report(
            metrics,
            run_manifest=run_manifest,
            plot_available=plot_available,
        ),
        encoding="utf-8",
    )
