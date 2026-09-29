#!/usr/bin/env python3
"""Summarize pretrained-versus-fine-tuned regime evaluations across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "task_success_rate",
    "public_exact_rate",
    "private_exact_rate",
    "concealment_rate",
    "format_rate",
    "avg_cot_words",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def collect_rows(
    root: Path,
    *,
    modes: tuple[str, ...],
    variants: tuple[str, ...],
    checkpoint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_root in sorted(path for path in root.glob("seed_*") if path.is_dir()):
        seed = seed_root.name.removeprefix("seed_")
        for mode in modes:
            for variant in variants:
                adapter_variant = (
                    "knowledge_easy_1fact"
                    if variant == "knowledge_easy_components"
                    else variant
                )
                summary_path = (
                    seed_root
                    / mode
                    / adapter_variant
                    / "per_variant_eval"
                    / checkpoint
                    / f"{variant}.json"
                )
                if not summary_path.exists():
                    rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "variant": variant,
                            "status": "missing",
                        }
                    )
                    continue
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                reports = payload.get("reports", [])
                if not reports:
                    rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "variant": variant,
                            "status": "missing",
                        }
                    )
                    continue
                report = reports[0]
                row: dict[str, Any] = {
                    "seed": seed,
                    "mode": mode,
                    "variant": variant,
                    "status": "ok",
                }
                for metric in METRICS:
                    pretrained = report.get("baseline", {}).get(metric)
                    finetuned = report.get("rl", {}).get(metric)
                    row[f"pretrained_{metric}"] = pretrained
                    row[f"finetuned_{metric}"] = finetuned
                    if pretrained is not None and finetuned is not None:
                        row[f"delta_{metric}"] = finetuned - pretrained
                rows.append(row)
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    modes: tuple[str, ...],
    variants: tuple[str, ...],
) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for mode in modes:
        for variant in variants:
            subset = [
                row
                for row in rows
                if row.get("status") == "ok"
                and row.get("mode") == mode
                and row.get("variant") == variant
            ]
            output: dict[str, Any] = {
                "mode": mode,
                "variant": variant,
                "status": "ok" if subset else "missing",
                "n_seeds": len(subset),
            }
            for metric in METRICS:
                for prefix in ("pretrained", "finetuned", "delta"):
                    key = f"{prefix}_{metric}"
                    values = [
                        float(row[key]) for row in subset if row.get(key) is not None
                    ]
                    if values:
                        mean, std = _mean_std(values)
                        output[f"{key}_mean"] = mean
                        output[f"{key}_std"] = std
            aggregate.append(output)
    return aggregate


def _format_rate(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def write_report(
    path: Path, aggregate: list[dict[str, Any]], *, checkpoint: str
) -> None:
    lines = [
        "# Regime Experiment Report",
        "",
        f"Primary checkpoint: `{checkpoint}` (fixed final epoch).",
        "",
        "| Mode | Variant | Seeds | Pretrained joint | Fine-tuned joint | Δ joint | Pretrained private | Fine-tuned private | Δ private |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {mode} | {variant} | {n_seeds} | {pre_joint} | {post_joint} | {delta_joint} | "
            "{pre_private} | {post_private} | {delta_private} |".format(
                mode=row["mode"],
                variant=row["variant"],
                n_seeds=row["n_seeds"],
                pre_joint=_format_rate(row.get("pretrained_task_success_rate_mean")),
                post_joint=_format_rate(row.get("finetuned_task_success_rate_mean")),
                delta_joint=_format_rate(row.get("delta_task_success_rate_mean")),
                pre_private=_format_rate(row.get("pretrained_private_exact_rate_mean")),
                post_private=_format_rate(row.get("finetuned_private_exact_rate_mean")),
                delta_private=_format_rate(row.get("delta_private_exact_rate_mean")),
            )
        )
    lines.extend(
        [
            "",
            "The pretrained and fine-tuned columns use the same validation prompt files and greedy decoding.",
            "The complete per-seed metrics are in `pre_post_metrics.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--modes", default="answer_only,public_cot")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--checkpoint", default="ckpt_final")
    args = parser.parse_args()
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    rows = collect_rows(
        args.root, modes=modes, variants=variants, checkpoint=args.checkpoint
    )
    aggregate = aggregate_rows(rows, modes=modes, variants=variants)
    _write_csv(args.root / "pre_post_metrics.csv", rows)
    _write_csv(args.root / "pre_post_metrics_mean_std.csv", aggregate)
    write_report(args.root / "REPORT.md", aggregate, checkpoint=args.checkpoint)
    (args.root / "summary.json").write_text(
        json.dumps(
            {"checkpoint": args.checkpoint, "per_seed": rows, "aggregate": aggregate},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Summary] wrote pretrained/fine-tuned report under {args.root}", flush=True)


if __name__ == "__main__":
    main()
