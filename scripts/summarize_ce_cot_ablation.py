#!/usr/bin/env python3
"""Summarize CE CoT-ablation runs and plot training curves.

Supports both the original single-seed layout:

    artifacts/ce_cot_ablation/<mode>/<variant>/...

and the multi-seed layout:

    artifacts/ce_cot_ablation/seed_<seed>/<mode>/<variant>/...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


MODES = ("answer_only", "public_cot", "verbose_public_cot")
VARIANTS = ("arith_piggyback", "arith_piggyback_control")
METRICS = ("task_success_rate", "public_exact_rate", "private_exact_rate")
FINAL_METRICS = (
    "task_success_rate",
    "public_exact_rate",
    "private_exact_rate",
    "concealment_rate",
    "format_rate",
    "avg_cot_words",
    "avg_reward",
)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_roots(root: Path) -> list[tuple[str, Path]]:
    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if seed_dirs:
        return [(path.name.removeprefix("seed_"), path) for path in seed_dirs]
    return [("0", root)]


def _collect_final_reports(seed_root: Path) -> dict[str, Any]:
    final: dict[str, Any] = {}
    for mode in MODES:
        final[mode] = {}
        for variant in VARIANTS:
            path = seed_root / mode / "per_variant_eval" / "ckpt_task" / f"{variant}.json"
            data = _load_json(path)
            reports = (data or {}).get("reports", [])
            final[mode][variant] = reports[0] if reports else None
    return final


def _collect_curve_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, seed_root in _seed_roots(root):
        for mode in MODES:
            for variant in VARIANTS:
                history_path = seed_root / mode / variant / "train_history.json"
                history = _load_json(history_path) or []
                for item in history:
                    validation = item.get("validation")
                    if not validation:
                        continue
                    row = {
                        "seed": seed,
                        "mode": mode,
                        "variant": variant,
                        "epoch": item.get("epoch"),
                        "step": item.get("step"),
                    }
                    for metric in METRICS:
                        row[metric] = validation.get(metric)
                    rows.append(row)
    return rows


def _write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _final_metric_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, seed_root in _seed_roots(root):
        final_reports = _collect_final_reports(seed_root)
        for mode in MODES:
            for variant in VARIANTS:
                report = final_reports.get(mode, {}).get(variant)
                if not report:
                    rows.append({"seed": seed, "mode": mode, "variant": variant, "status": "missing"})
                    continue
                trained = report.get("rl", {})
                row: dict[str, Any] = {
                    "seed": seed,
                    "mode": mode,
                    "variant": variant,
                    "status": "ok",
                }
                for metric in FINAL_METRICS:
                    row[metric] = trained.get(metric)
                rows.append(row)
    return rows


def _mean_std(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, 0
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std, len(clean)


def _aggregate_final_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for variant in VARIANTS:
            subset = [
                row for row in final_rows
                if row.get("status") == "ok" and row.get("mode") == mode and row.get("variant") == variant
            ]
            row: dict[str, Any] = {"mode": mode, "variant": variant, "status": "ok" if subset else "missing"}
            for metric in FINAL_METRICS:
                mean, std, n = _mean_std(row.get(metric) for row in subset)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_n"] = n
            rows.append(row)
    return rows


def _no_cot_comparison_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["seed"], row["mode"], row["variant"]): row
        for row in final_rows
        if row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    seeds = sorted({row["seed"] for row in final_rows})
    for seed in seeds:
        for mode in MODES:
            if mode == "answer_only":
                continue
            for variant in VARIANTS:
                current = lookup.get((seed, mode, variant))
                no_cot = lookup.get((seed, "answer_only", variant))
                if not current or not no_cot:
                    rows.append({"seed": seed, "mode": mode, "variant": variant, "status": "missing"})
                    continue
                row: dict[str, Any] = {
                    "seed": seed,
                    "mode": mode,
                    "variant": variant,
                    "baseline_mode": "answer_only",
                    "status": "ok",
                }
                for metric in FINAL_METRICS:
                    current_value = current.get(metric)
                    no_cot_value = no_cot.get(metric)
                    row[f"{metric}_mode"] = current_value
                    row[f"{metric}_answer_only"] = no_cot_value
                    if current_value is not None and no_cot_value is not None:
                        row[f"{metric}_gain_over_answer_only"] = current_value - no_cot_value
                rows.append(row)
    return rows


def _piggyback_control_delta_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["seed"], row["mode"], row["variant"]): row
        for row in final_rows
        if row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    seeds = sorted({row["seed"] for row in final_rows})
    for seed in seeds:
        for mode in MODES:
            piggyback = lookup.get((seed, mode, "arith_piggyback"))
            control = lookup.get((seed, mode, "arith_piggyback_control"))
            if not piggyback or not control:
                rows.append({"seed": seed, "mode": mode, "status": "missing"})
                continue
            row: dict[str, Any] = {"seed": seed, "mode": mode, "status": "ok"}
            for metric in FINAL_METRICS:
                piggyback_value = piggyback.get(metric)
                control_value = control.get(metric)
                row[f"{metric}_piggyback"] = piggyback_value
                row[f"{metric}_control"] = control_value
                if piggyback_value is not None and control_value is not None:
                    row[f"{metric}_piggyback_minus_control"] = piggyback_value - control_value
            rows.append(row)
    return rows


def _aggregate_delta_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        subset = [row for row in delta_rows if row.get("status") == "ok" and row.get("mode") == mode]
        row: dict[str, Any] = {"mode": mode, "status": "ok" if subset else "missing"}
        for metric in FINAL_METRICS:
            for suffix in ("piggyback", "control", "piggyback_minus_control"):
                key = f"{metric}_{suffix}"
                mean, std, n = _mean_std(item.get(key) for item in subset)
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
                row[f"{key}_n"] = n
        rows.append(row)
    return rows


def _aggregate_no_cot_rows(no_cot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        if mode == "answer_only":
            continue
        for variant in VARIANTS:
            subset = [
                row for row in no_cot_rows
                if row.get("status") == "ok" and row.get("mode") == mode and row.get("variant") == variant
            ]
            row: dict[str, Any] = {
                "mode": mode,
                "variant": variant,
                "baseline_mode": "answer_only",
                "status": "ok" if subset else "missing",
            }
            for metric in FINAL_METRICS:
                for suffix in ("mode", "answer_only", "gain_over_answer_only"):
                    key = f"{metric}_{suffix}"
                    mean, std, n = _mean_std(item.get(key) for item in subset)
                    row[f"{key}_mean"] = mean
                    row[f"{key}_std"] = std
                    row[f"{key}_n"] = n
            rows.append(row)
    return rows


def _aggregate_curve_rows(curve_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(row["mode"], row["variant"], int(row["step"])) for row in curve_rows})
    for mode, variant, step in keys:
        subset = [
            row for row in curve_rows
            if row["mode"] == mode and row["variant"] == variant and int(row["step"]) == step
        ]
        row: dict[str, Any] = {"mode": mode, "variant": variant, "step": step}
        epochs = [row.get("epoch") for row in subset if row.get("epoch") is not None]
        row["epoch"] = epochs[0] if epochs else None
        for metric in METRICS:
            mean, std, n = _mean_std(item.get(metric) for item in subset)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_n"] = n
        rows.append(row)
    return rows


def _write_plots(curve_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ablation] skip plots: matplotlib unavailable: {exc}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for axis, mode in zip(axes, MODES):
            for variant in VARIANTS:
                subset = [row for row in curve_rows if row["mode"] == mode and row["variant"] == variant]
                subset.sort(key=lambda row: (row.get("seed", ""), int(row["step"])))
                if not subset:
                    continue
                label = "piggyback" if variant == "arith_piggyback" else "control"
                seeds = sorted({row["seed"] for row in subset})
                for seed in seeds:
                    seed_rows = [row for row in subset if row["seed"] == seed]
                    seed_rows.sort(key=lambda row: int(row["step"]))
                    axis.plot(
                        [int(row["step"]) for row in seed_rows],
                        [float(row[metric]) for row in seed_rows],
                        alpha=0.25,
                        linewidth=1,
                    )
                agg_subset = [row for row in aggregate_rows if row["mode"] == mode and row["variant"] == variant]
                agg_subset.sort(key=lambda row: int(row["step"]))
                if agg_subset:
                    steps = [int(row["step"]) for row in agg_subset]
                    means = [float(row[f"{metric}_mean"]) for row in agg_subset]
                    stds = [float(row[f"{metric}_std"] or 0.0) for row in agg_subset]
                    axis.plot(steps, means, marker="o", linewidth=2, label=f"{label} mean")
                    if any(std > 0 for std in stds):
                        lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
                        upper = [min(1.0, mean + std) for mean, std in zip(means, stds)]
                        axis.fill_between(steps, lower, upper, alpha=0.15)
            axis.set_title(mode)
            axis.set_xlabel("training step")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(metric)
        axes[-1].legend(loc="lower right")
        fig.tight_layout()
        out_path = output_dir / f"{metric}_by_checkpoint.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[ablation] wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CE CoT-ablation runs.")
    parser.add_argument("--root", type=Path, default=Path("artifacts/ce_cot_ablation"))
    args = parser.parse_args()

    seed_info = [{"seed": seed, "root": str(seed_root)} for seed, seed_root in _seed_roots(args.root)]
    final_rows = _final_metric_rows(args.root)
    final_agg_rows = _aggregate_final_rows(final_rows)
    no_cot_rows = _no_cot_comparison_rows(final_rows)
    no_cot_agg_rows = _aggregate_no_cot_rows(no_cot_rows)
    piggyback_delta_rows = _piggyback_control_delta_rows(final_rows)
    piggyback_delta_agg_rows = _aggregate_delta_rows(piggyback_delta_rows)
    curve_rows = _collect_curve_rows(args.root)
    curve_agg_rows = _aggregate_curve_rows(curve_rows)

    summary = {
        "modes": list(MODES),
        "variants": list(VARIANTS),
        "seeds": seed_info,
        "answer_only_is_no_cot_baseline": True,
        "final_metric_rows": final_rows,
        "final_metric_mean_std_rows": final_agg_rows,
        "no_cot_comparisons": no_cot_rows,
        "no_cot_comparison_mean_std_rows": no_cot_agg_rows,
        "piggyback_control_deltas": piggyback_delta_rows,
        "piggyback_control_delta_mean_std_rows": piggyback_delta_agg_rows,
        "num_curve_points": len(curve_rows),
        "num_curve_mean_std_points": len(curve_agg_rows),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_rows_csv(curve_rows, args.root / "training_curves.csv")
    _write_rows_csv(curve_agg_rows, args.root / "training_curves_mean_std.csv")
    _write_rows_csv(final_rows, args.root / "final_metrics.csv")
    _write_rows_csv(final_agg_rows, args.root / "final_metrics_mean_std.csv")
    _write_rows_csv(no_cot_rows, args.root / "no_cot_comparison.csv")
    _write_rows_csv(no_cot_agg_rows, args.root / "no_cot_comparison_mean_std.csv")
    _write_rows_csv(piggyback_delta_rows, args.root / "piggyback_control_deltas.csv")
    _write_rows_csv(piggyback_delta_agg_rows, args.root / "piggyback_control_deltas_mean_std.csv")
    _write_plots(curve_rows, curve_agg_rows, args.root / "plots")

    for path_name in (
        "summary.json",
        "training_curves.csv",
        "training_curves_mean_std.csv",
        "final_metrics.csv",
        "final_metrics_mean_std.csv",
        "no_cot_comparison.csv",
        "no_cot_comparison_mean_std.csv",
        "piggyback_control_deltas.csv",
        "piggyback_control_deltas_mean_std.csv",
    ):
        print(f"[ablation] wrote {args.root / path_name}", flush=True)


if __name__ == "__main__":
    main()
