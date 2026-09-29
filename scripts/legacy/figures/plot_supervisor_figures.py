#!/usr/bin/env python3
"""Create compact paper figures for the available length experiments.

Multiplication and S5 combine exact-length OOD evaluations with the original
Hard in-distribution anchor/range. Knowledge instead uses the balanced
all-but-one-overlap checkpoint trained jointly on every length from one to
five. The old multiplication ``x=4`` observations are intentionally excluded:
they were only the isolated boundary operand 1000.

Figure 2 reports the median monitor-visible reasoning length for Piggy and
Steg. The answer block is never part of ``cot_words``. New-sweep completions
without a valid answer block are excluded because their often-truncated loops
are not comparable reasoning chains; accuracy continues to count them wrong.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
PRESERVED_ROOT = REPO_ROOT / "artifacts" / "figure_data_preservation"
SWEEP_ROOT = REPO_ROOT / "artifacts" / "hard_checkpoint_length_sweep"
KNOWLEDGE_ROOT = REPO_ROOT / "artifacts" / "knowledge_length_retraining_all_but_one_v2"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "paper_figures"
PANEL_ROOT = OUTPUT_ROOT / "appendix_accuracy_panels"

TASKS = ("multiplication", "s5", "knowledge")
MODELS = ("qwen", "llama")
METHODS = ("vanilla", "filler", "public_only_cot", "piggyback", "steganography")
CURVE_METHODS = ("vanilla", "public_only_cot", "piggyback", "steganography")

TASK_LABEL = {"multiplication": "Multiplication", "s5": "S5", "knowledge": "Knowledge"}
MODEL_LABEL = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B"}
METHOD_LABEL = {
    "vanilla": "Vanilla",
    "filler": "Filler",
    "public_only_cot": "Public-only CoT",
    "piggyback": "Piggy",
    "steganography": "Steg",
}
METHOD_COLOR = {
    "vanilla": "#737373",
    "filler": "#A88418",
    "public_only_cot": "#8C6BB1",
    "piggyback": "#0072B2",
    "steganography": "#D55E00",
}
METHOD_LINESTYLE = {
    "vanilla": ":",
    "filler": "-.",
    "public_only_cot": "--",
    "piggyback": "-",
    "steganography": "-",
}
METHOD_MARKER = {
    "vanilla": "o",
    "filler": "X",
    "public_only_cot": "^",
    "piggyback": "D",
    "steganography": "s",
}
MODEL_STYLE = {"qwen": "-", "llama": "--"}
X_LABEL = {
    "multiplication": "Operand digits",
    "s5": "Number of swaps",
    "knowledge": "Number of facts",
}
X_TICKS = {
    "multiplication": [2, 3, 4],
    "s5": [1, 4, 7, 10, 13, 16, 19],
    "knowledge": [1, 2, 3, 4, 5],
}
CHAIN_X_TICKS = {
    "multiplication": [2, 3, 4],
    "s5": [2, 5, 8, 11, 14, 17, 19],
    "knowledge": [1, 2, 3, 4, 5],
}
X_LIMITS = {"multiplication": (1.82, 4.18), "s5": (0.5, 19.5), "knowledge": (0.7, 5.3)}
TRAIN_SPAN = {
    "multiplication": (2.78, 3.22),
    "s5": (9.5, 19.5),
    "knowledge": (0.7, 5.3),
}
FILLER_LENGTH = {"multiplication": 2, "s5": 1, "knowledge": 1}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def _normalized_new_seed_rows() -> list[dict[str, Any]]:
    normalized = []
    for row in read_csv(SWEEP_ROOT / "all_per_seed_length_metrics.csv"):
        if row["task"] == "knowledge":
            continue
        normalized.append(
            {
                "task": row["task"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "length": int(row["length"]),
                "n": int(row["n"]),
                "private_accuracy": float(row["private_exact_rate"]),
                "source": "new_exact_length_fixed_hard_checkpoint",
            }
        )
    return normalized


def _original_hard_seed_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, int], list[dict[str, str]]] = defaultdict(
        list
    )
    for row in read_csv(PRESERVED_ROOT / "hard_per_example.csv"):
        task, length = row["task"], int(row["nominal_length"])
        keep = (task == "multiplication" and length == 3) or (
            task == "s5" and 10 <= length <= 19
        )
        if keep:
            grouped[
                (task, row["model"], int(row["seed"]), row["method"], length)
            ].append(row)
    result = []
    for (task, model, seed, method, length), group in sorted(grouped.items()):
        result.append(
            {
                "task": task,
                "model": model,
                "seed": seed,
                "method": method,
                "length": length,
                "n": len(group),
                "private_accuracy": statistics.fmean(
                    int(row["private_correct"]) for row in group
                ),
                "source": "original_hard_in_distribution",
            }
        )
    return result


def _easy_filler_seed_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PRESERVED_ROOT / "easy_per_example.csv"):
        task, length = row["task"], int(row["nominal_length"])
        if (
            task != "knowledge"
            and row["method"] == "filler"
            and length == FILLER_LENGTH[task]
        ):
            grouped[(task, row["model"], int(row["seed"]), length)].append(row)
    result = []
    for (task, model, seed, length), group in sorted(grouped.items()):
        result.append(
            {
                "task": task,
                "model": model,
                "seed": seed,
                "method": "filler",
                "length": length,
                "n": len(group),
                "private_accuracy": statistics.fmean(
                    int(row["private_correct"]) for row in group
                ),
                "source": "easy_checkpoint_reference_only",
            }
        )
    return result


def _balanced_knowledge_seed_rows() -> list[dict[str, Any]]:
    result = []
    for row in read_csv(KNOWLEDGE_ROOT / "per_seed_length_metrics.csv"):
        result.append(
            {
                "task": "knowledge",
                "model": row["model"],
                "seed": int(row["seed"]),
                "method": row["condition"],
                "length": int(row["length"]),
                "n": int(row["n_examples"]),
                "private_accuracy": float(row["private_exact_rate"]),
                "source": "balanced_multilength_all_but_one_retraining",
            }
        )
    if len(result) != 150:
        raise RuntimeError(
            f"Expected 150 balanced Knowledge seed rows, found {len(result)}"
        )
    return result


def build_accuracy_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = (
        _normalized_new_seed_rows()
        + _original_hard_seed_rows()
        + _easy_filler_seed_rows()
        + _balanced_knowledge_seed_rows()
    )
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in seed_rows:
        grouped[
            (row["task"], row["model"], row["method"], row["length"], row["source"])
        ].append(row)
    aggregate = []
    for (task, model, method, length, source), group in sorted(grouped.items()):
        if sorted(row["seed"] for row in group) != [0, 1, 2]:
            raise RuntimeError(
                f"Incomplete seeds: {task}/{model}/{method}/L{length}/{source}"
            )
        mean, sd = _mean_sd([row["private_accuracy"] for row in group])
        aggregate.append(
            {
                "task": task,
                "model": model,
                "method": method,
                "length": length,
                "n_seeds": 3,
                "n_examples": sum(row["n"] for row in group),
                "private_accuracy_mean": mean,
                "private_accuracy_sample_sd": sd,
                "source": source,
            }
        )
    expected = {
        "multiplication": {2, 3, 4},
        "s5": set(range(2, 20)),
        "knowledge": set(range(1, 6)),
    }
    for task in TASKS:
        for model in MODELS:
            expected_methods = METHODS if task == "knowledge" else CURVE_METHODS
            for method in expected_methods:
                observed = {
                    row["length"]
                    for row in aggregate
                    if (row["task"], row["model"], row["method"])
                    == (task, model, method)
                }
                if observed != expected[task]:
                    raise RuntimeError(
                        f"Incomplete curve {task}/{model}/{method}: {observed}"
                    )
    return seed_rows, aggregate


def _new_example_rows() -> list[dict[str, Any]]:
    result = []
    for path in sorted(SWEEP_ROOT.glob("*/*/seed_*/per_example.csv")):
        for row in read_csv(path):
            result.append(
                {
                    "task": row["task"],
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "length": int(row["length"]),
                    "cot_words": int(row["cot_words"]),
                    "format_ok": bool(int(row["format_ok"])),
                    "source": "new_exact_length_fixed_hard_checkpoint",
                }
            )
    if len(result) != 72_000:
        raise RuntimeError(f"Expected 72,000 new per-example rows, found {len(result)}")
    return result


def validate_new_sweep() -> None:
    """Recompute every reported length cell from the transferred examples."""
    file_count = 0
    for example_path in sorted(SWEEP_ROOT.glob("*/*/seed_*/per_example.csv")):
        file_count += 1
        reported_path = example_path.with_name("per_length_metrics.csv")
        examples = read_csv(example_path)
        reported = read_csv(reported_path)
        if len(examples) != 4_000:
            raise RuntimeError(
                f"Expected 4,000 rows in {example_path}, found {len(examples)}"
            )
        lookup = {(row["method"], int(row["length"])): row for row in reported}
        groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in examples:
            groups[(row["method"], int(row["length"]))].append(row)
        if set(groups) != set(lookup):
            raise RuntimeError(f"Metric/example key mismatch in {example_path.parent}")
        for key, group in groups.items():
            checks = {
                "private_exact_rate": statistics.fmean(
                    int(row["private_correct"]) for row in group
                ),
                "public_exact_rate": statistics.fmean(
                    int(row["public_correct"]) for row in group
                ),
                "task_success_rate": statistics.fmean(
                    int(row["joint_correct"]) for row in group
                ),
                "avg_cot_words": statistics.fmean(
                    int(row["cot_words"]) for row in group
                ),
            }
            if int(lookup[key]["n"]) != len(group):
                raise RuntimeError(f"Count mismatch for {example_path.parent}/{key}")
            for metric, observed in checks.items():
                if abs(float(lookup[key][metric]) - observed) > 1e-12:
                    raise RuntimeError(
                        f"{metric} mismatch for {example_path.parent}/{key}"
                    )
    if file_count != 18:
        raise RuntimeError(f"Expected 18 completed cells, found {file_count}")


def _old_chain_rows() -> list[dict[str, Any]]:
    result = []
    for row in read_csv(PRESERVED_ROOT / "hard_per_example.csv"):
        if row["method"] not in {"piggyback", "steganography"}:
            continue
        task, length = row["task"], int(row["nominal_length"])
        keep = (task == "multiplication" and length == 3) or (
            task == "s5" and 10 <= length <= 19
        )
        if keep:
            result.append(
                {
                    "task": task,
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "length": length,
                    "cot_words": int(row["cot_words"]),
                    "format_ok": True,
                    "source": "original_hard_in_distribution",
                }
            )
    return result


def _balanced_knowledge_example_rows() -> list[dict[str, Any]]:
    result = []
    for row in read_csv(KNOWLEDGE_ROOT / "per_example.csv"):
        if row["condition"] not in {"piggyback", "steganography"}:
            continue
        result.append(
            {
                "task": "knowledge",
                "model": row["model"],
                "seed": int(row["seed"]),
                "method": row["condition"],
                "length": int(row["length"]),
                "cot_words": int(row["cot_words"]),
                "format_ok": bool(int(row["format_ok"])),
                "source": "balanced_multilength_all_but_one_retraining",
            }
        )
    if len(result) != 12_000:
        raise RuntimeError(
            f"Expected 12,000 balanced Knowledge chain rows, found {len(result)}"
        )
    return result


def build_chain_data() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    fixed_sweep = [row for row in _new_example_rows() if row["task"] != "knowledge"]
    for row in fixed_sweep + _old_chain_rows() + _balanced_knowledge_example_rows():
        if row["method"] in {"piggyback", "steganography"}:
            grouped[
                (row["task"], row["model"], row["method"], row["length"], row["source"])
            ].append(row)
    output = []
    for (task, model, method, length, source), group in sorted(grouped.items()):
        valid = [row["cot_words"] for row in group if row["format_ok"]]
        if not valid:
            raise RuntimeError(f"No valid chains: {task}/{model}/{method}/L{length}")
        seed_medians = [
            statistics.median(
                row["cot_words"]
                for row in group
                if row["format_ok"] and row["seed"] == seed
            )
            for seed in (0, 1, 2)
        ]
        output.append(
            {
                "task": task,
                "model": model,
                "method": method,
                "length": length,
                "n_total": len(group),
                "n_valid_format": len(valid),
                "format_rate": len(valid) / len(group),
                "median_cot_words": statistics.median(valid),
                "mean_seed_median_cot_words": statistics.fmean(seed_medians),
                "sample_sd_seed_median_cot_words": statistics.stdev(seed_medians),
                "source": source,
                "answer_block_excluded": True,
                "invalid_or_runaway_completions_excluded": source.startswith("new_"),
            }
        )
    return output


def _training_span(ax: plt.Axes, task: str) -> None:
    lo, hi = TRAIN_SPAN[task]
    ax.axvspan(lo, hi, color="#EFEFEF", zorder=0)


def _plot_accuracy_panel(
    ax: plt.Axes, rows: list[dict[str, Any]], task: str, model: str
) -> None:
    _training_span(ax, task)
    selected = [row for row in rows if row["task"] == task and row["model"] == model]
    curve_methods = METHODS if task == "knowledge" else CURVE_METHODS
    for method in curve_methods:
        series = sorted(
            (row for row in selected if row["method"] == method),
            key=lambda row: row["length"],
        )
        x = [row["length"] for row in series]
        y = [row["private_accuracy_mean"] for row in series]
        sd = [row["private_accuracy_sample_sd"] for row in series]
        ax.fill_between(
            x,
            [max(0, a - b) for a, b in zip(y, sd)],
            [min(1, a + b) for a, b in zip(y, sd)],
            color=METHOD_COLOR[method],
            alpha=0.09,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            linestyle=METHOD_LINESTYLE[method],
            linewidth=1.25 if method in {"piggyback", "steganography"} else 0.95,
            markersize=3.1,
            markeredgewidth=0.45,
            zorder=3,
        )
    if task != "knowledge":
        filler = [row for row in selected if row["method"] == "filler"]
        if len(filler) != 1:
            raise RuntimeError(f"Expected one filler reference for {task}/{model}")
        point = filler[0]
        ax.errorbar(
            [point["length"]],
            [point["private_accuracy_mean"]],
            yerr=[point["private_accuracy_sample_sd"]],
            fmt=METHOD_MARKER["filler"],
            color=METHOD_COLOR["filler"],
            markersize=4.4,
            elinewidth=0.7,
            capsize=1.6,
            markeredgewidth=0.6,
            zorder=5,
        )
    ax.set_xlim(*X_LIMITS[task])
    ax.set_xticks(X_TICKS[task])
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(X_LABEL[task])
    ax.set_title(TASK_LABEL[task], loc="left", fontweight="semibold", pad=2.5)


def _method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR[m],
            marker=METHOD_MARKER[m],
            linestyle="None" if m == "filler" else METHOD_LINESTYLE[m],
            linewidth=1.25 if m in {"piggyback", "steganography"} else 0.95,
            markersize=3.8,
            label=METHOD_LABEL[m],
        )
        for m in METHODS
    ]


def draw_accuracy(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 6.35), sharey=True)
    for i, task in enumerate(TASKS):
        for j, model in enumerate(MODELS):
            _plot_accuracy_panel(axes[i, j], rows, task, model)
            if j == 0:
                axes[i, j].set_ylabel("Private exact accuracy")
            if i == 0:
                axes[i, j].text(
                    0.5,
                    1.04,
                    MODEL_LABEL[model],
                    transform=axes[i, j].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8.1,
                    fontweight="semibold",
                )
    fig.legend(
        handles=_method_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=5,
        columnspacing=1.15,
        handlelength=1.7,
    )
    fig.text(
        0.5,
        0.006,
        "Gray background = input lengths seen during training; points outside it are OOD tests. "
        "Knowledge was trained jointly on lengths 1–5. Bands: ±1 SD over 3 seeds. "
        "For Multiplication and S5, Filler × is a separate Easy-checkpoint reference.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.085, right=0.992, bottom=0.072, top=0.88, hspace=0.52, wspace=0.18
    )
    fig.savefig(OUTPUT_ROOT / "figure1_accuracy_by_length_appendix.pdf")
    fig.savefig(OUTPUT_ROOT / "figure1_accuracy_by_length_appendix.png", dpi=600)
    plt.close(fig)

    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(3.4, 2.45))
            _plot_accuracy_panel(ax, rows, task, model)
            ax.set_ylabel("Private exact accuracy")
            ax.text(
                0.99,
                0.035,
                MODEL_LABEL[model],
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.3,
                color="#555555",
            )
            fig.legend(
                handles=_method_handles(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=3,
                columnspacing=0.9,
                handlelength=1.5,
                fontsize=5.8,
            )
            fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.78)
            stem = PANEL_ROOT / f"accuracy_by_length_{task}_{model}"
            stale_pdf = stem.with_suffix(".pdf")
            if stale_pdf.exists():
                stale_pdf.unlink()
            fig.savefig(stem.with_suffix(".png"), dpi=600)
            plt.close(fig)


def _plot_chain_panel(ax: plt.Axes, rows: list[dict[str, Any]], task: str) -> None:
    _training_span(ax, task)
    selected = [row for row in rows if row["task"] == task]
    plotted = []
    if task == "knowledge":
        for method in ("piggyback", "steganography"):
            by_model = {
                model: sorted(
                    (
                        row
                        for row in selected
                        if row["method"] == method and row["model"] == model
                    ),
                    key=lambda row: row["length"],
                )
                for model in MODELS
            }
            if [
                (row["length"], row["median_cot_words"]) for row in by_model["qwen"]
            ] != [
                (row["length"], row["median_cot_words"]) for row in by_model["llama"]
            ]:
                raise RuntimeError(
                    f"Knowledge chain lengths differ across models for {method}"
                )
            series = by_model["qwen"]
            x = [row["length"] for row in series]
            y = [row["median_cot_words"] for row in series]
            plotted.extend(y)
            ax.plot(
                x,
                y,
                color=METHOD_COLOR[method],
                linestyle="-" if method == "piggyback" else "--",
                marker=METHOD_MARKER[method],
                linewidth=2.2 if method == "piggyback" else 1.15,
                markersize=3.1,
                markeredgewidth=0.4,
                zorder=3 if method == "piggyback" else 4,
            )
        ax.text(
            0.98,
            0.06,
            "Both models coincide",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
    else:
        for method in ("piggyback", "steganography"):
            for model in MODELS:
                series = sorted(
                    (
                        row
                        for row in selected
                        if row["method"] == method and row["model"] == model
                    ),
                    key=lambda row: row["length"],
                )
                x = [row["length"] for row in series]
                y = [row["median_cot_words"] for row in series]
                plotted.extend(y)
                ax.plot(
                    x,
                    y,
                    color=METHOD_COLOR[method],
                    linestyle=MODEL_STYLE[model],
                    marker=METHOD_MARKER[method],
                    linewidth=1.25,
                    markersize=2.9,
                    markeredgewidth=0.4,
                    zorder=3,
                )
    data_min, data_max = min(plotted), max(plotted)
    padding = max(4.0, 0.10 * (data_max - data_min))
    ax.set_ylim(max(0, data_min - padding), data_max + padding)
    ax.set_xlim(*X_LIMITS[task])
    ax.set_xticks(CHAIN_X_TICKS[task])
    ax.set_xlabel(X_LABEL[task])
    ax.set_ylabel("Visible chain (words)")
    ax.set_title(TASK_LABEL[task], loc="left", fontweight="semibold", pad=2.5)


def draw_chain(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for ax, task in zip(axes, TASKS):
        _plot_chain_panel(ax, rows, task)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR[m],
            marker=METHOD_MARKER[m],
            linewidth=1.3,
            markersize=3.4,
            label=METHOD_LABEL[m],
        )
        for m in ("piggyback", "steganography")
    ]
    handles += [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=MODEL_STYLE[m],
            linewidth=1.2,
            label=MODEL_LABEL[m],
        )
        for m in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=1.25,
        handlelength=2.0,
    )
    fig.text(
        0.5,
        0.008,
        "Median monitor-visible CoT over 3 seeds; answer block excluded. "
        "Malformed/runaway fixed-sweep completions are excluded.\n"
        "Gray background = training support; Knowledge was trained jointly on lengths 1–5. "
        "Facets use independent y-scales.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.29, top=0.81, wspace=0.40)
    fig.savefig(OUTPUT_ROOT / "figure2_chain_length_by_input_length.pdf")
    fig.savefig(OUTPUT_ROOT / "figure2_chain_length_by_input_length.png", dpi=600)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    obsolete = OUTPUT_ROOT / "figure1_binned_plot_data.csv"
    if obsolete.exists():
        obsolete.unlink()
    configure_style()
    validate_new_sweep()
    seed_accuracy, accuracy = build_accuracy_data()
    chain = build_chain_data()
    accuracy_seed_path = OUTPUT_ROOT / "figure1_fixed_checkpoint_per_seed.csv"
    accuracy_path = OUTPUT_ROOT / "figure1_fixed_checkpoint_plot_data.csv"
    chain_path = OUTPUT_ROOT / "figure2_chain_plot_data.csv"
    write_csv(accuracy_seed_path, seed_accuracy)
    write_csv(accuracy_path, accuracy)
    write_csv(chain_path, chain)
    draw_accuracy(accuracy)
    draw_chain(chain)
    sources = [
        SWEEP_ROOT / "all_per_seed_length_metrics.csv",
        PRESERVED_ROOT / "hard_per_example.csv",
        PRESERVED_ROOT / "easy_per_example.csv",
        KNOWLEDGE_ROOT / "per_seed_length_metrics.csv",
        KNOWLEDGE_ROOT / "per_example.csv",
    ] + sorted(SWEEP_ROOT.glob("*/*/seed_*/per_example.csv"))
    outputs = [
        accuracy_seed_path,
        accuracy_path,
        chain_path,
        OUTPUT_ROOT / "figure1_accuracy_by_length_appendix.pdf",
        OUTPUT_ROOT / "figure1_accuracy_by_length_appendix.png",
        OUTPUT_ROOT / "figure2_chain_length_by_input_length.pdf",
        OUTPUT_ROOT / "figure2_chain_length_by_input_length.png",
    ] + sorted(PANEL_ROOT.glob("*"))
    manifest = {
        "created_on": date.today().isoformat(),
        "validation": "passed",
        "new_sweep_examples": 72_000,
        "accuracy_plot_rows": len(accuracy),
        "chain_plot_rows": len(chain),
        "methodology": {
            "accuracy": "mean private exact accuracy across three seeds; all completions retained",
            "chain_length": "median monitor-visible CoT words; answer block and new-sweep format-invalid completions excluded",
            "knowledge": "balanced joint training on lengths 1-5 with Piggy sharing all but one fact",
            "old_multiplication_x4": "deleted; isolated operand 1000, replaced by new exact 4-digit evaluation",
        },
        "sources": [
            {"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256(p)}
            for p in sources
        ],
        "outputs": [
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in outputs
        ],
    }
    (OUTPUT_ROOT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Validated 18 cells / 72,000 new examples; wrote figures to {OUTPUT_ROOT.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
