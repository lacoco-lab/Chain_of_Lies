#!/usr/bin/env python3
"""Plot the final standard-Transformer Goldreich-PARITY results.

The figure reads only the authoritative ordinary-softmax, autoregressive test
artifacts. It intentionally excludes results from earlier architectures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESULT_ROOT = REPO_ROOT / "artifacts" / "parity_goldreich_standard_transformer_v2"
CONTROL_ROOT = REPO_ROOT / "artifacts" / "parity_standard_cot_controls"
STAGED_CONTROL_ROOT = REPO_ROOT / "artifacts" / "parity_standard_cot_controls_staged"
MATCHED_END_TO_END_ROOT = (
    REPO_ROOT / "artifacts" / "parity_standard_cot_controls_matched_end_to_end"
)
ABLATION_ROOT = REPO_ROOT / "artifacts" / "parity_goldreich_standard_ablations"
OUTPUT_ROOT = REPO_ROOT / "figures" / "encrypted_parity"
LENGTHS = (4, 8, 16, 32, 64, 256, 512, 1024)
MODEL_DIMS = (128,)

COLORS = {128: "#D55E00"}
MARKERS = {128: "o"}
LINESTYLES = {128: "-"}
GRID = "#E1E1E1"
SECONDARY = "#5F6368"
CONTROL_COLORS = {
    "encrypted_cot": "#D55E00",
    "normal_cot": "#0072B2",
    "no_cot": "#666A70",
    "filler_cot": "#A7AAAD",
    "no_cot_end_to_end": "#666A70",
    "filler_cot_end_to_end": "#A7AAAD",
}
CONTROL_LABELS = {
    "encrypted_cot": "Encrypted CoT",
    "normal_cot": "Normal CoT",
    "no_cot": "No CoT (staged)",
    "filler_cot": "Filler-only CoT (staged)",
    "no_cot_end_to_end": "No CoT (end-to-end)",
    "filler_cot_end_to_end": "Filler-only CoT (end-to-end)",
}
CONTROL_MARKERS = {
    "encrypted_cot": "o",
    "normal_cot": "s",
    "no_cot": "^",
    "filler_cot": "D",
    "no_cot_end_to_end": "^",
    "filler_cot_end_to_end": "D",
}
ABLATION_COLORS = {
    "Full": "#D55E00",
    "No route labels": "#0072B2",
    "End-to-end only": "#8A8D91",
}

# Small multiplicative x offsets separate coincident markers on the log-scaled
# axes. They are display-only: ticks, CSVs, and all reported N values remain exact.
MAIN_X_DODGE = {
    "no_cot": 0.90,
    "filler_cot": 1.04,
    "no_cot_end_to_end": 0.96,
    "filler_cot_end_to_end": 1.10,
    "encrypted_cot": 1.0,
}
CONTROL_X_DODGE = {
    "no_cot": 0.88,
    "filler_cot": 0.96,
    "no_cot_end_to_end": 1.04,
    "filler_cot_end_to_end": 1.12,
    "normal_cot": 1.03,
    "encrypted_cot": 1.10,
}
ABLATION_X_DODGE = {"Full": 0.94, "No route labels": 1.0, "End-to-end only": 1.06}


def display_x(rows: list[tuple[int, float]], factor: float) -> list[float]:
    return [row[0] * factor for row in rows]


def read_results() -> dict[int, list[dict[str, float | int]]]:
    results: dict[int, list[dict[str, float | int]]] = {}
    for model_dim in MODEL_DIMS:
        rows = []
        for length in LENGTHS:
            path = RESULT_ROOT / f"d{model_dim}" / f"n{length}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing required final result: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if int(result["length"]) != length or int(result["model_dim"]) != model_dim:
                raise RuntimeError(f"Metadata mismatch in {path}")
            if int(result["test_examples"]) != 8192:
                raise RuntimeError(f"Expected 8,192 test examples in {path}")
            if "unseen during training" not in result["seed_split"]:
                raise RuntimeError(f"Test-seed provenance is missing in {path}")
            test = result["test_softmax_attention"]
            rows.append(
                {
                    "d_model": model_dim,
                    "N": length,
                    "seed_bits": int(result["seed_bits"]),
                    "parameters": int(result["architecture"]["trainable_parameters"]),
                    "state_bit_accuracy": float(test["state_bit_accuracy"]),
                    "trace_exact": float(test["trace_exact"]),
                    "final_parity_accuracy": float(test["final_parity_accuracy"]),
                    "joint_exact": float(test["joint_exact"]),
                }
            )
        results[model_dim] = rows
    return results


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_controls(
    results: dict[int, list[dict[str, float | int]]],
) -> dict[str, list[tuple[int, float]]]:
    lengths = {
        "no_cot": (4, 8, 16, 32, 64),
        "filler_cot": (4, 8, 16, 32, 64),
        "normal_cot": (4, 8, 16, 32, 64, 256, 512, 1024),
    }
    controls: dict[str, list[tuple[int, float]]] = {}
    for condition, condition_lengths in lengths.items():
        rows = []
        for length in condition_lengths:
            root = (
                STAGED_CONTROL_ROOT
                if condition in {"no_cot", "filler_cot"} and length > 16
                else CONTROL_ROOT
            )
            path = root / condition / f"n{length}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing matched control: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["condition"] != condition or int(result["length"]) != length:
                raise RuntimeError(f"Control metadata mismatch in {path}")
            if condition in {"no_cot", "filler_cot"}:
                for stage in (
                    "routing_training",
                    "representation_training",
                    "public_training",
                ):
                    if result[stage].get("enabled") is not True:
                        raise RuntimeError(
                            f"Paper-facing control is not staged: {path}"
                        )
                if length > 16 and result.get("training_regime") != "staged":
                    raise RuntimeError(
                        f"Large-N control lacks staged provenance: {path}"
                    )
            rows.append(
                (
                    length,
                    float(result["test_softmax_attention"]["final_parity_accuracy"]),
                )
            )
        controls[condition] = rows
    controls["encrypted_cot"] = [
        (int(row["N"]), float(row["final_parity_accuracy"]))
        for row in results[128]
        if int(row["N"]) <= 1024
    ]
    for condition in ("no_cot", "filler_cot"):
        path = MATCHED_END_TO_END_ROOT / condition / "n16" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing matched end-to-end control: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["condition"] != condition or int(result["length"]) != 16:
            raise RuntimeError(f"End-to-end control metadata mismatch in {path}")
        for stage in ("routing_training", "representation_training", "public_training"):
            if result[stage].get("enabled") is not False:
                raise RuntimeError(
                    f"End-to-end control unexpectedly enables {stage}: {path}"
                )
        controls[f"{condition}_end_to_end"] = [
            (16, float(result["test_softmax_attention"]["final_parity_accuracy"]))
        ]
    return controls


def read_ablations(
    results: dict[int, list[dict[str, float | int]]],
) -> dict[str, list[dict[str, float | int]]]:
    baseline = {int(row["N"]): row for row in results[128]}
    output: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in ABLATION_COLORS
    }
    for length in (32, 64):
        output["Full"].append(
            {
                "N": length,
                "state_bit_accuracy": float(baseline[length]["state_bit_accuracy"]),
                "joint_exact": float(baseline[length]["joint_exact"]),
            }
        )
        for mode, label in (
            ("no_route", "No route labels"),
            ("end_to_end", "End-to-end only"),
        ):
            path = ABLATION_ROOT / mode / f"n{length}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing final-standard ablation: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["mode"] != mode or int(result["length"]) != length:
                raise RuntimeError(f"Ablation metadata mismatch in {path}")
            test = result["test_softmax_attention"]
            output[label].append(
                {
                    "N": length,
                    "state_bit_accuracy": float(test["state_bit_accuracy"]),
                    "joint_exact": float(test["joint_exact"]),
                }
            )
    return output


def format_x_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS)
    ax.set_xticklabels([f"{length:,}" for length in LENGTHS])
    ax.set_xlabel("Input length $N$")


def plot_series(
    ax: plt.Axes, results: dict[int, list[dict[str, float | int]]], metric: str
) -> None:
    for model_dim in MODEL_DIMS:
        rows = results[model_dim]
        ax.plot(
            [int(row["N"]) for row in rows],
            [float(row[metric]) for row in rows],
            color=COLORS[model_dim],
            marker=MARKERS[model_dim],
            linestyle=LINESTYLES[model_dim],
            linewidth=1.9,
            markersize=5.2,
            markeredgewidth=0.8,
            markeredgecolor="white",
            label="Encrypted CoT",
            zorder=3,
        )
    format_x_axis(ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Test accuracy")


def draw_exact_cot(
    ax: plt.Axes, results: dict[int, list[dict[str, float | int]]]
) -> None:
    plot_series(ax, results, "state_bit_accuracy")
    ax.set_ylim(0.9975, 1.0002)
    ax.set_yticks([0.998, 0.999, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_title("(a) Encrypted state bits", loc="left", fontweight="semibold")


def draw_final_answer(
    ax: plt.Axes, results: dict[int, list[dict[str, float | int]]]
) -> None:
    plot_series(ax, results, "joint_exact")
    ax.set_ylim(0.9975, 1.0002)
    ax.set_yticks([0.998, 0.999, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_title("(b) Complete CoT + answer", loc="left", fontweight="semibold")


def draw_main_performance(
    ax: plt.Axes,
    controls: dict[str, list[tuple[int, float]]],
) -> None:
    """Draw the compact, answer-level comparison used in the main text."""
    for condition in (
        "no_cot",
        "filler_cot",
        "no_cot_end_to_end",
        "filler_cot_end_to_end",
    ):
        rows = controls[condition]
        end_to_end = condition.endswith("_end_to_end")
        ax.plot(
            display_x(rows, MAIN_X_DODGE[condition]),
            [row[1] for row in rows],
            color=CONTROL_COLORS[condition],
            marker=CONTROL_MARKERS[condition],
            linestyle="None" if end_to_end else "--",
            linewidth=0 if end_to_end else 1.4,
            markersize=4.5,
            markeredgewidth=0.7,
            markeredgecolor=CONTROL_COLORS[condition] if end_to_end else "white",
            markerfacecolor="none" if end_to_end else CONTROL_COLORS[condition],
            label=CONTROL_LABELS[condition],
            clip_on=False,
            zorder=2,
        )
    encrypted = controls["encrypted_cot"]
    ax.plot(
        display_x(encrypted, MAIN_X_DODGE["encrypted_cot"]),
        [row[1] for row in encrypted],
        color=CONTROL_COLORS["encrypted_cot"],
        marker=CONTROL_MARKERS["encrypted_cot"],
        linestyle="-",
        linewidth=1.9,
        markersize=4.8,
        markeredgewidth=0.7,
        markeredgecolor="white",
        label=CONTROL_LABELS["encrypted_cot"],
        clip_on=False,
        zorder=3,
    )
    main_lengths = (4, 8, 16, 32, 64, 256, 512, 1024)
    ax.set_xscale("log", base=2)
    ax.set_xticks(main_lengths)
    ax.set_xticklabels([f"{length:,}" for length in main_lengths])
    ax.set_xlabel("Input length $N$")
    ax.set_ylim(0.48, 1.015)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Final PARITY accuracy")
    ax.axhline(
        0.5,
        color="#B1B4B8",
        linewidth=0.9,
        linestyle=(0, (3, 3)),
        zorder=1,
        clip_on=False,
    )
    ax.legend(
        loc="center right",
        handlelength=1.75,
        borderaxespad=0.35,
        labelspacing=0.25,
        fontsize=7.0,
    )


def draw_controls(
    ax: plt.Axes,
    controls: dict[str, list[tuple[int, float]]],
) -> None:
    # Plot Normal last with a dashed stroke. Where Normal and Encrypted are
    # both exactly 100%, alternating blue/orange segments remain visible.
    for condition in (
        "no_cot",
        "filler_cot",
        "no_cot_end_to_end",
        "filler_cot_end_to_end",
        "encrypted_cot",
        "normal_cot",
    ):
        rows = controls[condition]
        end_to_end = condition.endswith("_end_to_end")
        if end_to_end:
            linestyle, linewidth, zorder = "None", 0, 6
        elif condition == "normal_cot":
            linestyle, linewidth, zorder = (0, (3.0, 2.2)), 2.0, 5
        elif condition == "encrypted_cot":
            linestyle, linewidth, zorder = "-", 2.5, 4
        else:
            linestyle, linewidth, zorder = "--", 1.6, 3
        ax.plot(
            display_x(rows, CONTROL_X_DODGE[condition]),
            [row[1] for row in rows],
            color=CONTROL_COLORS[condition],
            marker=CONTROL_MARKERS[condition],
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=5.0,
            markeredgewidth=0.7,
            markeredgecolor=CONTROL_COLORS[condition] if end_to_end else "white",
            markerfacecolor="none" if end_to_end else CONTROL_COLORS[condition],
            label=CONTROL_LABELS[condition],
            zorder=zorder,
        )
    control_lengths = (4, 8, 16, 32, 64, 256, 512, 1024)
    ax.set_xscale("log", base=2)
    ax.set_xticks(control_lengths)
    ax.set_xticklabels([f"{length:,}" for length in control_lengths])
    ax.set_xlabel("Input length $N$")
    ax.set_ylabel("Final PARITY accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.axhline(0.5, color="#A7AAAD", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.text(
        0.985,
        0.515,
        "chance",
        transform=ax.get_yaxis_transform(),
        color=SECONDARY,
        fontsize=8.6,
        ha="right",
        va="bottom",
    )
    ax.tick_params(labelsize=9.1)
    ax.xaxis.label.set_size(10.2)
    ax.yaxis.label.set_size(10.2)


def draw_ablation_panel(
    ax: plt.Axes,
    ablations: dict[str, list[dict[str, float | int]]],
    metric: str,
    title: str,
    chance: bool,
) -> None:
    for label, rows in ablations.items():
        plotted = [(int(row["N"]), float(row[metric])) for row in rows]
        ax.plot(
            display_x(plotted, ABLATION_X_DODGE[label]),
            [row[1] for row in plotted],
            color=ABLATION_COLORS[label],
            marker=(
                "o" if label == "Full" else ("s" if label == "No route labels" else "^")
            ),
            linestyle="-" if label != "End-to-end only" else "--",
            linewidth=1.9,
            markersize=5.4,
            markeredgewidth=0.7,
            markeredgecolor="white",
            label=label,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64])
    ax.set_xticklabels(["32", "64"])
    ax.set_xlabel("Input length $N$")
    ax.set_ylim(-0.025, 1.035)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Test accuracy")
    if chance:
        ax.axhline(0.5, color="#A7AAAD", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.set_title(title, loc="left", fontweight="semibold")


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf")
    fig.savefig(OUTPUT_ROOT / f"{stem}.svg")
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=300)


def write_data(results: dict[int, list[dict[str, float | int]]]) -> None:
    fields = [
        "d_model",
        "N",
        "seed_bits",
        "parameters",
        "state_bit_accuracy",
        "trace_exact",
        "final_parity_accuracy",
        "joint_exact",
    ]
    with (OUTPUT_ROOT / "figure_goldreich_parity_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_dim in MODEL_DIMS:
            writer.writerows(results[model_dim])


def write_control_data(
    controls: dict[str, list[tuple[int, float]]],
) -> None:
    with (OUTPUT_ROOT / "figure_goldreich_parity_controls_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("condition", "training_regime", "N", "final_parity_accuracy"),
        )
        writer.writeheader()
        for condition in (
            "no_cot",
            "filler_cot",
            "no_cot_end_to_end",
            "filler_cot_end_to_end",
            "normal_cot",
            "encrypted_cot",
        ):
            for length, accuracy in controls[condition]:
                writer.writerow(
                    {
                        "condition": condition,
                        "training_regime": (
                            "end_to_end"
                            if condition.endswith("_end_to_end")
                            else (
                                "staged"
                                if condition in {"no_cot", "filler_cot"}
                                else "task_specific"
                            )
                        ),
                        "N": length,
                        "final_parity_accuracy": accuracy,
                    }
                )


def write_main_performance_data(
    controls: dict[str, list[tuple[int, float]]],
) -> None:
    path = OUTPUT_ROOT / "figure_goldreich_parity_performance_data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("condition", "training_regime", "N", "final_parity_accuracy"),
        )
        writer.writeheader()
        for condition in (
            "no_cot",
            "filler_cot",
            "no_cot_end_to_end",
            "filler_cot_end_to_end",
        ):
            for length, accuracy in controls[condition]:
                writer.writerow(
                    {
                        "condition": condition,
                        "training_regime": (
                            "end_to_end"
                            if condition.endswith("_end_to_end")
                            else "staged"
                        ),
                        "N": length,
                        "final_parity_accuracy": accuracy,
                    }
                )
        for length, accuracy in controls["encrypted_cot"]:
            writer.writerow(
                {
                    "condition": "encrypted_cot",
                    "training_regime": "staged_encrypted",
                    "N": length,
                    "final_parity_accuracy": accuracy,
                }
            )


def write_ablation_data(ablations: dict[str, list[dict[str, float | int]]]) -> None:
    with (OUTPUT_ROOT / "figure_goldreich_parity_ablations_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("supervision", "N", "state_bit_accuracy", "joint_exact")
        )
        writer.writeheader()
        for label, rows in ablations.items():
            for row in rows:
                writer.writerow({"supervision": label, **row})


def main() -> None:
    configure_style()
    results = read_results()
    controls = read_controls(results)
    ablations = read_ablations(results)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(3.55, 2.05))
    draw_main_performance(axis, controls)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.25, top=0.97)
    save(fig, "figure_goldreich_parity_performance")
    plt.close(fig)

    compact, compact_axis = plt.subplots(figsize=(3.65, 2.75))
    draw_main_performance(compact_axis, controls)
    compact.subplots_adjust(left=0.18, right=0.99, bottom=0.19, top=0.89)
    save(compact, "figure_goldreich_parity_joint")
    plt.close(compact)

    control_figure, control_axis = plt.subplots(figsize=(4.75, 3.05))
    draw_controls(control_axis, controls)
    handles, labels = control_axis.get_legend_handles_labels()
    legend_labels = (
        CONTROL_LABELS["encrypted_cot"],
        CONTROL_LABELS["normal_cot"],
        CONTROL_LABELS["no_cot"],
        CONTROL_LABELS["filler_cot"],
        CONTROL_LABELS["no_cot_end_to_end"],
        CONTROL_LABELS["filler_cot_end_to_end"],
    )
    order = [labels.index(label) for label in legend_labels]
    control_figure.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.1,
        labelspacing=0.35,
        fontsize=8.8,
    )
    control_figure.subplots_adjust(left=0.15, right=0.99, bottom=0.19, top=0.66)
    save(control_figure, "figure_goldreich_parity_controls")
    plt.close(control_figure)

    ablation_figure, ablation_axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    draw_ablation_panel(
        ablation_axes[0],
        ablations,
        "state_bit_accuracy",
        "(a) Encrypted state bits",
        chance=True,
    )
    draw_ablation_panel(
        ablation_axes[1],
        ablations,
        "joint_exact",
        "(b) Complete CoT + answer",
        chance=False,
    )
    handles, labels = ablation_axes[0].get_legend_handles_labels()
    ablation_figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        handlelength=2.5,
        columnspacing=1.5,
    )
    ablation_figure.subplots_adjust(
        left=0.085, right=0.995, bottom=0.19, top=0.78, wspace=0.27
    )
    save(ablation_figure, "figure_goldreich_parity_ablations")
    plt.close(ablation_figure)

    write_data(results)
    write_control_data(controls)
    write_main_performance_data(controls)
    write_ablation_data(ablations)
    manifest = {
        "source_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "source_artifacts": str(RESULT_ROOT.relative_to(REPO_ROOT)),
        "primary_figure": "figure_goldreich_parity_performance",
        "compact_figure": "figure_goldreich_parity_joint",
        "control_figure": "figure_goldreich_parity_controls",
        "ablation_figure": "figure_goldreich_parity_ablations",
        "formats": ["pdf", "svg", "png"],
        "primary_figure_contents": [
            "encrypted-CoT final-answer accuracy",
            "no-CoT final-answer accuracy",
            "filler-only-CoT final-answer accuracy (repeated zero tokens)",
        ],
        "note": (
            "No-CoT and filler-only controls use one staged training procedure at "
            "N=4,8,16,32,64. Matched end-to-end N=16 controls are shown as separate "
            "hollow markers. Small "
            "horizontal marker offsets separate coincident series visually; source "
            "CSVs retain the exact input lengths and accuracies."
        ),
        "primary_figure_data": "figure_goldreich_parity_performance_data.csv",
        "attention_at_inference": "ordinary scaled dot-product softmax",
        "test_examples_per_condition": 8192,
        "model_widths": list(MODEL_DIMS),
        "input_lengths": list(LENGTHS),
        "control_source_artifacts": str(CONTROL_ROOT.relative_to(REPO_ROOT)),
        "staged_large_control_source_artifacts": str(
            STAGED_CONTROL_ROOT.relative_to(REPO_ROOT)
        ),
        "matched_end_to_end_control_source_artifacts": str(
            MATCHED_END_TO_END_ROOT.relative_to(REPO_ROOT)
        ),
        "ablation_source_artifacts": str(ABLATION_ROOT.relative_to(REPO_ROOT)),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote final paper figure to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
