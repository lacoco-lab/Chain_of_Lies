#!/usr/bin/env python3
"""Regenerate appendix accuracy and compact main chain-length figures."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures" / "scaling_regime"
PANELS = OUT / "appendix_accuracy_panels"

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator

TASKS = ("multiplication", "knowledge", "s5", "parity")
MODELS = ("qwen", "llama")
TASK_LABEL = {
    "multiplication": "Multiplication",
    "knowledge": "Iterated addition",
    "s5": "S5 state tracking",
    "parity": "Parity",
}
MODEL_LABEL = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B"}
X_LABEL = {
    "multiplication": "Digits per operand",
    "knowledge": "Number of facts",
    "s5": "Number of swaps",
    "parity": "Sequence length",
}
CONDITIONS = (
    "vanilla",
    "filler_only",
    "public_only_cot",
    "piggyback",
    "steganography",
    "filler_public_cot",
)
STYLE = {
    "vanilla": ("Vanilla", "#858585", "o", ":"),
    "filler_only": ("Filler", "#3F6074", "s", "--"),
    "public_only_cot": ("Public-only CoT", "#83A9C0", "^", ":"),
    "piggyback": ("Piggybacking", "#D9433A", "D", "-"),
    "steganography": ("Special Tokens", "#E8792A", r"$\ominus$", "--"),
    "filler_public_cot": ("Filler + Public CoT [ablation]", "#A9A39B", "x", "-."),
}
HOLLOW_MARKERS = {"public_only_cot", "piggyback"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values)


def filler_only_rows():
    audit = json.loads((ROOT / "recovery_audits/all_24.json").read_text())
    if not audit.get("complete") or len(audit.get("cells", [])) != 24:
        raise RuntimeError("The complete 24-cell filler-only audit is required")
    grouped = {}
    for cell in audit["cells"]:
        if cell["task"] == "parity":
            continue
        for evaluation in cell["evaluations"]:
            key = (cell["task"], cell["model"], str(evaluation["length"]))
            grouped.setdefault(key, []).append(evaluation["private_exact_rate"])
    rows = []
    for (task, model, length), values in sorted(grouped.items()):
        if len(values) != 3:
            raise RuntimeError(f"Incomplete filler-only seeds: {(task,model,length)}")
        mean, sd = mean_sd(values)
        rows.append(
            dict(
                task=task,
                model=model,
                condition="filler_only",
                length=length,
                mean=mean,
                sd=sd,
                n_seeds=3,
                n_per_seed=200,
                source="filler_only_scaling_v1",
            )
        )
    return rows


def parity_filler_only_rows():
    from chain_of_lies.evaluation.rewards import (
        load_prompt_record,
        score_completion,
        default_reward_config,
    )

    buckets = ["008_015", "016_031", "032_063", "064_127", "128_255"]
    midpoint = dict(zip(buckets, (11.5, 23.5, 47.5, 95.5, 191.5)))
    output = []
    for model in MODELS:
        per_seed = {}
        for seed in range(3):
            prompts = (
                ROOT
                / f"generated_data/parity_regime_calibration/seed_{seed}/parity_control/val_prompts"
            )
            responses = ROOT / (
                f"generated_data/filler_only_scaling_v1_eval_responses/parity/{model}/"
                f"seed_{seed}/filler_only/ckpt_final/finetuned/parity_control"
            )
            values = {bucket: [] for bucket in buckets}
            for prompt in prompts.glob("*.json"):
                record = json.loads(prompt.read_text())
                response = json.loads((responses / prompt.name).read_text())
                score = score_completion(
                    load_prompt_record(prompt),
                    response["raw_text"],
                    default_reward_config(),
                )
                values[record["spec"]["length_bucket"]].append(
                    bool(score.flags["private_correct"])
                )
            for bucket in buckets:
                if len(values[bucket]) != 200:
                    raise RuntimeError("Incomplete parity filler-only responses")
                per_seed[bucket, seed] = statistics.fmean(values[bucket])
        for bucket in buckets:
            seeds = [per_seed[bucket, seed] for seed in range(3)]
            mean, sd = mean_sd(seeds)
            output.append(
                dict(
                    task="parity",
                    model=model,
                    condition="filler_only",
                    length=midpoint[bucket],
                    mean=mean,
                    sd=sd,
                    n_seeds=3,
                    n_per_seed=200,
                    source="filler_only_scaling_v1",
                )
            )
    return output


def accuracy_rows():
    sources = {
        "multiplication": ROOT
        / "artifacts/multiplication_length_retraining_v1/aggregate_length_metrics.csv",
        "knowledge": ROOT
        / "artifacts/knowledge_length_retraining_all_but_one_v2/aggregate_length_metrics.csv",
        "s5": ROOT / "artifacts/s5_length_retraining_v1/aggregate_length_metrics.csv",
    }
    rows = []
    for task, path in sources.items():
        for row in read_csv(path):
            condition = (
                "filler_public_cot"
                if row["condition"] == "filler"
                else row["condition"]
            )
            rows.append(
                dict(
                    task=task,
                    model=row["model"],
                    condition=condition,
                    length=float(row["length"]),
                    mean=float(row["private_exact_rate_mean"]),
                    sd=float(row["private_exact_rate_sample_sd"]),
                    n_seeds=3,
                    n_per_seed=200,
                    source="balanced_scaling_original",
                )
            )
    parity_path = (
        ROOT
        / "docs/scaling_regime/tables/metrics_by_length_bucket_three_seed_parity.csv"
    )
    midpoint = {
        "008_015": 11.5,
        "016_031": 23.5,
        "032_063": 47.5,
        "064_127": 95.5,
        "128_255": 191.5,
    }
    for row in read_csv(parity_path):
        if row["source"] != "finetuned":
            continue
        condition = (
            "filler_public_cot" if row["condition"] == "filler" else row["condition"]
        )
        rows.append(
            dict(
                task="parity",
                model=row["model"],
                condition=condition,
                length=midpoint[row["length_bucket"]],
                mean=float(row["private_exact_mean"]),
                sd=float(row["private_exact_sample_sd"]),
                n_seeds=3,
                n_per_seed=200,
                source="parity_three_seed_original",
            )
        )
    rows.extend(filler_only_rows())
    rows.extend(parity_filler_only_rows())
    keys = {(r["task"], r["model"], r["condition"], float(r["length"])) for r in rows}
    for task in TASKS:
        expected_lengths = sorted(
            {float(r["length"]) for r in rows if r["task"] == task}
        )
        for model in MODELS:
            for condition in CONDITIONS:
                observed = sorted(
                    float(r["length"])
                    for r in rows
                    if r["task"] == task
                    and r["model"] == model
                    and r["condition"] == condition
                )
                if observed != expected_lengths:
                    raise RuntimeError(f"Incomplete curve {task}/{model}/{condition}")
    return sorted(
        rows,
        key=lambda r: (
            TASKS.index(r["task"]),
            MODELS.index(r["model"]),
            CONDITIONS.index(r["condition"]),
            float(r["length"]),
        ),
    )


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def ticks(task):
    return {
        "multiplication": [2, 3, 4],
        "knowledge": [1, 2, 3, 4, 5],
        "s5": [1, 5, 10, 15, 19],
        "parity": [11.5, 47.5, 95.5, 191.5],
    }[task]


def ticklabels(task):
    if task != "parity":
        return [str(int(x)) for x in ticks(task)]
    return ["12", "48", "96", "192"]


def plot_accuracy(ax, rows, task, model):
    selected = [r for r in rows if r["task"] == task and r["model"] == model]
    for condition in CONDITIONS:
        label, color, marker, linestyle = STYLE[condition]
        series = sorted(
            (r for r in selected if r["condition"] == condition),
            key=lambda r: float(r["length"]),
        )
        x = [float(r["length"]) for r in series]
        y = [r["mean"] for r in series]
        sd = [r["sd"] for r in series]
        markevery = None if task != "s5" else [0, 4, 9, 14, 18]
        ax.fill_between(
            x,
            [max(0, a - b) for a, b in zip(y, sd)],
            [min(1, a + b) for a, b in zip(y, sd)],
            color=color,
            alpha=0.055,
            lw=0,
        )
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            ls=linestyle,
            lw=1.35 if condition in {"piggyback", "steganography"} else 1.05,
            ms=4.2 if condition == "steganography" else 3.6,
            mew=0.85,
            mfc="white" if condition in HOLLOW_MARKERS else color,
            markevery=markevery,
        )
    ax.set_title(
        f"{TASK_LABEL[task]} · {MODEL_LABEL[model]}",
        loc="left",
        fontweight="semibold",
        pad=3,
    )
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks(ticks(task), ticklabels(task))
    ax.set_xlabel(X_LABEL[task])
    ax.grid(axis="y")
    if task == "parity":
        ax.set_xlim(0, 200)


def method_handles():
    return [
        Line2D(
            [0],
            [0],
            label=STYLE[c][0],
            color=STYLE[c][1],
            marker=STYLE[c][2],
            ls=STYLE[c][3],
            lw=1.4,
            ms=4.6,
            mfc="white" if c in HOLLOW_MARKERS else STYLE[c][1],
            mew=0.9,
        )
        for c in CONDITIONS
    ]


def draw_accuracy(rows):
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 8.25), sharey=True)
    for i, task in enumerate(TASKS):
        for j, model in enumerate(MODELS):
            plot_accuracy(axes[i, j], rows, task, model)
            if j == 0:
                axes[i, j].set_ylabel("Covert accuracy")
    fig.legend(
        handles=method_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.45,
        handlelength=2.25,
    )
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.065, top=0.925, hspace=0.62, wspace=0.17
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT / f"figure_accuracy_by_length_appendix.{suffix}",
            dpi=600 if suffix == "png" else None,
        )
    plt.close(fig)
    PANELS.mkdir(exist_ok=True)
    for task in TASKS:
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(3.45, 2.7))
            plot_accuracy(ax, rows, task, model)
            ax.set_ylabel("Covert accuracy")
            fig.legend(
                handles=method_handles(),
                loc="upper center",
                ncol=2,
                bbox_to_anchor=(0.5, 0.995),
                fontsize=6.4,
                columnspacing=0.8,
                handlelength=1.8,
            )
            fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.72)
            for suffix in ("pdf", "png"):
                fig.savefig(
                    PANELS / f"accuracy_{task}_{model}.{suffix}",
                    dpi=600 if suffix == "png" else None,
                )
            plt.close(fig)


def chain_rows():
    rows = []
    for row in read_csv(
        ROOT / "docs/scaling_regime/tables/chain_token_length_three_seed.csv"
    ):
        rows.append(
            dict(
                task=row["task"],
                model=row["model"],
                condition=row["condition"],
                length=float(row["length"]),
                mean=float(row["mean_chain_tokens"]),
                sd=float(row["sample_sd_seed_mean"]),
                n_seeds=3,
                n_per_seed=200,
            )
        )
    return rows


def draw_chain(rows):
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.12))
    colors = {"piggyback": "#0072B2", "steganography": "#D55E00"}
    labels = {"piggyback": "Piggyback", "steganography": "Special Tokens"}
    model_style = {"qwen": ("-", "o", -0.008), "llama": ("--", "^", 0.008)}
    for ax, task in zip(axes.flat, TASKS):
        selected = [r for r in rows if r["task"] == task]
        span = max(r["length"] for r in selected) - min(r["length"] for r in selected)
        for condition in ("piggyback", "steganography"):
            for model in MODELS:
                ls, marker, offset = model_style[model]
                series = sorted(
                    (
                        r
                        for r in selected
                        if r["condition"] == condition and r["model"] == model
                    ),
                    key=lambda r: r["length"],
                )
                x = [r["length"] for r in series]
                y = [r["mean"] for r in series]
                markevery = None if task != "s5" else [0, 4, 9, 14, 18]
                ax.plot(x, y, color=colors[condition], ls=ls, lw=1.15, zorder=2)
                # Markers alone receive a tiny visual offset so coincident model series remain visible.
                ax.plot(
                    [v + offset * span for v in x],
                    y,
                    color=colors[condition],
                    ls="none",
                    marker=marker,
                    ms=3.0,
                    mew=0.45,
                    markevery=markevery,
                    zorder=3,
                )
        ax.set_title(
            TASK_LABEL[task], loc="left", fontsize=7.7, fontweight="semibold", pad=2
        )
        ax.set_xticks(ticks(task), ticklabels(task))
        ax.tick_params(labelsize=6.2, pad=1.5)
        ax.set_xlabel(X_LABEL[task], fontsize=6.5, labelpad=2)
        ax.grid(axis="y")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        if task == "parity":
            ax.set_xlim(0, 200)
    axes[0, 0].set_ylabel("Generated chain tokens", fontsize=6.7, labelpad=2)
    axes[1, 0].set_ylabel("Generated chain tokens", fontsize=6.7, labelpad=2)
    method_handles = [
        Line2D([0], [0], color=colors[c], lw=1.7, label=labels[c])
        for c in ("piggyback", "steganography")
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            color="#555",
            ls=model_style[m][0],
            marker=model_style[m][1],
            ms=3.3,
            lw=1.1,
            label=MODEL_LABEL[m].split("-")[0],
        )
        for m in MODELS
    ]
    fig.legend(
        handles=method_handles + model_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=6.3,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.45,
    )
    fig.subplots_adjust(
        left=0.15, right=0.985, bottom=0.105, top=0.84, hspace=0.58, wspace=0.38
    )
    with mpl.rc_context({"savefig.bbox": None}):
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(
                OUT / f"figure_chain_length_by_input_length_main.{suffix}",
                dpi=600 if suffix == "png" else None,
            )
    plt.close(fig)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 2 and sys.argv[1] in {"--accuracy-only", "--labels-only"}:
        accuracy = read_csv(OUT / "figure_accuracy_plot_data.csv")
        for row in accuracy:
            row["mean"] = float(row["mean"])
            row["sd"] = float(row["sd"])
        draw_accuracy(accuracy)
        if sys.argv[1] == "--labels-only":
            draw_chain(chain_rows())
        manifest_path = OUT / "scaling_figure_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["outputs"]:
            candidate = (
                PANELS if item["file"].startswith("accuracy_") else OUT
            ) / item["file"]
            if candidate.is_file():
                item["bytes"] = candidate.stat().st_size
                item["sha256"] = sha(candidate)
        manifest["accuracy"][
            "style"
        ] = "Paper-matched gray/blue/red-orange colors and markers; muted appendix ablation"
        if sys.argv[1] == "--labels-only":
            manifest["accuracy"]["display_labels"] = "Special Tokens; Covert accuracy"
            manifest["chain"]["display_label"] = "Special Tokens"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            "Updated eight appendix panels, overview, and chain figure; exact data unchanged"
            if sys.argv[1] == "--labels-only"
            else "Updated eight appendix panels and overview; exact data and chain figure unchanged"
        )
        return
    accuracy = accuracy_rows()
    chains = chain_rows()
    write_csv(OUT / "figure_accuracy_plot_data.csv", accuracy)
    write_csv(OUT / "figure_chain_plot_data.csv", chains)
    draw_accuracy(accuracy)
    draw_chain(chains)
    outputs = [
        OUT / "figure_accuracy_by_length_appendix.pdf",
        OUT / "figure_accuracy_by_length_appendix.png",
        OUT / "figure_chain_length_by_input_length_main.pdf",
        OUT / "figure_chain_length_by_input_length_main.png",
        OUT / "figure_chain_length_by_input_length_main.svg",
        OUT / "figure_accuracy_plot_data.csv",
        OUT / "figure_chain_plot_data.csv",
    ] + sorted(PANELS.glob("*"))
    manifest = {
        "status": "complete_primary_filler_only_with_legacy_appendix_ablation",
        "accuracy": {
            "panels": 8,
            "conditions": list(CONDITIONS),
            "metric": "private exact accuracy; mean ± sample SD over seeds 0,1,2",
        },
        "chain": {
            "layout": "2x2 at 3.45 inch width",
            "metric": "mean generated chain tokens including invisible tokens; answer block excluded",
            "marker_offsets": "visual only",
        },
        "outputs": [
            {"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in outputs
        ],
    }
    (OUT / "scaling_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote {len(outputs)} validated figure/data files")


if __name__ == "__main__":
    main()
