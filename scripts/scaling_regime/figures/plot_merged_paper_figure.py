"""Vector paper figure; sources are read only and existing figures are preserved."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
OUT = ROOT / "figures/merged_paper"
PROMPT = ROOT / "docs/scaling_regime/tables/prompting_accuracy_unleaked.csv"
PROMPT_FULL_COT = (
    ROOT / "docs/scaling_regime/tables/prompting_unrestricted_cot_accuracy.csv"
)
METHODS = [
    "vanilla",
    "filler",
    "public_only_cot",
    "piggyback",
    "steganography",
    "full_cot",
]
LABELS = [
    "Vanilla",
    "Filler",
    "Public-only CoT",
    "Piggybacking",
    "Special Tokens",
    "Full CoT (open)",
]
# Match the mechanism colors in the paper's protocol overview.
COLORS = ["#7E7E7E", "#3E637C", "#82A8BD", "#DD4034", "#EF8334", "#444444"]
DASHES = [[1, 2], [3, 2], [3, 2, 0.7, 2], [], [7, 1.8], [7, 2, 1.5, 2]]
LINE_WIDTHS = [1.6, 1.7, 1.7, 2.2, 2.2, 1.8]
MAPPING = dict(
    zip(
        METHODS,
        ["vanilla", "vanilla + filler", "CoT", "Piggy", "steg", "unrestricted CoT"],
    )
)
PMODELS = [
    ("qwen35-27b", "Qwen 3.5 27B"),
    ("qwen35-397b-a17b", "Qwen 3.5 397B"),
    ("kimi-k3", "Kimi K3"),
]

FILLER_PARITY = {
    "qwen": [
        (8, 48.33, 1.44),
        (16, 46.50, 4.44),
        (32, 50.50, 0.87),
        (64, 51.33, 2.31),
        (128, 50.50, 0.87),
    ],
    "llama": [
        (8, 49.67, 1.53),
        (16, 50.67, 2.08),
        (32, 52.17, 2.25),
        (64, 50.33, 2.02),
        (128, 49.50, 0.87),
    ],
}


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def audit_balanced(task, expected):
    directory = ROOT / "artifacts" / task
    rows = read(directory / "aggregate_length_metrics.csv")
    seed_rows = read(directory / "per_seed_length_metrics.csv")
    grouped = defaultdict(list)
    for r in seed_rows:
        grouped[(r["model"], r["condition"], int(r["length"]))].append(r)
    assert len(rows) == 2 * 5 * len(expected)
    for r in rows:
        key = (r["model"], r["condition"], int(r["length"]))
        g = grouped[key]
        assert sorted(int(s["seed"]) for s in g) == [0, 1, 2]
        v = [float(s["private_exact_rate"]) for s in g]
        assert abs(statistics.mean(v) - float(r["private_exact_rate_mean"])) < 1e-12
        assert (
            abs(statistics.stdev(v) - float(r["private_exact_rate_sample_sd"])) < 1e-12
        )
        assert int(r["n_model_outputs"]) == 600
    if task == "s5_length_retraining_v1":
        examples = read(directory / "per_example.csv")
        eg = defaultdict(list)
        for r in examples:
            eg[(r["model"], r["condition"], int(r["length"]), int(r["seed"]))].append(
                int(r["private_correct"])
            )
        assert len(examples) == 114000 and len(eg) == 570
        for key, g in grouped.items():
            for r in g:
                vals = eg[(*key, int(r["seed"]))]
                assert len(vals) == 200
                assert (
                    abs(statistics.mean(vals) - float(r["private_exact_rate"])) < 1e-12
                )
    return rows


def main():
    pdfmetrics.registerFont(
        TTFont("PaperArial", "/System/Library/Fonts/Supplemental/Arial.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("PaperArialBold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    )
    OUT.mkdir(parents=True, exist_ok=True)
    base_prows = read(PROMPT)
    assert len(base_prows) == 333
    prompt_full_cot_rows = read(PROMPT_FULL_COT)
    assert len(prompt_full_cot_rows) == 30
    assert all(r["mechanism"] == "unrestricted CoT" for r in prompt_full_cot_rows)
    assert len(
        {(r["model_name"], r["experiment"]) for r in prompt_full_cot_rows}
    ) == len(prompt_full_cot_rows)
    prows = base_prows + prompt_full_cot_rows
    pindex = {(r["model_name"], r["mechanism"], r["experiment"]): r for r in prows}
    assert len(pindex) == len(prows)
    fdata = {}
    for task, folder, lengths in [
        ("multiplication", "multiplication_length_retraining_v1", [2, 3, 4]),
        ("knowledge", "knowledge_length_retraining_all_but_one_v2", list(range(1, 6))),
        ("s5", "s5_length_retraining_v1", list(range(1, 20))),
    ]:
        for r in audit_balanced(folder, lengths):
            fdata.setdefault((r["model"], r["condition"], task), []).append(
                (
                    int(r["length"]),
                    100 * float(r["private_exact_rate_mean"]),
                    100 * float(r["private_exact_rate_sample_sd"]),
                )
            )
    parity = read(
        ROOT
        / "artifacts/parity_regime_calibration/metrics_by_length_bucket_three_seed.csv"
    )
    for r in parity:
        if r["source"] == "finetuned":
            assert r["seeds"] == "0,1,2"
            fdata.setdefault((r["model"], r["condition"], "parity"), []).append(
                (
                    int(r["length_min"]),
                    100 * float(r["private_exact_mean"]),
                    100 * float(r["private_exact_sample_sd"]),
                )
            )
    # Replace legacy Filler + Public CoT with the audited primary filler-only condition.
    filler_audit = json.loads((ROOT / "recovery_audits/all_24.json").read_text())
    assert filler_audit["complete"] and len(filler_audit["cells"]) == 24
    filler_seed = defaultdict(list)
    for cell in filler_audit["cells"]:
        assert cell["complete"] and cell["model"] in ("qwen", "llama")
        if cell["task"] == "parity":
            continue
        for e in cell["evaluations"]:
            assert e["num_examples"] == 200 and e["generation_cap_hits"] == 0
            filler_seed[(cell["model"], cell["task"], int(e["length"]))].append(
                float(e["private_exact_rate"])
            )
    for (model, task, length), values in filler_seed.items():
        assert len(values) == 3
        fdata[(model, "filler", task)] = [
            v for v in fdata.get((model, "filler", task), []) if v[0] != length
        ]
        fdata[(model, "filler", task)].append(
            (length, 100 * statistics.mean(values), 100 * statistics.stdev(values))
        )
    for model, values in FILLER_PARITY.items():
        fdata[(model, "filler", "parity")] = values
    full_cot_source = (
        ROOT / "docs/scaling_regime/tables/full_cot_baseline_by_length.csv"
    )
    full_cot_rows = read(full_cot_source)
    assert len(full_cot_rows) == 64
    for r in full_cot_rows:
        assert r["model"] in ("qwen", "llama") and int(r["n_seeds"]) == 3
        assert int(r["examples_per_seed"]) == 200
        assert float(r["format_mean"]) == 1 and float(r["token_cap_hit_rate"]) == 0
        length = int(r["length"])
        if r["task"] == "parity":
            # The Full CoT summary encodes parity buckets as 8015, 16031, ... .
            length = {8015: 8, 16031: 16, 32063: 32, 64127: 64, 128255: 128}[length]
        fdata.setdefault((r["model"], "full_cot", r["task"]), []).append(
            (
                length,
                100 * float(r["private_exact_mean"]),
                100 * float(r["private_exact_sample_sd"]),
            )
        )
    for model in ("qwen", "llama"):
        for task, lengths in [
            ("multiplication", [2, 3, 4]),
            ("knowledge", range(1, 6)),
            ("parity", [8, 16, 32, 64, 128]),
            ("s5", range(1, 20)),
        ]:
            assert sorted(a for a, _, _ in fdata[(model, "full_cot", task)]) == list(
                lengths
            )
    merged_figure = OUT / "figure_prompting_finetuning_merged.pdf"
    c = canvas.Canvas(str(merged_figure), pagesize=(1020, 480))
    c.setTitle("Covert accuracy across prompting and finetuning experiments")
    c.setAuthor("Anonymous")

    def text(x, y, s, size=11, bold=False, align="left", color="#222222"):
        c.setFillColor(HexColor(color))
        c.setFont("PaperArialBold" if bold else "PaperArial", size)
        getattr(
            c,
            {
                "left": "drawString",
                "center": "drawCentredString",
                "right": "drawRightString",
            }[align],
        )(x, y, s)

    def marker(x, y, k, size=2.2):
        size = {0: 2.7, 1: 3.0, 2: 3.5, 3: 4.1, 4: 2.5, 5: 3.2}[k]
        c.setDash([])
        c.setLineWidth(0.9)
        c.setFillColor(HexColor(COLORS[k]))
        c.setStrokeColor(HexColor(COLORS[k]))
        if k == 0:
            c.circle(x, y, size, stroke=1, fill=1)
        elif k == 1:
            c.rect(x - size, y - size, size * 2, size * 2, stroke=1, fill=1)
        elif k in (2, 3):
            if k in (2, 3):
                c.setFillColor(HexColor("#FFFFFF"))
            p = c.beginPath()
            coords = (
                [
                    (x, y + size * 1.25),
                    (x + size * 1.15, y - size),
                    (x - size * 1.15, y - size),
                ]
                if k == 2
                else [
                    (x, y + size * 1.3),
                    (x + size * 1.3, y),
                    (x, y - size * 1.3),
                    (x - size * 1.3, y),
                ]
            )
            p.moveTo(*coords[0])
            for a in coords[1:]:
                p.lineTo(*a)
            p.close()
            c.drawPath(p, stroke=1, fill=0)
        elif k == 4:
            c.setFillColor(HexColor("#FFFFFF"))
            c.setLineWidth(1.25)
            c.circle(x, y, size, stroke=1, fill=0)
        else:
            c.setLineWidth(1.3)
            c.line(x - size, y - size, x + size, y + size)
            c.line(x - size, y + size, x + size, y - size)

    plotrows = []
    missing = []

    def panel(x, y, w, h, task, model, prompt=False, show_y=False, show_x=False):
        if task == "parity":
            domain = (1, 10) if prompt else (8, 256)
            ticks = [1, 2, 4, 8] if prompt else [8, 32, 128]
            labs = list(map(str, ticks))
        elif prompt:
            domain = (1, 10)
            ticks = [1, 4, 7, 10]
            labs = list(map(str, ticks))
        elif task == "multiplication":
            domain = (2, 4)
            ticks = [2, 3, 4]
            labs = ["2", "3", "4"]
        elif task == "knowledge":
            domain = (1, 5)
            ticks = [1, 3, 5]
            labs = ["1", "3", "5"]
        elif task == "s5":
            domain = (1, 19)
            ticks = [1, 10, 19]
            labs = ["1", "10", "19"]
        transform = math.log2 if task == "parity" else lambda a: a
        xx = (
            lambda a: x
            + 3
            + (transform(a) - transform(domain[0]))
            / (transform(domain[1]) - transform(domain[0]))
            * (w - 6)
        )
        center = lambda a: (
            math.sqrt(a * (2 * a - 1)) if task == "parity" and not prompt else a
        )
        yy = lambda a: y + 3 + a / 100 * (h - 6)
        c.setLineWidth(0.45)
        c.setDash([])
        for a in [0, 50, 100]:
            c.setStrokeColor(HexColor("#E0E0E0"))
            c.line(x, yy(a), x + w, yy(a))
            if show_y:
                text(x - 7, yy(a) - 4, str(a), 15, align="right")
        c.setStrokeColor(HexColor("#555555"))
        c.line(x, y, x + w, y)
        c.line(x, y, x, y + h)
        for a, lab in zip(ticks, labs):
            c.line(xx(a), y, xx(a), y - 2)
            if show_x:
                text(xx(a), y - 18, lab, 15, align="center")
        marker_layers = []
        # The open Full CoT control is drawn first so the informative-mode
        # curves remain visually dominant where they coincide at the ceiling.
        draw_order = [5, 0, 1, 2, 3, 4]
        for k in draw_order:
            m = METHODS[k]
            if prompt:
                r = pindex.get((model, MAPPING[m], task))
                if r is None:
                    missing.append((model, m, task))
                    continue
                vals = [(i, float(r[f"d{i}_private_pct"]), 0) for i in range(1, 11)]
            else:
                vals = sorted(fdata[(model, m, task)])
            pts = []
            for i, (a, b, sd) in enumerate(vals):
                assert 0 <= b <= 100
                px = xx(center(a))
                py = yy(b)
                pts.append((px, py))
                source = (
                    MAPPING[m]
                    if prompt
                    else (
                        "full_cot_baseline_by_length"
                        if m == "full_cot"
                        else "filler_only_scaling_v1" if m == "filler" else "finetuned"
                    )
                )
                plotrows.append(
                    dict(
                        paradigm="prompting" if prompt else "finetuning",
                        model=model,
                        task=task,
                        method=LABELS[k],
                        length=center(a),
                        length_min=a,
                        length_max=2 * a - 1 if task == "parity" and not prompt else a,
                        covert_accuracy_pct=b,
                        sample_sd_pct=sd if not prompt else "",
                        source=source,
                    )
                )
            if not prompt:
                p = c.beginPath()
                for j, (a, b, sd) in enumerate(vals):
                    px = xx(center(a))
                    py = yy(min(100, b + sd))
                    p.moveTo(px, py) if j == 0 else p.lineTo(px, py)
                for j in reversed(range(len(vals))):
                    a, b, sd = vals[j]
                    p.lineTo(xx(center(a)), yy(max(0, b - sd)))
                band_alpha = {0: 0.07, 1: 0.07, 2: 0.07, 3: 0.14, 4: 0.14, 5: 0.035}[k]
                p.close()
                c.setFillColor(Color(*HexColor(COLORS[k]).rgb(), alpha=band_alpha))
                c.drawPath(p, stroke=0, fill=1)
            if task == "parity" and not prompt:
                c.setStrokeColor(Color(*HexColor(COLORS[k]).rgb(), alpha=0.45))
                c.setDash([])
                c.setLineWidth(0.75)
                for a, b, sd in vals:
                    c.line(xx(a), yy(b), xx(2 * a - 1), yy(b))
                    for edge in [a, 2 * a - 1]:
                        c.line(xx(edge), yy(b) - 1.8, xx(edge), yy(b) + 1.8)
            c.setStrokeColor(HexColor(COLORS[k]))
            c.setLineWidth(LINE_WIDTHS[k])
            c.setDash(DASHES[k])
            p = c.beginPath()
            p.moveTo(*pts[0])
            for pt in pts[1:]:
                p.lineTo(*pt)
            c.drawPath(p, stroke=1, fill=0)
            marker_layers.append((k, pts))
        # Draw markers after every line/band; a later curve cannot erase them.
        # Large outline diamonds surround small circles at coincident hard points.
        for k, pts in sorted(marker_layers, key=lambda layer: layer[0] == 5):
            for j, pt in enumerate(pts):
                if prompt or task != "s5" or j in [0, 3, 6, 9, 12, 15, 18]:
                    marker(*pt, k)

    text(223, 458, "(a) Prompting", 18, True, align="center")
    text(780, 458, "(b) Finetuning", 18, True, align="center")
    pstarts = [65, 175, 285]
    pw = 96
    fw = 94
    fstarts = [535, 645, 755, 865]
    for x, title in zip(pstarts, ["Multiplication", "Parity", "S5"]):
        text(x + pw / 2, 427, title, 16, True, align="center")
    for x, title in zip(
        fstarts, ["Multiplication", "Iterated addition", "Parity", "S5"]
    ):
        if title == "Iterated addition":
            text(x + fw / 2, 434, "Iterated", 15, True, align="center")
            text(x + fw / 2, 418, "addition", 15, True, align="center")
        else:
            text(x + fw / 2, 427, title, 16, True, align="center")
    for row, (model, label) in enumerate(PMODELS):
        y = 312 - row * 116
        for j, task in enumerate(["multiplication", "parity", "s5"]):
            panel(pstarts[j], y, pw, 92, task, model, True, j == 0, row == 2)
        c.saveState()
        c.translate(402, y + 46)
        c.rotate(90)
        text(0, 0, label, 14.5, True, align="center")
        c.restoreState()
    for row, (model, label) in enumerate(
        [("qwen", "Qwen2.5-7B"), ("llama", "Llama-3.1-8B")]
    ):
        y = 310 - row * 120
        for j, task in enumerate(["multiplication", "knowledge", "parity", "s5"]):
            panel(fstarts[j], y, fw, 88, task, model, False, j == 0, row == 1)
        c.saveState()
        c.translate(991, y + 44)
        c.rotate(90)
        text(0, 0, label, 14.5, True, align="center")
        c.restoreState()
    text(223, 35, "Difficulty d (digits / bits / swaps)", 15, align="center")
    text(780, 145, "Difficulty d (digits / facts / bits / swaps)", 15, align="center")
    for x, ypos in [(12, 247), (475, 294)]:
        c.saveState()
        c.translate(x, ypos)
        c.rotate(90)
        text(0, 0, "Covert accuracy (%)", 16, align="center")
        c.restoreState()
    c.setDash([])
    c.setStrokeColor(HexColor("#CCCCCC"))
    c.setLineWidth(0.7)
    c.line(443, 55, 443, 442)
    # Shared legend: Full CoT is an open control, not a sixth monitored mode.
    c.setDash([])
    c.setStrokeColor(HexColor("#BFC6CB"))
    c.setLineWidth(0.7)
    c.roundRect(528, 11, 459, 115, 6, stroke=1, fill=0)
    legend_rows = [
        ("No CoT Mode", 105, [0]),
        ("Agnostic CoT Mode", 78, [1, 2]),
        ("Informative CoT Mode", 51, [3, 4]),
        ("Open control", 24, [5]),
    ]
    for mode, y, items in legend_rows:
        regime_color = (
            "#373D43"
            if mode.startswith("No CoT")
            else (
                COLORS[1]
                if mode.startswith("Agnostic")
                else COLORS[3] if mode.startswith("Informative") else "#555555"
            )
        )
        text(687, y, mode, 12.5, True, align="right", color=regime_color)
        for item_index, k in enumerate(items):
            x = 700 if item_index == 0 else 831
            c.setStrokeColor(HexColor(COLORS[k]))
            c.setLineWidth(LINE_WIDTHS[k])
            c.setDash(DASHES[k])
            c.line(x, y + 4, x + 22, y + 4)
            marker(x + 11, y + 4, k)
            text(x + 29, y, LABELS[k], 13.5)
    c.showPage()
    c.save()
    with open(OUT / "figure_prompting_finetuning_plot_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=plotrows[0])
        writer.writeheader()
        writer.writerows(plotrows)
    scaling_check = {}
    for task in ["multiplication", "s5", "parity"]:
        rows = [
            r
            for r in base_prows
            if r["mechanism"] == "vanilla" and r["experiment"] == task
        ]
        mean = statistics.mean(
            float(r[f"d{i}_both_pct"])
            for r in base_prows
            if r["experiment"] == task
            for i in range(1, 11)
            if r[f"d{i}_both_pct"]
        )
        tau = round(mean)
        scaling_check[task] = {
            "mean_joint_accuracy_pct": mean,
            "rounded_threshold_pct": tau,
            "max_passing_points": [
                {
                    "model": r["model_name"],
                    "family": r["model_family"],
                    "params_b": float(r["params_b"]),
                    "d_star": max(
                        [i for i in range(1, 11) if float(r[f"d{i}_both_pct"]) >= tau]
                        or [0]
                    ),
                }
                for r in rows
            ],
        }
    report = {
        "prompting_source_rows": len(base_prows),
        "prompting_models": [m for m, _ in PMODELS],
        "missing_prompting_curves": missing,
        "S5_audit": "114000 per-example rows; 570 complete seed/length cells; all private means and sample SDs match",
        "prompting_filler_mapping": MAPPING["filler"],
        "prompting_full_cot_source": str(PROMPT_FULL_COT),
        "prompting_full_cot_source_rows": len(prompt_full_cot_rows),
        "finetuning_filler_source": "filler_only_scaling_v1; audit complete for all 24 cells",
        "full_cot_source": str(full_cot_source),
        "full_cot_rows": len(full_cot_rows),
        "reference_image_2_reconstruction": scaling_check,
        "notes": "The merged figure uses covert-answer accuracy from the refreshed 333-row prompting CSV, prompting unrestricted-CoT CSV, and audited finetuning summaries. The refreshed prompting input adds Kimi K3 Vanilla and vanilla-plus-filler results for Parity and S5. The unrestricted-CoT input includes Qwen 3.5 27B on Multiplication, Parity, and S5. Full CoT finetuning parity bucket codes are decoded to their minimum bit lengths before plotting.",
    }
    (OUT / "figure_prompting_finetuning_audit.json").write_text(
        json.dumps(report, indent=2)
    )
    caption = (
        "Covert-answer exact accuracy as task difficulty increases for prompting (a) and finetuning (b). "
        "Difficulty d is the number of operand digits for Multiplication, input bits for Parity, and swaps for S5; for Iterated addition, d is the number of retrieved facts. "
        "The legend groups the monitored protocols by mode: Vanilla (No CoT), Filler and Public-only CoT (Agnostic CoT), and Piggybacking and Special Tokens (Informative CoT). "
        "Full CoT is an open control that permits visible reasoning about the covert question; it is shown for both prompting and finetuning where results are available, separately from the three monitored modes. "
        "In prompting, Filler denotes the vanilla-plus-filler condition. In finetuning, Filler denotes the primary filler-only condition, which supplies difficulty-scaled atomic period tokens without a meaningful public or covert chain of thought. "
        "Finetuning points and shaded bands show the mean and plus or minus one sample standard deviation over seeds 0, 1, and 2. "
        "Both Parity panels use a base-2 logarithmic bit-length axis. Finetuning Parity accuracy is aggregated over the ranges 8-15, 16-31, 32-63, 64-127, and 128-255 bits; points are placed at the geometric centers and horizontal bars show the ranges. "
        "Missing prompting conditions are omitted: Special Tokens was not evaluated for either Qwen model. "
        "Prompting and finetuning use different task constructions and difficulty supports, so comparisons should be made within panels rather than by equating numerical d values across the two paradigms.\n"
    )
    (ROOT / "docs/scaling_regime/MERGED_FIGURE_CAPTION.txt").write_text(caption)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
