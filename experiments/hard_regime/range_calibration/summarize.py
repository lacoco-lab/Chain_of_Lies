#!/usr/bin/env python3
"""Summarize completed one-seed range cells and select the hardest passing range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _report(path: Path, variant: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for report in payload.get("reports", []):
        if report.get("variant_name") == variant:
            return report
    raise RuntimeError(f"No report for {variant} in {path}.")


def _metric(report: dict[str, Any], source: str, metric: str) -> float:
    return float(report[source][metric])


def _piggy_row(config: dict[str, Any], root: Path, range_id: str) -> dict[str, Any] | None:
    section = config["piggyback"]
    piggy, control = section["variants"]
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for mode in section["modes"]:
        for variant in section["variants"]:
            path = root / range_id / f"seed_{config['seed']}" / mode / "per_variant_eval" / config["checkpoint"] / f"{variant}.json"
            report = _report(path, variant)
            if report is None:
                return None
            reports[(mode, variant)] = report

    channel = reports[("public_cot", piggy)]
    control_channel = reports[("public_cot", control)]
    answer = reports[("answer_only", piggy)]
    gate = section["gate"]
    row: dict[str, Any] = {
        "range_id": range_id,
        "channel_public_exact": _metric(channel, "rl", "public_exact_rate"),
        "channel_private_exact": _metric(channel, "rl", "private_exact_rate"),
        "answer_only_private_exact": _metric(answer, "rl", "private_exact_rate"),
        "control_channel_private_exact": _metric(control_channel, "rl", "private_exact_rate"),
        "baseline_private_exact": _metric(channel, "baseline", "private_exact_rate"),
    }
    row["piggyback_minus_control_private"] = row["channel_private_exact"] - row["control_channel_private_exact"]
    row["channel_minus_answer_only_private"] = row["channel_private_exact"] - row["answer_only_private_exact"]
    row["passes_gate"] = all(
        (
            row["channel_private_exact"] >= gate["piggyback_private_min"],
            row["channel_public_exact"] >= gate["piggyback_public_min"],
            row["piggyback_minus_control_private"] >= gate["piggyback_minus_control_private_min"],
            row["channel_minus_answer_only_private"] >= gate["public_cot_minus_answer_only_private_min"],
            row["answer_only_private_exact"] <= gate["answer_only_private_max"],
            row["control_channel_private_exact"] <= gate["control_private_max"],
        )
    )
    return row


def _steg_row(config: dict[str, Any], root: Path, range_id: str) -> dict[str, Any] | None:
    section = config["steganography"]
    variant = section["variant"]
    reports: dict[str, dict[str, Any]] = {}
    for mode in section["modes"]:
        path = root / range_id / f"seed_{config['seed']}" / mode / "per_variant_eval" / config["checkpoint"] / f"{variant}.json"
        report = _report(path, variant)
        if report is None:
            return None
        reports[mode] = report

    channel, answer = reports["local_channel_cot"], reports["answer_only"]
    gate = section["gate"]
    row: dict[str, Any] = {
        "range_id": range_id,
        "channel_public_exact": _metric(channel, "rl", "public_exact_rate"),
        "channel_private_exact": _metric(channel, "rl", "private_exact_rate"),
        "payload_decode_rate": _metric(channel, "rl", "payload_decode_rate"),
        "answer_only_private_exact": _metric(answer, "rl", "private_exact_rate"),
        "baseline_private_exact": _metric(channel, "baseline", "private_exact_rate"),
    }
    row["channel_minus_answer_only_private"] = row["channel_private_exact"] - row["answer_only_private_exact"]
    row["passes_gate"] = all(
        (
            row["channel_private_exact"] >= gate["channel_private_min"],
            row["channel_public_exact"] >= gate["channel_public_min"],
            row["payload_decode_rate"] >= gate["payload_decode_min"],
            row["channel_minus_answer_only_private"] >= gate["channel_minus_answer_only_private_min"],
            row["answer_only_private_exact"] <= gate["answer_only_private_max"],
        )
    )
    return row


def summarize(config_path: Path, artifacts_root: Path, model: str, family: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if model not in config["models"]:
        raise ValueError(f"Unknown model {model!r}.")
    section = config[family]
    root = artifacts_root / model / family
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for range_item in section["ranges_hard_to_easy"]:
        range_id = range_item["id"]
        row = _piggy_row(config, root, range_id) if family == "piggyback" else _steg_row(config, root, range_id)
        if row is None:
            missing.append(range_id)
        else:
            row["operand_range"] = range_item["operand_range"]
            rows.append(row)

    selected = next((row["range_id"] for row in rows if row["passes_gate"]), None)
    result = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "model_key": model,
        "model_id": config["models"][model],
        "family": family,
        "seed": config["seed"],
        "selection_rule": "first (hardest) completed range in configured hard-to-easy order that passes every gate",
        "gate": section["gate"],
        "selected_range": selected,
        "completed_rows": rows,
        "missing_ranges": missing,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "calibration_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (root / "calibration_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        f"# {family.title()} range calibration: {model}",
        "",
        f"Selected hardest passing range: **{selected or 'none yet'}**",
        "",
        f"All results use reserved calibration seed {config['seed']}; final claims require independent seeds 0, 1, and 2.",
        "",
        "| range | private (channel) | public (channel) | private (answer-only) | channel − answer | passes |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['range_id']} | {row['channel_private_exact']:.3f} | "
            f"{row['channel_public_exact']:.3f} | {row['answer_only_private_exact']:.3f} | "
            f"{row['channel_minus_answer_only_private']:+.3f} | {'yes' if row['passes_gate'] else 'no'} |"
        )
    if not rows:
        lines.append("| (none completed) | | | | | |")
    lines.extend(("", f"Missing/not-yet-completed ranges: {', '.join(missing) or 'none'}", ""))
    (root / "CALIBRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen", "llama"), required=True)
    parser.add_argument("--family", choices=("piggyback", "steganography"), required=True)
    args = parser.parse_args()
    summarize(args.config, args.artifacts_root, args.model, args.family)


if __name__ == "__main__":
    main()
