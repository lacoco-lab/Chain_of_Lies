#!/usr/bin/env python3
"""Summarize final full-trace accuracy, retention, privacy, and length gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _passes(auto: dict, local_accuracy: float, gate: dict) -> bool:
    return (
        auto["trace_bit_accuracy"] >= gate["trace_bit_accuracy"]
        and auto["selector_accuracy"] >= gate["selector_accuracy"]
        and auto["answer_accuracy"] >= gate["answer_accuracy"]
        and local_accuracy >= gate["local_bit_accuracy"]
        and auto["twin_trace_match"] >= gate["twin_trace_match"]
        and auto["seed_leak_rate"] <= gate["maximum_seed_leak_rate"]
    )


def summarize(config_path: Path, training_root: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads((training_root / "training_metrics.json").read_text(encoding="utf-8"))
    local_accuracy = metrics["final_local_retention"]["bit_accuracy"]
    rows, checks = [], {}
    for length in [*config["core_lengths"], *config["extension_lengths"]]:
        values = metrics["final_evaluations"][str(length)]
        auto, teacher = values["autoregressive"], values["teacher"]
        passed = _passes(auto, local_accuracy, config["final_gate"])
        checks[str(length)] = passed
        rows.append({
            "length": length, "n": auto["n"], "teacher_bit_accuracy": teacher["bit_accuracy"],
            "trace_bit_accuracy": auto["trace_bit_accuracy"],
            "trace_exact": auto["trace_exact"], "selector_accuracy": auto["selector_accuracy"],
            "answer_accuracy": auto["answer_accuracy"], "full_exact": auto["full_exact"],
            "parse_rate": auto["parse_rate"], "twin_trace_match": auto["twin_trace_match"],
            "seed_leak_rate": auto["seed_leak_rate"], "passes_gate": passed,
        })
    core_pass = all(checks[str(length)] for length in config["core_lengths"])
    extension_pass = all(checks[str(length)] for length in config["extension_lengths"])
    first_failure = next((int(length) for length, passed in checks.items() if not passed), None)
    result = {
        "schema_version": 1, "experiment_name": config["experiment_name"],
        "model": config["model"], "seed": config["seed"],
        "initial_adapter": metrics["initial_adapter"],
        "baseline_local_bit_accuracy": metrics["baseline_local"]["bit_accuracy"],
        "final_local_bit_accuracy": local_accuracy,
        "length_checks": checks, "passes_core": core_pass,
        "passes_64_extension": extension_pass, "first_failed_length": first_failure,
        "elapsed_seconds": metrics["elapsed_seconds"], "metrics": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Attention-guided full Goldreich encrypted CoT", "",
        f"Core lengths: **{'PASS' if core_pass else 'FAIL'}**", "",
        f"Length 64: **{'PASS' if extension_pass else 'FAIL'}**", "",
        f"One-bit retention: {metrics['baseline_local']['bit_accuracy']:.3f} before full training, "
        f"{local_accuracy:.3f} after full training.", "",
        "| Length | N | Teacher bits | Generated bits | Trace exact | Selectors | Answer | Full exact | Twin match | Leak | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['length']} | {row['n']} | {row['teacher_bit_accuracy']:.3f} | "
            f"{row['trace_bit_accuracy']:.3f} | {row['trace_exact']:.3f} | "
            f"{row['selector_accuracy']:.3f} | {row['answer_accuracy']:.3f} | "
            f"{row['full_exact']:.3f} | {row['twin_trace_match']:.3f} | "
            f"{row['seed_leak_rate']:.3f} | {'pass' if row['passes_gate'] else 'fail'} |"
        )
    if first_failure is None:
        lines.extend(["", "All predeclared mechanism and privacy gates passed."])
    else:
        lines.extend(["", f"The first failed final length was {first_failure}."])
    lines.append("")
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--training-root", type=Path,
                        default=Path("artifacts/ce_parity_goldreich_full_attention/training"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("artifacts/ce_parity_goldreich_full_attention"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.config, args.training_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
