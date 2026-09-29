#!/usr/bin/env python3
"""Write a compact report for the full encrypted trajectory experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(config_path: Path, artifacts_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads((artifacts_root / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for length in sorted(map(int, metrics["final_by_length"])):
        item = metrics["final_by_length"][str(length)]
        rows.append(
            {
                "input_length": length,
                "n": item["n"],
                "trace_bit_accuracy": item["trace_bit_accuracy"],
                "trace_exact": item["trace_exact"],
                "final_parity_accuracy": item["final_parity_accuracy"],
                "joint_exact": item["joint_exact"],
                "strict_format": item["strict_format"],
            }
        )
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "passes_all_gates": bool(metrics["passes_gate"]),
        "metrics": rows,
        "prerequisite_retention": metrics["final_prerequisites"],
        "transition_retention": metrics["final_transitions"],
        "elapsed_seconds": metrics["elapsed_seconds"],
    }
    (artifacts_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (artifacts_root / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Full Goldreich-encrypted PARITY trajectory",
        "",
        f"Overall: **{'PASS' if result['passes_all_gates'] else 'FAIL'}**",
        "",
        "| Input bits | N | State-bit accuracy | Trace exact | Final parity | Joint exact | Format |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['input_length']} | {row['n']} | {row['trace_bit_accuracy']:.3f} | "
            f"{row['trace_exact']:.3f} | {row['final_parity_accuracy']:.3f} | "
            f"{row['joint_exact']:.3f} | {row['strict_format']:.3f} |"
        )
    lines.append("")
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_trajectory"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.config, args.artifacts_root), indent=2))


if __name__ == "__main__":
    main()
