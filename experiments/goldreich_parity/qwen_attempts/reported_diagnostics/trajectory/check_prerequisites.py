#!/usr/bin/env python3
"""Check that the successful single-step stack is present before trajectory training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(output: Path) -> dict:
    roots = {
        "mechanism": Path(
            "artifacts/ce_parity_goldreich_update_ablation/base/best_adapter"
        ),
        "positions": Path("artifacts/ce_parity_goldreich_final/positions/best_adapter"),
        "full": Path("artifacts/ce_parity_goldreich_final/full/best_adapter"),
    }
    metrics_path = Path("artifacts/ce_parity_goldreich_final/full/metrics.json")
    checks = {
        f"{name}_adapter": all(
            (root / filename).is_file()
            for filename in ("adapter_config.json", "adapter_model.safetensors")
        )
        for name, root in roots.items()
    }
    checks["full_metrics"] = metrics_path.is_file()
    full_accuracy = None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        full_accuracy = metrics["final_evaluations"]["full_update"]["greedy_accuracy"]
        checks["full_gate"] = bool(metrics["passes_gate"] and full_accuracy >= 0.95)
    else:
        checks["full_gate"] = False
    result = {
        "checks": checks,
        "full_update_accuracy": full_accuracy,
        "passes": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_trajectory_prerequisite_gate.json"),
    )
    args = parser.parse_args()
    result = check(args.output)
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit("Trajectory prerequisites failed")


if __name__ == "__main__":
    main()
