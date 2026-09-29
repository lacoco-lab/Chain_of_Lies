#!/usr/bin/env python3
"""Gate the established mechanism and the newly learned position compositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("mechanism", "positions"), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = config["success_gate"]
    if args.kind == "mechanism":
        metrics_path = Path(
            "artifacts/ce_parity_goldreich_update_ablation/base/metrics.json"
        )
        adapter = Path(
            "artifacts/ce_parity_goldreich_update_ablation/base/best_adapter"
        )
        if not metrics_path.exists():
            result = {
                "kind": args.kind,
                "checks": {"metrics": False, "adapter_weights": False},
                "passes": False,
                "error": f"Missing prerequisite metrics: {metrics_path}",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            raise SystemExit("mechanism gate failed; downstream training was stopped")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final = metrics["final_evaluations"]
        checks = {
            "local": final["local"]["greedy_accuracy"]
            >= gate["minimum_local_retention"],
            "supplied_update": final["supplied_update"]["greedy_accuracy"]
            >= gate["minimum_supplied_update_retention"],
            "adapter_weights": (adapter / "adapter_model.safetensors").exists(),
        }
        values = {
            "local_accuracy": final["local"]["greedy_accuracy"],
            "supplied_update_accuracy": final["supplied_update"]["greedy_accuracy"],
        }
    else:
        metrics_path = Path(
            "artifacts/ce_parity_goldreich_final/positions/metrics.json"
        )
        adapter = Path("artifacts/ce_parity_goldreich_final/positions/best_adapter")
        if not metrics_path.exists():
            result = {
                "kind": args.kind,
                "checks": {"metrics": False, "adapter_weights": False},
                "passes": False,
                "error": f"Positions training did not produce metrics: {metrics_path}",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            raise SystemExit("positions gate failed; downstream training was stopped")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final = metrics["final_evaluations"]
        checks = {
            "derive_previous": final["derive_previous"]["greedy_accuracy"]
            >= gate["minimum_composition_accuracy"],
            "derive_current": final["derive_current"]["greedy_accuracy"]
            >= gate["minimum_composition_accuracy"],
            "local": final["local"]["greedy_accuracy"]
            >= gate["minimum_local_retention"],
            "supplied_update": final["supplied_update"]["greedy_accuracy"]
            >= gate["minimum_supplied_update_retention"],
            "adapter_weights": (adapter / "adapter_model.safetensors").exists(),
        }
        values = {f"{task}_accuracy": final[task]["greedy_accuracy"] for task in final}
    result = {
        "kind": args.kind,
        **values,
        "checks": checks,
        "passes": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(f"{args.kind} gate failed; downstream training was stopped")


if __name__ == "__main__":
    main()
