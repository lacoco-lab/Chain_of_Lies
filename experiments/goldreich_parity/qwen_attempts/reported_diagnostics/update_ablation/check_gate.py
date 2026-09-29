#!/usr/bin/env python3
"""Gate the known one-mask prerequisite or the newly trained supplied-update base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("prerequisite", "base"), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.kind == "prerequisite":
        metrics_path = Path(
            "artifacts/ce_parity_goldreich_bit_attention/bit_attention/metrics.json"
        )
        adapter = Path(
            "artifacts/ce_parity_goldreich_bit_attention/bit_attention/best_adapter"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result = {
            "kind": args.kind,
            "accuracy": metrics["best_validation_greedy_accuracy"],
            "threshold": metrics["gate_threshold"],
            "adapter_weights_present": (adapter / "adapter_model.safetensors").exists(),
        }
        result["passes"] = bool(
            metrics["passes_gate"] and result["adapter_weights_present"]
        )
    else:
        metrics_path = Path(
            "artifacts/ce_parity_goldreich_update_ablation/base/metrics.json"
        )
        adapter = Path(
            "artifacts/ce_parity_goldreich_update_ablation/base/best_adapter"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        supplied = metrics["final_evaluations"]["supplied_update"]["greedy_accuracy"]
        local = metrics["final_evaluations"]["local"]["greedy_accuracy"]
        result = {
            "kind": args.kind,
            "supplied_update_accuracy": supplied,
            "local_retention_accuracy": local,
            "supplied_threshold": config["success_gate"][
                "minimum_supplied_update_accuracy"
            ],
            "local_threshold": config["success_gate"]["minimum_local_retention"],
            "adapter_weights_present": (adapter / "adapter_model.safetensors").exists(),
        }
        result["passes"] = bool(
            supplied >= result["supplied_threshold"]
            and local >= result["local_threshold"]
            and result["adapter_weights_present"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(f"{args.kind} gate failed; downstream jobs were stopped")


if __name__ == "__main__":
    main()
