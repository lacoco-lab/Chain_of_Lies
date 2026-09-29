#!/usr/bin/env python3
"""Require the successful one-mask plus supplied-update mechanism checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/ce_parity_goldreich_composition_rank/prerequisite_gate.json"
        ),
    )
    args = parser.parse_args()
    metrics_path = Path(
        "artifacts/ce_parity_goldreich_update_ablation/base/metrics.json"
    )
    adapter = Path("artifacts/ce_parity_goldreich_update_ablation/base/best_adapter")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    local = metrics["final_evaluations"]["local"]["greedy_accuracy"]
    supplied = metrics["final_evaluations"]["supplied_update"]["greedy_accuracy"]
    result = {
        "prerequisite": "merged_one_mask_and_supplied_update",
        "local_accuracy": local,
        "supplied_update_accuracy": supplied,
        "minimum_local_accuracy": 0.95,
        "minimum_supplied_update_accuracy": 0.98,
        "adapter_config_present": (adapter / "adapter_config.json").exists(),
        "adapter_weights_present": (adapter / "adapter_model.safetensors").exists(),
    }
    result["passes"] = bool(
        local >= result["minimum_local_accuracy"]
        and supplied >= result["minimum_supplied_update_accuracy"]
        and result["adapter_config_present"]
        and result["adapter_weights_present"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit("Mechanism prerequisite failed; rank jobs were stopped")


if __name__ == "__main__":
    main()
