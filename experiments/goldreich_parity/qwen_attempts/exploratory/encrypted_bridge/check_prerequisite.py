#!/usr/bin/env python3
"""Stop the bridge DAG unless the rebuilt one-mask checkpoint reproduces its success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "artifacts/ce_parity_goldreich_bit_attention/bit_attention/metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/ce_parity_goldreich_encrypted_bridge/prerequisite_gate.json"
        ),
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = {
        "prerequisite": "one_mask_bit_attention",
        "best_validation_accuracy": metrics["best_validation_greedy_accuracy"],
        "threshold": metrics["gate_threshold"],
        "passes": metrics["passes_gate"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(
            "One-mask prerequisite failed; bridge GPU job intentionally stopped."
        )


if __name__ == "__main__":
    main()
