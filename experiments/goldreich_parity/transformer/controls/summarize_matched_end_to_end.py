#!/usr/bin/env python3
"""Validate and package the matched end-to-end N=16/N=32 control comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CONDITIONS = ("no_cot", "filler_cot")
COMMON_ARCHITECTURE_FIELDS = (
    "architecture",
    "heads",
    "model_dim",
    "head_dim",
    "token_embedding_dim",
    "ff_dim",
    "attention",
    "layer_norm",
    "nonlinear_blocks_after_attention",
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_95(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return center - radius, center + radius


def validate_end_to_end(row: dict[str, Any], condition: str, length: int) -> None:
    assert row["condition"] == condition
    assert int(row["length"]) == length
    assert int(row["seed"]) == 0
    assert row["data_fields_loaded"] == ["id", "input_bits", "answer"]
    assert row["private_seed_given_to_model"] is False
    for stage in ("routing_training", "representation_training", "public_training"):
        assert row[stage]["enabled"] is False
    assert int(row["integration_training"]["steps"]) == 3000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("artifacts/parity_standard_cot_controls_matched_end_to_end"),
    )
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=Path("artifacts/parity_standard_cot_controls"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    matched: dict[tuple[str, int], dict[str, Any]] = {}
    for condition in CONDITIONS:
        for length, root in ((16, args.new_root), (32, args.existing_root)):
            path = root / condition / f"n{length}" / "metrics.json"
            result = load(path)
            validate_end_to_end(result, condition, length)
            matched[condition, length] = result
            test = result["test_softmax_attention"]
            n = int(test["n"])
            accuracy = float(test["final_parity_accuracy"])
            successes = round(accuracy * n)
            low, high = wilson_95(successes, n)
            rows.append(
                {
                    "condition": condition,
                    "N": length,
                    "training_regime": "end_to_end_only",
                    "seed": 0,
                    "train_examples": int(result["train_examples"]),
                    "test_examples": n,
                    "final_parity_accuracy": accuracy,
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                    "filler_exact": test["cot_exact"],
                    "integration_steps": int(result["integration_training"]["steps"]),
                    "elapsed_seconds": float(result["elapsed_seconds"]),
                    "metrics_path": str(path),
                    "metrics_sha256": sha256(path),
                }
            )

        arch16 = matched[condition, 16]["architecture"]
        arch32 = matched[condition, 32]["architecture"]
        for field in COMMON_ARCHITECTURE_FIELDS:
            assert arch16[field] == arch32[field], (condition, field)

    csv_path = args.output_root / "matched_end_to_end_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "question": (
            "Do the N=16 versus N=32 No-CoT and fixed-filler results show a length "
            "transition when the training procedure is held fixed?"
        ),
        "design": {
            "seed": 0,
            "conditions": list(CONDITIONS),
            "lengths": [16, 32],
            "training": "End-to-end output supervision only",
            "integration_steps": 3000,
            "batch_size": 256,
            "integration_learning_rate": 0.0003,
            "staged_auxiliary_supervision": False,
            "note": (
                "N=16 uses its exhaustive finite-domain split and N=32 uses the existing "
                "disjoint sampled split; N=32 has more training examples."
            ),
        },
        "validated_common_architecture_fields": list(COMMON_ARCHITECTURE_FIELDS),
        "results": rows,
        "conclusion": (
            "Under the matched end-to-end recipe, No CoT is approximately 50% at both "
            "lengths. Fixed filler is 52.23% at N=16 and approximately 50% at N=32 while "
            "emitting its fixed filler exactly. The earlier near-perfect N=16 versus "
            "chance N=32 contrast therefore does not establish a pure length threshold; "
            "the staged supervision used at N=16 materially changes the comparison."
        ),
    }
    (args.output_root / "matched_end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    by_key = {(row["condition"], row["N"]): row for row in rows}
    readme = f"""# Matched end-to-end control at N=16 and N=32

## Purpose

The original control curve changed training procedures at N=32: N=16 received
direct-route, routed-value, public truth-table, and integration supervision,
whereas N=32 received end-to-end output supervision. This follow-up holds the
training recipe fixed by training both No CoT and fixed filler at N=16 with the
same end-to-end-only recipe already used at N=32.

## Fixed recipe

- training seed: 0;
- one-layer, 16-head, width-128 standard causal Transformer;
- 3,000 integration updates, batch size 256, learning rate 0.0003;
- no routing, representation, or public truth-table stage;
- inputs restricted to ID, input bits, and final PARITY;
- ordinary softmax attention at training and inference.

The N=16 run uses the exhaustive disjoint finite-domain split (32,768 train and
16,384 test examples). The existing N=32 run uses 131,072 train and 8,192 test
examples. This difference favors N=32 in training-data quantity.

## Results

| Condition | N | Final PARITY | 95% Wilson interval | Filler exact |
|:--|--:|--:|--:|--:|
| No CoT | 16 | {by_key['no_cot', 16]['final_parity_accuracy']:.4f} | [{by_key['no_cot', 16]['wilson_95_low']:.4f}, {by_key['no_cot', 16]['wilson_95_high']:.4f}] | -- |
| No CoT | 32 | {by_key['no_cot', 32]['final_parity_accuracy']:.4f} | [{by_key['no_cot', 32]['wilson_95_low']:.4f}, {by_key['no_cot', 32]['wilson_95_high']:.4f}] | -- |
| Fixed filler | 16 | {by_key['filler_cot', 16]['final_parity_accuracy']:.4f} | [{by_key['filler_cot', 16]['wilson_95_low']:.4f}, {by_key['filler_cot', 16]['wilson_95_high']:.4f}] | {by_key['filler_cot', 16]['filler_exact']:.4f} |
| Fixed filler | 32 | {by_key['filler_cot', 32]['final_parity_accuracy']:.4f} | [{by_key['filler_cot', 32]['wilson_95_low']:.4f}, {by_key['filler_cot', 32]['wilson_95_high']:.4f}] | {by_key['filler_cot', 32]['filler_exact']:.4f} |

## Paper-facing interpretation

When trained with the same end-to-end-only procedure, the No-CoT control is at
approximately 50% at both N=16 and N=32. Fixed filler reaches 52.23% at N=16
and approximately 50% at N=32, despite emitting every filler token correctly.
Accordingly, the earlier near-perfect N=16 versus chance N=32 contrast does not
by itself identify a pure length threshold: staged direct-route and value
supervision at N=16 materially affects the result. The supported conclusion is
narrower: fixed filler does not supply a useful serial state under the matched
end-to-end recipe, and the strong N=16 direct-control result relies on the
mechanistically valid staged supervision available at that length.

This is a single-training-seed diagnostic, matching the rest of the Goldreich
experiment. The 52.23% filler score indicates a small residual signal and
should be reported numerically rather than described as exactly chance.
"""
    (args.output_root / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "summary": str(args.output_root / "matched_end_to_end_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
