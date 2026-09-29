#!/usr/bin/env python3
"""Validate the final curriculum's balance, formulas, privacy, and partitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_data import RANK32_CALIBRATION_DIGESTS, fixed_graph, mask_bit


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate(config_path: Path, data_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not manifest["all_seed_partitions_disjoint"]
        or not manifest["paired_task_seeds"]
    ):
        raise ValueError("Invalid seed partition manifest")
    calibration = manifest.get("rank32_calibration_recipe", {})
    if bool(config.get("enforce_rank32_calibration", True)):
        if not calibration.get("enforced") or not calibration.get("digests_verified"):
            raise ValueError("Rank-32 calibration lock was not enforced")
        if calibration.get("digests") != RANK32_CALIBRATION_DIGESTS:
            raise ValueError(
                "Rank-32 calibration digests do not match the successful recipe"
            )
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    report = {}
    seed_sets = {}
    for name, expected_n in manifest["dataset_sizes"].items():
        rows = read_jsonl(data_root / f"{name}.jsonl")
        if len(rows) != expected_n:
            raise ValueError(f"Wrong size for {name}: {len(rows)} != {expected_n}")
        counts = Counter(int(row["supervised_suffix"]) for row in rows)
        task_name = name.rsplit("_", 1)[0]
        if task_name in {"mask_delta", "full_zero"}:
            # These labels are the natural XOR of two PRG outputs and therefore
            # cannot be forced without changing the private seed distribution.
            if set(counts) != {0, 1} or abs(counts[0] - counts[1]) > expected_n * 0.10:
                raise ValueError(
                    f"Severely imbalanced natural targets for {name}: {counts}"
                )
        elif counts != {0: expected_n // 2, 1: expected_n // 2}:
            raise ValueError(f"Unbalanced targets for {name}: {counts}")
        seeds = set()
        for row in rows:
            if row["supervised_suffix"] not in {"0", "1"}:
                raise ValueError(f"Non-binary target in {row['experiment_id']}")
            spec = row["spec"]
            task = spec["task"]
            if task == "local_mask":
                if spec["gold_bit"] != mask_bit(
                    spec["prg_seed"], graph[spec["edge_index"]]
                ):
                    raise ValueError(f"Wrong local mask in {row['experiment_id']}")
                continue
            if task == "supplied_update":
                previous_z, current_z = spec["previous_mask"], spec["current_mask"]
            else:
                seeds.add(spec["prg_seed"])
                derived_previous = mask_bit(spec["prg_seed"], spec["previous_edge"])
                derived_current = mask_bit(spec["prg_seed"], spec["current_edge"])
                both_derived = {
                    "mask_delta",
                    "full_zero",
                    "full_one",
                    "full_two",
                    "full_update",
                }
                if (
                    task in {"derive_previous", *both_derived}
                    and derived_previous != spec["previous_mask"]
                ):
                    raise ValueError(f"Wrong previous mask in {row['experiment_id']}")
                if (
                    task in {"derive_current", *both_derived}
                    and derived_current != spec["current_mask"]
                ):
                    raise ValueError(f"Wrong current mask in {row['experiment_id']}")
                expected_groups = 2 if task in both_derived else 1
                if len(spec["attention_groups"]) != expected_groups:
                    raise ValueError(
                        f"Wrong attention groups in {row['experiment_id']}"
                    )
                previous_z, current_z = spec["previous_mask"], spec["current_mask"]
            if task == "mask_delta":
                expected = previous_z ^ current_z
                if expected != spec["gold_bit"]:
                    raise ValueError(
                        f"Wrong mask-delta target in {row['experiment_id']}"
                    )
                continue
            left, right = spec["input_pair"]
            if task == "full_zero" and (spec["previous_c"] or left or right):
                raise ValueError(
                    f"Nonzero public term in full_zero: {row['experiment_id']}"
                )
            if task == "full_one" and (left or right):
                raise ValueError(
                    f"Extra public term in full_one: {row['experiment_id']}"
                )
            if task == "full_two" and right:
                raise ValueError(
                    f"Extra public term in full_two: {row['experiment_id']}"
                )
            expected = spec["previous_c"] ^ previous_z ^ left ^ right ^ current_z
            if expected != spec["gold_bit"]:
                raise ValueError(f"Wrong update target in {row['experiment_id']}")
        if seeds:
            seed_sets[name] = seeds
        report[name] = {"n": len(rows), "zeros": counts[0], "ones": counts[1]}

    for split in ("train", "validation"):
        names = [
            f"{task}_{split}"
            for task in (
                "derive_previous",
                "derive_current",
                "mask_delta",
                "full_zero",
                "full_one",
                "full_two",
                "full_update",
            )
        ]
        if not all(seed_sets[name] == seed_sets[names[0]] for name in names[1:]):
            raise ValueError(f"Composition tasks do not use paired {split} seeds")
    if seed_sets["derive_previous_train"] & seed_sets["derive_previous_validation"]:
        raise ValueError("Composition train/validation seeds overlap")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_final/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.data_root), indent=2))


if __name__ == "__main__":
    main()
