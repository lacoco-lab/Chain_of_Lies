#!/usr/bin/env python3
"""Validate curriculum counts, targets, formulas, and held-out PRG seeds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from generate_data import encrypted, fixed_graph, masks, parity, predicate, states


def _load(directory: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    if not rows:
        raise ValueError(f"No records in {directory}")
    return rows


def validate(config_path: Path, data_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    graph = fixed_graph(
        int(config["seed_bits"]),
        int(config["max_mask_bits"]),
        int(config["graph_seed"]),
    )
    if graph != manifest["fixed_graph"]:
        raise RuntimeError("Stored fixed graph does not match the configured graph.")

    lengths = [int(value) for value in config["lengths"]]
    expected_tasks = ["local_update", "predicate_local"] + [
        f"{kind}_{length}"
        for length in lengths
        for kind in ("mask", "supplied", "joint")
    ]
    train_seeds: set[str] = set()
    validation_seeds: set[str] = set()
    checked = 0
    for split, expected_n in (
        ("base_train", int(config["base_train_n"])),
        ("validation", int(config["base_validation_n"])),
    ):
        for task in expected_tasks:
            rows = _load(data_root / split / task)
            if task in {"local_update", "predicate_local"}:
                expected = int(
                    config["local_train_n"]
                    if split == "base_train"
                    else config["local_validation_n"]
                )
            else:
                expected = expected_n
            if len(rows) != expected:
                raise RuntimeError(
                    f"Expected {expected} rows for {split}/{task}; found {len(rows)}"
                )
            for row in rows:
                spec = row["spec"]
                target = str(row["supervised_suffix"])
                if task == "local_update":
                    c_prev, z_prev, left, right, z_current = map(int, spec["inputs"])
                    gold = c_prev ^ z_prev ^ left ^ right ^ z_current
                    if (
                        int(spec["gold_bit"]) != gold
                        or f"Encrypted state: {gold}" not in target
                    ):
                        raise RuntimeError("Invalid local encrypted-update target.")
                elif task == "predicate_local":
                    gold = predicate(map(int, spec["inputs"]))
                    if int(spec["gold_bit"]) != gold or f"Mask: {gold}" not in target:
                        raise RuntimeError("Invalid local predicate target.")
                elif task.startswith("mask_"):
                    seed = str(spec["prg_seed"])
                    length = int(spec["length"])
                    gold_sequence = masks(seed, graph, length)
                    if spec["gold_sequence"] != gold_sequence:
                        raise RuntimeError("Invalid mask-generation target.")
                    (train_seeds if split == "base_train" else validation_seeds).add(
                        seed
                    )
                else:
                    bits = str(spec["input_bits"])
                    seed = str(spec["prg_seed"])
                    length = int(spec["length"])
                    gold_masks = masks(seed, graph, length)
                    if spec["gold_masks"] != gold_masks:
                        raise RuntimeError("Invalid trace masks.")
                    if spec["plain_states"] != states(bits):
                        raise RuntimeError("Invalid plaintext states.")
                    if spec["gold_sequence"] != encrypted(bits, gold_masks):
                        raise RuntimeError("Invalid encrypted trace.")
                    if int(spec["final_parity"]) != parity(bits):
                        raise RuntimeError("Invalid final parity.")
                    (train_seeds if split == "base_train" else validation_seeds).add(
                        seed
                    )
                if not re.search(r"</(?:BIT|MASK|ANSWER)>\s*$", target):
                    raise RuntimeError(f"Malformed target for {row['experiment_id']}")
                checked += 1

    if train_seeds & validation_seeds:
        raise RuntimeError("Training and validation PRG seeds overlap.")
    for split in ("base_train", "validation"):
        for length in lengths:
            supplied = _load(data_root / split / f"supplied_{length}")
            joint = _load(data_root / split / f"joint_{length}")
            supplied_items = sorted(
                (row["spec"]["input_bits"], row["spec"]["prg_seed"]) for row in supplied
            )
            joint_items = sorted(
                (row["spec"]["input_bits"], row["spec"]["prg_seed"]) for row in joint
            )
            if supplied_items != joint_items:
                raise RuntimeError(
                    f"Supplied and joint items differ for {split}, length {length}."
                )
    for track, stages_key in (
        ("update", "update_stages"),
        ("goldreich", "goldreich_stages"),
    ):
        for stage in config[stages_key]:
            rows = _load(data_root / "curriculum" / track / stage["name"])
            if len(rows) != int(config["curriculum_train_n"]):
                raise RuntimeError(f"Wrong curriculum size for {track}/{stage['name']}")

    report = {
        "schema_version": 1,
        "checked_base_records": checked,
        "train_validation_seed_overlap": 0,
        "fixed_graph": True,
        "targets_valid": True,
        "curricula_valid": True,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_curriculum/seed_0"),
    )
    args = parser.parse_args()
    validate(args.config, args.data_root)


if __name__ == "__main__":
    main()
