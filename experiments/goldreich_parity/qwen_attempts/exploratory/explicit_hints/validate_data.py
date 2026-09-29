#!/usr/bin/env python3
"""Validate explicit-hint formulas, selectors, privacy formatting, and split isolation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from generate_data import (
    encrypted,
    fixed_graph,
    mask_bit,
    masks,
    parity,
    predicate,
    selector,
    states,
)

SEED_VALUE_RE = re.compile(r"\b[A-P]\s*[:=]?\s*[01]\b")


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
        int(config["seed_bits"]), int(config["max_steps"]), int(config["graph_seed"])
    )
    if graph != manifest["fixed_graph"]:
        raise RuntimeError("Stored graph differs from configured graph.")
    lengths = [
        *map(int, config["core_lengths"]),
        *map(int, config["extension_lengths"]),
    ]
    tasks = [
        "predicate_local",
        "hinted_mask_local",
        "hinted_start_local",
        "hinted_update_local",
    ] + [f"hinted_joint_{length}" for length in lengths]
    split_seeds: dict[str, set[str]] = {"base_train": set(), "validation": set()}
    checked = 0
    for split in ("base_train", "validation"):
        for task in tasks:
            rows = _load(data_root / split / task)
            if task.startswith("hinted_joint_"):
                expected_n = int(
                    config["trace_train_n"]
                    if split == "base_train"
                    else config["trace_validation_n"]
                )
            else:
                expected_n = int(
                    config["local_train_n"]
                    if split == "base_train"
                    else config["local_validation_n"]
                )
            if len(rows) != expected_n:
                raise RuntimeError(
                    f"Wrong count for {split}/{task}: {len(rows)} != {expected_n}"
                )
            twin_groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                spec, target = row["spec"], str(row["supervised_suffix"])
                if task == "predicate_local":
                    gold = predicate(map(int, spec["inputs"]))
                    if gold != int(spec["gold_bit"]):
                        raise RuntimeError("Invalid local predicate target.")
                elif task == "hinted_mask_local":
                    seed, edge = str(spec["prg_seed"]), list(map(int, spec["edge"]))
                    if selector(edge) != spec["selector"] or mask_bit(
                        seed, edge
                    ) != int(spec["gold_bit"]):
                        raise RuntimeError("Invalid hinted-mask target.")
                    split_seeds[split].add(seed)
                elif task == "hinted_update_local":
                    seed = str(spec["prg_seed"])
                    z_previous = mask_bit(seed, list(map(int, spec["previous_edge"])))
                    z_current = mask_bit(seed, list(map(int, spec["current_edge"])))
                    left, right = map(int, spec["input_pair"])
                    gold = (
                        int(spec["c_previous"]) ^ z_previous ^ left ^ right ^ z_current
                    )
                    if gold != int(spec["gold_bit"]):
                        raise RuntimeError("Invalid hinted-update target.")
                    split_seeds[split].add(seed)
                elif task == "hinted_start_local":
                    seed, edge = str(spec["prg_seed"]), list(map(int, spec["edge"]))
                    left, right = map(int, spec["input_pair"])
                    gold = left ^ right ^ mask_bit(seed, edge)
                    if selector(edge) != spec["selector"] or gold != int(
                        spec["gold_bit"]
                    ):
                        raise RuntimeError("Invalid hinted-start target.")
                    split_seeds[split].add(seed)
                else:
                    seed, bits, length = (
                        str(spec["prg_seed"]),
                        str(spec["input_bits"]),
                        int(spec["length"]),
                    )
                    expected_selectors = [selector(edge) for edge in graph[:length]]
                    expected_masks = masks(seed, graph, length)
                    if (
                        spec["selectors"] != expected_selectors
                        or spec["gold_masks"] != expected_masks
                    ):
                        raise RuntimeError(
                            "Invalid explicit selector or mask sequence."
                        )
                    if spec["plain_states"] != states(bits) or spec[
                        "gold_sequence"
                    ] != encrypted(bits, expected_masks):
                        raise RuntimeError("Invalid encrypted trace.")
                    if int(spec["final_parity"]) != parity(bits):
                        raise RuntimeError("Invalid parity target.")
                    if (
                        SEED_VALUE_RE.search(target)
                        or "PRIVATE" in target
                        or "Mask:" in target
                    ):
                        raise RuntimeError(
                            "Public CoT target leaks a seed value or mask."
                        )
                    for hint, bit in zip(expected_selectors, spec["gold_sequence"]):
                        if f"{hint} {bit}" not in target:
                            raise RuntimeError(
                                "Public CoT is missing an explicit selector/state line."
                            )
                    split_seeds[split].add(seed)
                    twin_groups.setdefault(str(spec["twin_id"]), []).append(row)
                checked += 1
            if task.startswith("hinted_joint_"):
                for twins in twin_groups.values():
                    if len(twins) != 2:
                        raise RuntimeError("Incomplete complementary twin pair.")
                    left, right = twins
                    if left["spec"]["gold_sequence"] != right["spec"]["gold_sequence"]:
                        raise RuntimeError(
                            "Complementary twins do not share an encrypted trace."
                        )
    overlap = split_seeds["base_train"] & split_seeds["validation"]
    if overlap:
        raise RuntimeError(f"Training and validation seeds overlap ({len(overlap)}).")
    for track, key in (("core", "core_stages"), ("extension", "extension_stages")):
        for stage in config[key]:
            rows = _load(data_root / "curriculum" / track / stage["name"])
            if len(rows) != int(config["curriculum_train_n"]):
                raise RuntimeError(
                    f"Wrong curriculum count for {track}/{stage['name']}"
                )
    report = {
        "schema_version": 1,
        "checked_base_records": checked,
        "train_validation_seed_overlap": 0,
        "fixed_graph": True,
        "targets_valid": True,
        "public_targets_hide_seed_values_and_masks": True,
        "complementary_twins_valid": True,
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
        default=Path("generated_data/parity_goldreich_explicit_hints/seed_0"),
    )
    args = parser.parse_args()
    validate(args.config, args.data_root)


if __name__ == "__main__":
    main()
