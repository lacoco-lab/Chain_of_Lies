#!/usr/bin/env python3
"""Validate Goldreich parity splits, fixed graph, targets, and held-out PRG seeds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from generate_splits import (  # noqa: E402
    complement,
    encrypted_trace,
    final_parity,
    fixed_hypergraph,
    goldreich_masks,
    pair_cumulative_trace,
    supervised_suffix,
)
from chain_of_lies.training.shared.trainer_utils import (  # noqa: E402
    _canonical_public_cot_suffix,
)


def _load(directory: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]
    if not records:
        raise ValueError(f"No prompt records in {directory}")
    return records


def validate(config_path: Path, split_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed_root = split_root / f"seed_{config['seed']}"
    graph = fixed_hypergraph(
        seed_bits=int(config["seed_bits"]),
        mask_bits=int(config["mask_bits"]),
        graph_seed=int(config["graph_seed"]),
    )
    condition_map = {str(item["name"]): str(item["trace_protocol"]) for item in config["conditions"]}
    items_by_condition_split: dict[tuple[str, str], list[tuple[str, str]]] = {}
    seeds_by_split: dict[str, set[str]] = {}
    twin_groups_checked = 0

    for condition, protocol in condition_map.items():
        for split, expected_n in (
            ("train", int(config["train_n"])),
            ("val", int(config["validation_n"])),
        ):
            records = _load(seed_root / condition / f"{split}_prompts")
            if len(records) != expected_n:
                raise RuntimeError(
                    f"Expected {expected_n} {condition}/{split} records, found {len(records)}"
                )
            items: list[tuple[str, str]] = []
            twins: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                spec = record["spec"]
                bits = str(spec["input_bits"])
                prg_seed = str(spec["prg_seed"])
                items.append((bits, prg_seed))
                twins[str(spec["twin_id"])].append(record)
                if record.get("task_type") != "parity_goldreich":
                    raise RuntimeError(f"Bad task type in {record['experiment_id']}")
                if record.get("variant_name") != condition:
                    raise RuntimeError(f"Bad condition in {record['experiment_id']}")
                if len(bits) != int(config["input_bits"]) or set(bits) - {"0", "1"}:
                    raise RuntimeError(f"Invalid input bits in {record['experiment_id']}")
                if len(prg_seed) != int(config["seed_bits"]) or set(prg_seed) - {"0", "1"}:
                    raise RuntimeError(f"Invalid PRG seed in {record['experiment_id']}")
                if int(spec["final_parity"]) != final_parity(bits):
                    raise RuntimeError(f"Bad final parity in {record['experiment_id']}")
                if spec["trace_protocol"] != protocol:
                    raise RuntimeError(f"Wrong protocol in {record['experiment_id']}")
                states = pair_cumulative_trace(bits)
                masks = goldreich_masks(prg_seed, graph)
                expected_trace = states if protocol == "pair_cumulative" else encrypted_trace(bits, masks)
                if spec["plain_states"] != states:
                    raise RuntimeError(f"Bad plain states in {record['experiment_id']}")
                if spec["gold_masks"] != masks:
                    raise RuntimeError(f"Bad masks in {record['experiment_id']}")
                if spec["gold_trace"] != expected_trace:
                    raise RuntimeError(f"Bad gold trace in {record['experiment_id']}")
                expected_suffix = supervised_suffix(bits, protocol, masks=masks)
                if record.get("supervised_suffix") != expected_suffix:
                    raise RuntimeError(f"Bad target in {record['experiment_id']}")
                if (
                    _canonical_public_cot_suffix(record, supervision_mode="record_target")
                    != expected_suffix
                ):
                    raise RuntimeError(f"Shared trainer cannot load {record['experiment_id']}")
                combined_text = f"{record['prompt_text']}\n{expected_suffix}".lower()
                if "private" in combined_text:
                    raise RuntimeError("This experiment must not introduce a private task.")

            if len({bits for bits, _ in items}) != len(items):
                raise RuntimeError(f"Duplicate inputs in {condition}/{split}")
            parity_counts = Counter(final_parity(bits) for bits, _ in items)
            if parity_counts != Counter({0: expected_n // 2, 1: expected_n // 2}):
                raise RuntimeError(f"Unbalanced parity in {condition}/{split}: {parity_counts}")
            twin_seed_values: set[str] = set()
            for twin_id, pair in twins.items():
                if len(pair) != 2:
                    raise RuntimeError(f"Twin group {twin_id} has {len(pair)} records")
                first, second = pair
                first_spec, second_spec = first["spec"], second["spec"]
                first_bits = str(first_spec["input_bits"])
                second_bits = str(second_spec["input_bits"])
                if complement(first_bits) != second_bits:
                    raise RuntimeError(f"Twin group {twin_id} is not complementary")
                if first_spec["prg_seed"] != second_spec["prg_seed"]:
                    raise RuntimeError(f"Twin group {twin_id} does not share its PRG seed")
                if first_spec["gold_trace"] != second_spec["gold_trace"]:
                    raise RuntimeError(f"Twin group {twin_id} does not share its trace")
                twin_seed_values.add(str(first_spec["prg_seed"]))
                twin_groups_checked += 1
            if len(twin_seed_values) != expected_n // 2:
                raise RuntimeError(f"PRG seeds are not unique across twin groups in {condition}/{split}")
            items_by_condition_split[(condition, split)] = items
            seeds_by_split.setdefault(split, twin_seed_values)
            if seeds_by_split[split] != twin_seed_values:
                raise RuntimeError(f"Conditions do not share the same {split} seeds")

    reference = next(iter(condition_map))
    for split in ("train", "val"):
        expected_items = items_by_condition_split[(reference, split)]
        for condition in condition_map:
            if items_by_condition_split[(condition, split)] != expected_items:
                raise RuntimeError(f"Conditions do not use identical {split} items")
    train_inputs = {bits for bits, _ in items_by_condition_split[(reference, "train")]}
    val_inputs = {bits for bits, _ in items_by_condition_split[(reference, "val")]}
    if train_inputs & val_inputs:
        raise RuntimeError("Train and validation inputs overlap")
    if seeds_by_split["train"] & seeds_by_split["val"]:
        raise RuntimeError("Train and validation PRG seeds overlap")

    manifest = json.loads((seed_root / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["fixed_ordered_hypergraph_zero_based"] != graph:
        raise RuntimeError("Manifest graph does not match the configured fixed graph")

    return {
        "train_n": len(train_inputs),
        "validation_n": len(val_inputs),
        "train_validation_input_overlap": 0,
        "train_unique_prg_seeds": len(seeds_by_split["train"]),
        "validation_unique_prg_seeds": len(seeds_by_split["val"]),
        "train_validation_seed_overlap": 0,
        "conditions": condition_map,
        "conditions_share_identical_items": True,
        "balanced_final_parity": True,
        "complementary_twin_groups_checked": twin_groups_checked,
        "fixed_graph": True,
        "no_private_task": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--split-root", type=Path, default=Path("generated_data/parity_goldreich"))
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.split_root), indent=2))


if __name__ == "__main__":
    main()
