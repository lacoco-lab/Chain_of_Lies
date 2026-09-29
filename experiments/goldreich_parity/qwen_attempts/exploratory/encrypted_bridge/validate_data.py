#!/usr/bin/env python3
"""Validate bridge formulas, balance, graph coverage, and held-out seed partitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_data import fixed_graph, mask_bit


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate(config_path: Path, data_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_seed_partitions_disjoint"] is True
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    result = {}
    names = ["local_replay", "local_validation"] + [
        f"{task}_{split}"
        for task in (
            "supplied_start",
            "derived_start",
            "supplied_update",
            "derived_update",
        )
        for split in ("train", "validation")
    ]
    for name in names:
        rows = read_jsonl(data_root / f"{name}.jsonl")
        assert len(rows) == manifest["dataset_sizes"][name]
        output_counts = Counter(int(row["spec"]["gold_bit"]) for row in rows)
        assert output_counts[0] == output_counts[1]
        for row in rows:
            spec, gold = row["spec"], int(row["supervised_suffix"])
            assert gold == int(spec["gold_bit"]) and row["supervised_suffix"] in {
                "0",
                "1",
            }
            task = spec["task"]
            if task == "local_mask":
                assert gold == mask_bit(spec["prg_seed"], spec["edge"])
            elif task == "supplied_start":
                left, right = spec["input_pair"]
                assert gold == left ^ right ^ spec["supplied_mask"]
            elif task == "derived_start":
                left, right = spec["input_pair"]
                assert spec["edge"] == graph[spec["edge_index"]]
                assert spec["gold_mask"] == mask_bit(spec["prg_seed"], spec["edge"])
                assert gold == left ^ right ^ spec["gold_mask"]
            elif task == "supplied_update":
                left, right = spec["input_pair"]
                assert gold == (
                    spec["previous_c"]
                    ^ spec["previous_mask"]
                    ^ left
                    ^ right
                    ^ spec["current_mask"]
                )
            elif task == "derived_update":
                left, right = spec["input_pair"]
                assert spec["previous_edge"] == graph[spec["step"] - 1]
                assert spec["current_edge"] == graph[spec["step"]]
                assert spec["previous_mask"] == mask_bit(
                    spec["prg_seed"], spec["previous_edge"]
                )
                assert spec["current_mask"] == mask_bit(
                    spec["prg_seed"], spec["current_edge"]
                )
                assert gold == (
                    spec["previous_c"]
                    ^ spec["previous_mask"]
                    ^ left
                    ^ right
                    ^ spec["current_mask"]
                )
            else:
                raise AssertionError(task)
        result[name] = {
            "n": len(rows),
            "zeros": output_counts[0],
            "ones": output_counts[1],
        }
    # Main bridge data cover every fixed edge/transition and output exactly evenly.
    for split in ("train", "validation"):
        start = read_jsonl(data_root / f"derived_start_{split}.jsonl")
        start_counts = Counter(
            (row["spec"]["edge_index"], row["spec"]["gold_bit"]) for row in start
        )
        assert len(start_counts) == 128 and len(set(start_counts.values())) == 1
        update = read_jsonl(data_root / f"derived_update_{split}.jsonl")
        update_counts = Counter(
            (row["spec"]["step"], row["spec"]["gold_bit"]) for row in update
        )
        assert len(update_counts) == 126 and len(set(update_counts.values())) == 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_encrypted_bridge/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.data_root), indent=2))


if __name__ == "__main__":
    main()
