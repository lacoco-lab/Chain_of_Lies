#!/usr/bin/env python3
"""Validate balance, privacy format, formulas, and seed separation."""

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
    assert manifest["all_seed_partitions_disjoint"]
    assert manifest["paired_condition_seeds"]
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    report = {}
    for name, expected_n in manifest["dataset_sizes"].items():
        rows = read_jsonl(data_root / f"{name}.jsonl")
        assert len(rows) == expected_n
        counts = Counter(int(row["supervised_suffix"]) for row in rows)
        assert counts == {0: expected_n // 2, 1: expected_n // 2}
        for row in rows:
            assert row["supervised_suffix"] in {"0", "1"}
            assert "gold_mask" not in row["prompt_text"]
            spec = row["spec"]
            if spec["task"] == "local_mask":
                assert spec["gold_bit"] == mask_bit(
                    spec["prg_seed"], graph[spec["edge_index"]]
                )
            elif spec["task"] != "supplied_update":
                previous_z = mask_bit(spec["prg_seed"], spec["previous_edge"])
                current_z = mask_bit(spec["prg_seed"], spec["current_edge"])
                if spec["task"] == "derive_previous":
                    assert previous_z == spec["previous_mask"]
                else:
                    assert current_z == spec["current_mask"]
                left, right = spec["input_pair"]
                expected = (
                    spec["previous_c"]
                    ^ spec["previous_mask"]
                    ^ left
                    ^ right
                    ^ spec["current_mask"]
                )
                assert expected == spec["gold_bit"]
        report[name] = {"n": len(rows), "zeros": counts[0], "ones": counts[1]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_update_ablation/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.data_root), indent=2))


if __name__ == "__main__":
    main()
