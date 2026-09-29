#!/usr/bin/env python3
"""Validate balance, fixed graph use, targets, and train/validation separation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_data import LABELS, fixed_graph, mask_bit, selector


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate(config_path: Path, data_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    split_rows = {
        name: read_jsonl(data_root / f"{name}.jsonl")
        for name in ("train", "validation")
    }
    seed_sets: dict[str, set[str]] = {}
    result = {}
    for split, rows in split_rows.items():
        assert len(rows) == int(config[f"{split}_n"])
        counts = Counter()
        seeds = set()
        for row in rows:
            spec = row["spec"]
            edge_index, seed = int(spec["edge_index"]), str(spec["prg_seed"])
            edge = graph[edge_index]
            gold = mask_bit(seed, edge)
            assert spec["edge"] == edge
            assert spec["edge_labels"] == [LABELS[position] for position in edge]
            assert int(spec["gold_bit"]) == gold
            assert row["supervised_suffix"] == str(gold)
            assert selector(edge) in row["prompt_text"]
            for label, value in zip(LABELS, seed):
                assert f"{label}={value}" in row["prompt_text"]
            counts[(edge_index, gold)] += 1
            seeds.add(seed)
        expected = len(rows) // (2 * len(graph))
        assert set(counts.values()) == {expected}
        assert len(seeds) == len(rows)
        seed_sets[split] = seeds
        result[split] = {
            "rows": len(rows),
            "unique_seeds": len(seeds),
            "examples_per_edge_and_bit": expected,
        }
    assert not (seed_sets["train"] & seed_sets["validation"])
    result["seed_overlap"] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_bit_attention/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.data_root), indent=2))


if __name__ == "__main__":
    main()
