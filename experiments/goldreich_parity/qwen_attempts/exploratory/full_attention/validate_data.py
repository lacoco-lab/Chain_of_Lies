#!/usr/bin/env python3
"""Validate encryption, offsets, twins, balance, graphs, and all seed partitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_data import LABELS, fixed_graph, mask_bit, parity, plain_states, selector


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate(config_path: Path, data_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reserved_local_seed_n"] == 4096 + int(config["local_validation_n"])
    assert manifest["all_seed_partitions_disjoint"] is True
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    local_replay = read_jsonl(data_root / "local_replay.jsonl")
    local_validation = read_jsonl(data_root / "local_validation.jsonl")
    assert len(local_replay) == int(config["local_replay_pool_n"])
    assert len(local_validation) == int(config["local_validation_n"])
    local_seeds = set()
    for rows in (local_replay, local_validation):
        counts = Counter()
        for row in rows:
            spec, suffix = row["spec"], row["supervised_suffix"]
            edge = graph[int(spec["edge_index"])]
            gold = mask_bit(spec["prg_seed"], edge)
            assert spec["edge"] == edge and int(suffix) == gold
            assert spec["bit_offsets"] == [0]
            counts[(spec["edge_index"], gold)] += 1
            local_seeds.add(spec["prg_seed"])
        assert len(set(counts.values())) == 1

    result = {
        "local_replay_n": len(local_replay),
        "local_validation_n": len(local_validation),
        "lengths": {},
    }
    trace_train_seeds, trace_validation_seeds = set(), set()
    for length in [*config["core_lengths"], *config["extension_lengths"]]:
        split_seeds = {}
        for split, expected_n in (
            ("train", config["trace_train_n"]),
            ("validation", config["trace_validation_n"]),
        ):
            rows = read_jsonl(data_root / f"trace_{split}_{length}.jsonl")
            assert len(rows) == int(expected_n)
            final_counts, twins, seeds = Counter(), {}, set()
            for row in rows:
                spec, target = row["spec"], row["supervised_suffix"]
                seed, bits = spec["prg_seed"], spec["input_bits"]
                assert (
                    len(bits) == 2 * int(length)
                    and parity(bits) == spec["final_parity"]
                )
                assert spec["edges"] == graph[: int(length)]
                assert spec["selectors"] == [
                    selector(edge) for edge in graph[: int(length)]
                ]
                masks = [mask_bit(seed, edge) for edge in graph[: int(length)]]
                states = plain_states(bits)
                encrypted = [state ^ mask for state, mask in zip(states, masks)]
                assert spec["gold_masks"] == masks and spec["plain_states"] == states
                assert spec["gold_sequence"] == encrypted
                for offset, bit in zip(spec["bit_offsets"], encrypted):
                    assert target[int(offset)] == str(bit)
                assert target[int(spec["answer_offset"])] == str(parity(bits))
                assert target.startswith("<COT>\n") and target.endswith("</ANSWER>")
                final_counts[parity(bits)] += 1
                seeds.add(seed)
                twins.setdefault(spec["twin_id"], []).append(spec)
            assert final_counts[0] == final_counts[1] == len(rows) // 2
            assert all(len(pair) == 2 for pair in twins.values())
            assert all(
                pair[0]["gold_sequence"] == pair[1]["gold_sequence"]
                for pair in twins.values()
            )
            split_seeds[split] = seeds
            (trace_train_seeds if split == "train" else trace_validation_seeds).update(
                seeds
            )
        assert not (split_seeds["train"] & split_seeds["validation"])
        result["lengths"][str(length)] = {
            "train_n": config["trace_train_n"],
            "validation_n": config["trace_validation_n"],
            "train_seed_n": len(split_seeds["train"]),
            "validation_seed_n": len(split_seeds["validation"]),
        }
    assert not (local_seeds & trace_train_seeds)
    assert not (local_seeds & trace_validation_seeds)
    assert not (trace_train_seeds & trace_validation_seeds)
    result["all_observed_seed_partitions_disjoint"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_full_attention/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.data_root), indent=2))


if __name__ == "__main__":
    main()
