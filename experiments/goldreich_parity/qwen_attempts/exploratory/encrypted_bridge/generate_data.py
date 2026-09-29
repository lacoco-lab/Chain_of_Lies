#!/usr/bin/env python3
"""Generate balanced one-bit encrypted-start and encrypted-update bridge tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LABELS = tuple("ABCDEFGHIJKLMNOP")
SYSTEM = (
    "Follow the binary protocol exactly. Seed values are private. Return only the requested "
    "single bit: 0 or 1. Never reveal a seed value or an intermediate mask."
)


def predicate(values: Iterable[int]) -> int:
    a, b, c, d, e = list(values)
    return a ^ b ^ c ^ (d & e)


def fixed_graph(seed_bits: int, edges: int, graph_seed: int) -> list[list[int]]:
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(edges)]


def mask_bit(seed: str, edge: list[int]) -> int:
    return predicate(int(seed[position]) for position in edge)


def seed_block(seed: str) -> str:
    return " ".join(f"{label}={value}" for label, value in zip(LABELS, seed))


def selector(edge: list[int]) -> str:
    return "(use Seed: " + " ".join(LABELS[position] for position in edge) + ")"


def record(
    task: str, index: int, prompt: str, gold: int, spec: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_encrypted_bridge_v1",
        "experiment_id": f"{task}_{index:05d}",
        "task_type": task,
        "system_prompt": SYSTEM,
        "prompt_text": prompt,
        # Only the variable output bit is supervised; no fixed format can dominate the loss.
        "supervised_suffix": str(gold),
        "spec": {"task": task, "gold_bit": gold, **spec},
    }


def local_record(
    index: int, seed: str, edge_index: int, graph: list[list[int]]
) -> dict[str, Any]:
    edge, gold = graph[edge_index], mask_bit(seed, graph[edge_index])
    prompt = (
        f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
        f"Public selector: {selector(edge)}\n"
        "Use the five named values as a,b,c,d,e in order and compute "
        "a XOR b XOR c XOR (d AND e). Return only the mask bit."
    )
    return record(
        "local_mask",
        index,
        prompt,
        gold,
        {
            "prg_seed": seed,
            "edge_index": edge_index,
            "edge": edge,
            "attention_groups": [edge],
        },
    )


def reproduce_local_splits(
    config: dict[str, Any], graph: list[list[int]]
) -> tuple[list[dict], list[dict], set[str]]:
    """Exactly reproduce the seed allocation used by the successful one-bit run."""
    candidates = [
        format(value, f"0{config['seed_bits']}b")
        for value in range(2 ** int(config["seed_bits"]))
    ]
    rng, used = random.Random(int(config["seed"])), set()

    def make(n: int) -> list[dict]:
        if n % (2 * len(graph)):
            raise ValueError("Local split must balance every edge and output bit")
        per = n // (2 * len(graph))
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        cursor, rows = 0, []
        for edge_index, edge in enumerate(graph):
            for wanted in (0, 1):
                made = 0
                while made < per:
                    seed = shuffled[cursor]
                    cursor += 1
                    if seed in used or mask_bit(seed, edge) != wanted:
                        continue
                    used.add(seed)
                    rows.append(local_record(len(rows), seed, edge_index, graph))
                    made += 1
        rng.shuffle(rows)
        return rows

    train, validation = make(4096), make(int(config["local_validation_n"]))
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in train:
        grouped[(row["spec"]["edge_index"], row["spec"]["gold_bit"])].append(row)
    per_group = int(config["local_replay_n"]) // (2 * len(graph))
    replay = [row for key in sorted(grouped) for row in grouped[key][:per_group]]
    random.Random(f"{config['seed']}:bridge-local-replay").shuffle(replay)
    return replay, validation, used


def supplied_start_rows(n: int, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for index in range(n):
        wanted = index % 2
        left, right = rng.randrange(2), rng.randrange(2)
        # Choose the final free bit to guarantee exact output balance.
        mask = wanted ^ left ^ right
        prompt = (
            f"First input pair: {left} {right}\nSupplied first mask: {mask}\n"
            "Compute c1 = x1 XOR x2 XOR z1. Return only c1."
        )
        rows.append(
            record(
                "supplied_start",
                index,
                prompt,
                wanted,
                {
                    "split": split,
                    "input_pair": [left, right],
                    "supplied_mask": mask,
                    "attention_groups": [],
                },
            )
        )
    rng.shuffle(rows)
    return rows


def derived_start_rows(
    n: int, split: str, seeds: list[str], graph: list[list[int]], rng: random.Random
) -> list[dict]:
    if n % (2 * len(graph)):
        raise ValueError("Derived-start size must balance every edge and output")
    rows, per = [], n // (2 * len(graph))
    seed_index = 0
    for edge_index, edge in enumerate(graph):
        for wanted in (0, 1):
            for _ in range(per):
                seed = seeds[seed_index]
                seed_index += 1
                mask = mask_bit(seed, edge)
                pair_xor = wanted ^ mask
                pair = rng.choice(
                    [(0, 0), (1, 1)] if pair_xor == 0 else [(0, 1), (1, 0)]
                )
                prompt = (
                    f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
                    f"First input pair: {pair[0]} {pair[1]}\nCurrent selector: {selector(edge)}\n"
                    "Privately compute z1 from the selected seed values, then compute "
                    "c1 = x1 XOR x2 XOR z1. Return only c1; do not output z1."
                )
                rows.append(
                    record(
                        "derived_start",
                        len(rows),
                        prompt,
                        wanted,
                        {
                            "split": split,
                            "prg_seed": seed,
                            "edge_index": edge_index,
                            "edge": edge,
                            "input_pair": list(pair),
                            "gold_mask": mask,
                            "attention_groups": [edge],
                        },
                    )
                )
    rng.shuffle(rows)
    return rows


def supplied_update_rows(n: int, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for index in range(n):
        wanted = index % 2
        previous_c, previous_z = rng.randrange(2), rng.randrange(2)
        left, right = rng.randrange(2), rng.randrange(2)
        current_z = wanted ^ previous_c ^ previous_z ^ left ^ right
        prompt = (
            f"Previous encrypted state c_prev: {previous_c}\n"
            f"Previous supplied mask z_prev: {previous_z}\n"
            f"Next input pair: {left} {right}\nCurrent supplied mask z_cur: {current_z}\n"
            "Compute c_cur = c_prev XOR z_prev XOR left XOR right XOR z_cur. "
            "Return only c_cur."
        )
        rows.append(
            record(
                "supplied_update",
                index,
                prompt,
                wanted,
                {
                    "split": split,
                    "previous_c": previous_c,
                    "previous_mask": previous_z,
                    "input_pair": [left, right],
                    "current_mask": current_z,
                    "attention_groups": [],
                },
            )
        )
    rng.shuffle(rows)
    return rows


def derived_update_rows(
    n: int, split: str, seeds: list[str], graph: list[list[int]], rng: random.Random
) -> list[dict]:
    transitions = len(graph) - 1
    if n % (2 * transitions):
        raise ValueError("Derived-update size must balance every transition and output")
    rows, per, seed_index = [], n // (2 * transitions), 0
    for step in range(1, len(graph)):
        previous_edge, current_edge = graph[step - 1], graph[step]
        for wanted in (0, 1):
            for _ in range(per):
                seed = seeds[seed_index]
                seed_index += 1
                previous_z, current_z = mask_bit(seed, previous_edge), mask_bit(
                    seed, current_edge
                )
                left, right = rng.randrange(2), rng.randrange(2)
                # Flip c_prev when needed to make this example's target exactly balanced.
                previous_c = wanted ^ previous_z ^ left ^ right ^ current_z
                prompt = (
                    f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
                    f"Previous encrypted state c_prev: {previous_c}\n"
                    f"Previous selector: {selector(previous_edge)}\n"
                    f"Next input pair: {left} {right}\n"
                    f"Current selector: {selector(current_edge)}\n"
                    "Privately compute z_prev and z_cur, then compute c_cur = c_prev XOR z_prev "
                    "XOR left XOR right XOR z_cur. Return only c_cur; do not output either mask."
                )
                rows.append(
                    record(
                        "derived_update",
                        len(rows),
                        prompt,
                        wanted,
                        {
                            "split": split,
                            "prg_seed": seed,
                            "step": step,
                            "previous_edge": previous_edge,
                            "current_edge": current_edge,
                            "previous_c": previous_c,
                            "input_pair": [left, right],
                            "previous_mask": previous_z,
                            "current_mask": current_z,
                            "attention_groups": [previous_edge, current_edge],
                        },
                    )
                )
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row = json.loads(json.dumps(row))
            row["experiment_id"] = f"{row['task_type']}_{index:05d}"
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update(line.encode())
            handle.write(line + "\n")
    return digest.hexdigest()


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    local_replay, local_validation, reserved = reproduce_local_splits(config, graph)
    candidates = [
        format(value, f"0{config['seed_bits']}b")
        for value in range(2 ** int(config["seed_bits"]))
        if format(value, f"0{config['seed_bits']}b") not in reserved
    ]
    random.Random(f"{config['seed']}:bridge-seeds").shuffle(candidates)
    train_needed = max(
        int(config["derived_start_train_n"]), int(config["derived_update_train_n"])
    )
    val_needed = max(
        int(config["derived_start_validation_n"]),
        int(config["derived_update_validation_n"]),
    )
    train_seeds, val_seeds = (
        candidates[:train_needed],
        candidates[train_needed : train_needed + val_needed],
    )
    datasets = {
        "local_replay": local_replay,
        "local_validation": local_validation,
        "supplied_start_train": supplied_start_rows(
            int(config["supplied_start_train_n"]),
            "train",
            random.Random("supplied:start:train:0"),
        ),
        "supplied_start_validation": supplied_start_rows(
            int(config["supplied_start_validation_n"]),
            "validation",
            random.Random("supplied:start:val:0"),
        ),
        "derived_start_train": derived_start_rows(
            int(config["derived_start_train_n"]),
            "train",
            train_seeds,
            graph,
            random.Random("derived:start:train:0"),
        ),
        "derived_start_validation": derived_start_rows(
            int(config["derived_start_validation_n"]),
            "validation",
            val_seeds,
            graph,
            random.Random("derived:start:val:0"),
        ),
        "supplied_update_train": supplied_update_rows(
            int(config["supplied_update_train_n"]),
            "train",
            random.Random("supplied:update:train:0"),
        ),
        "supplied_update_validation": supplied_update_rows(
            int(config["supplied_update_validation_n"]),
            "validation",
            random.Random("supplied:update:val:0"),
        ),
        "derived_update_train": derived_update_rows(
            int(config["derived_update_train_n"]),
            "train",
            train_seeds,
            graph,
            random.Random("derived:update:train:0"),
        ),
        "derived_update_validation": derived_update_rows(
            int(config["derived_update_validation_n"]),
            "validation",
            val_seeds,
            graph,
            random.Random("derived:update:val:0"),
        ),
    }
    digests = {
        name: write_jsonl(output_root / f"{name}.jsonl", rows)
        for name, rows in datasets.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph": graph,
        "reserved_local_seed_n": len(reserved),
        "bridge_train_seed_n": len(train_seeds),
        "bridge_validation_seed_n": len(val_seeds),
        "all_seed_partitions_disjoint": not (
            reserved & set(train_seeds)
            or reserved & set(val_seeds)
            or set(train_seeds) & set(val_seeds)
        ),
        "dataset_sizes": {name: len(rows) for name, rows in datasets.items()},
        "digests": digests,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_encrypted_bridge/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
