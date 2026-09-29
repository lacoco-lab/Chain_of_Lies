#!/usr/bin/env python3
"""Generate paired one-derived/one-supplied Goldreich update ablations."""

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
        "experiment_name": "parity_goldreich_update_ablation_v1",
        "experiment_id": f"{task}_{index:05d}",
        "task_type": task,
        "system_prompt": SYSTEM,
        "prompt_text": prompt,
        "supervised_suffix": str(gold),
        "spec": {"task": task, "gold_bit": gold, **spec},
    }


def local_record(
    index: int, seed: str, edge_index: int, graph: list[list[int]]
) -> dict[str, Any]:
    edge = graph[edge_index]
    gold = mask_bit(seed, edge)
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
    """Reproduce the successful one-mask experiment's train/validation seed allocation."""
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

    train = make(4096)
    validation = make(int(config["local_validation_n"]))
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in train:
        grouped[(row["spec"]["edge_index"], row["spec"]["gold_bit"])].append(row)
    per_group = int(config["local_replay_n"]) // (2 * len(graph))
    replay = [row for key in sorted(grouped) for row in grouped[key][:per_group]]
    random.Random(f"{config['seed']}:update-ablation-local").shuffle(replay)
    return replay, validation, used


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


def condition_rows(
    condition: str,
    n: int,
    split: str,
    seeds: list[str],
    graph: list[list[int]],
    rng: random.Random,
) -> list[dict]:
    transitions = len(graph) - 1
    if condition not in {"derive_previous", "derive_current"}:
        raise ValueError(f"Unknown condition: {condition}")
    if n % (2 * transitions):
        raise ValueError("Condition size must balance every transition and output bit")
    rows, per, seed_index = [], n // (2 * transitions), 0
    for step in range(1, len(graph)):
        previous_edge, current_edge = graph[step - 1], graph[step]
        for wanted in (0, 1):
            for _ in range(per):
                seed = seeds[seed_index]
                seed_index += 1
                previous_z = mask_bit(seed, previous_edge)
                current_z = mask_bit(seed, current_edge)
                previous_c, left, right = (
                    rng.randrange(2),
                    rng.randrange(2),
                    rng.randrange(2),
                )
                if condition == "derive_previous":
                    current_z = wanted ^ previous_c ^ previous_z ^ left ^ right
                    mask_lines = (
                        f"Previous selector: {selector(previous_edge)}\n"
                        f"Current supplied mask z_cur: {current_z}\n"
                    )
                    instruction = "Privately compute z_prev from the previous selector"
                    attention_group = previous_edge
                else:
                    previous_z = wanted ^ previous_c ^ left ^ right ^ current_z
                    mask_lines = (
                        f"Previous supplied mask z_prev: {previous_z}\n"
                        f"Current selector: {selector(current_edge)}\n"
                    )
                    instruction = "Privately compute z_cur from the current selector"
                    attention_group = current_edge
                prompt = (
                    f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
                    f"Previous encrypted state c_prev: {previous_c}\n"
                    f"{mask_lines}Next input pair: {left} {right}\n"
                    f"{instruction}, then compute c_cur = c_prev XOR z_prev XOR left XOR right "
                    "XOR z_cur. Return only c_cur; do not output the derived mask."
                )
                rows.append(
                    record(
                        condition,
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
                            "attention_groups": [attention_group],
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
    random.Random(f"{config['seed']}:update-ablation-seeds").shuffle(candidates)
    train_n = int(config["condition_train_n"])
    validation_n = int(config["condition_validation_n"])
    train_seeds = candidates[:train_n]
    validation_seeds = candidates[train_n : train_n + validation_n]
    datasets = {
        "local_replay": local_replay,
        "local_validation": local_validation,
        "supplied_update_train": supplied_update_rows(
            int(config["supplied_update_train_n"]),
            "train",
            random.Random("ablation:supplied:train:0"),
        ),
        "supplied_update_validation": supplied_update_rows(
            int(config["supplied_update_validation_n"]),
            "validation",
            random.Random("ablation:supplied:val:0"),
        ),
    }
    for condition in ("derive_previous", "derive_current"):
        datasets[f"{condition}_train"] = condition_rows(
            condition,
            train_n,
            "train",
            train_seeds,
            graph,
            random.Random(f"ablation:{condition}:train:0"),
        )
        datasets[f"{condition}_validation"] = condition_rows(
            condition,
            validation_n,
            "validation",
            validation_seeds,
            graph,
            random.Random(f"ablation:{condition}:val:0"),
        )
    digests = {
        name: write_jsonl(output_root / f"{name}.jsonl", rows)
        for name, rows in datasets.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph": graph,
        "reserved_one_mask_seed_n": len(reserved),
        "condition_train_seed_n": len(train_seeds),
        "condition_validation_seed_n": len(validation_seeds),
        "all_seed_partitions_disjoint": not (
            reserved & set(train_seeds)
            or reserved & set(validation_seeds)
            or set(train_seeds) & set(validation_seeds)
        ),
        "paired_condition_seeds": True,
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
        default=Path("generated_data/parity_goldreich_update_ablation/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
