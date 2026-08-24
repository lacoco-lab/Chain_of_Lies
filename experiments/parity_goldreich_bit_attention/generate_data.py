#!/usr/bin/env python3
"""Generate balanced one-mask-bit data with disjoint train/validation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


LABELS = tuple("ABCDEFGHIJKLMNOP")
SYSTEM = (
    "Follow the binary protocol exactly. The labeled seed is private. "
    "Return only one mask bit: 0 or 1."
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
    # The explicit LABEL=BIT representation makes every value span unambiguous.
    return " ".join(f"{label}={value}" for label, value in zip(LABELS, seed))


def selector(edge: list[int]) -> str:
    return "(use Seed: " + " ".join(LABELS[position] for position in edge) + ")"


def make_record(index: int, split: str, seed: str, edge_index: int,
                graph: list[list[int]]) -> dict[str, Any]:
    edge = graph[edge_index]
    answer = mask_bit(seed, edge)
    prompt = (
        f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
        f"Public selector: {selector(edge)}\n"
        "Use the five named seed values as a, b, c, d, e in that order.\n"
        "Compute a XOR b XOR c XOR (d AND e).\n"
        "Return only 0 or 1."
    )
    return {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_bit_attention_v1",
        "experiment_id": f"{split}_{index:05d}",
        "task_type": "goldreich_one_mask_bit",
        "variant_name": "one_mask_bit",
        "system_prompt": SYSTEM,
        "prompt_text": prompt,
        # There is deliberately no fixed-format target text: every supervised token varies.
        "supervised_suffix": str(answer),
        "spec": {
            "split": split,
            "prg_seed": seed,
            "edge_index": edge_index,
            "edge": edge,
            "edge_labels": [LABELS[position] for position in edge],
            "gold_bit": answer,
        },
    }


def balanced_records(n: int, split: str, graph: list[list[int]], candidates: list[str],
                     used: set[str], rng: random.Random) -> list[dict[str, Any]]:
    if n % (2 * len(graph)):
        raise ValueError(f"{split} size must be divisible by {2 * len(graph)}")
    per_edge_per_bit = n // (2 * len(graph))
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    cursor = 0
    records: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(graph):
        for wanted in (0, 1):
            made = 0
            while made < per_edge_per_bit:
                if cursor >= len(shuffled):
                    raise RuntimeError("Exhausted unique seeds while balancing the dataset")
                seed = shuffled[cursor]
                cursor += 1
                if seed in used or mask_bit(seed, edge) != wanted:
                    continue
                used.add(seed)
                records.append(make_record(len(records), split, seed, edge_index, graph))
                made += 1
    rng.shuffle(records)
    for index, record in enumerate(records):
        record["experiment_id"] = f"{split}_{index:05d}"
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update(line.encode())
            handle.write(line + "\n")
    return digest.hexdigest()


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = random.Random(int(config["seed"]))
    seed_bits = int(config["seed_bits"])
    graph = fixed_graph(seed_bits, int(config["graph_edges"]), int(config["graph_seed"]))
    candidates = [format(value, f"0{seed_bits}b") for value in range(2 ** seed_bits)]
    used: set[str] = set()
    train = balanced_records(int(config["train_n"]), "train", graph, candidates, used, rng)
    validation = balanced_records(
        int(config["validation_n"]), "validation", graph, candidates, used, rng
    )
    train_digest = write_jsonl(output_root / "train.jsonl", train)
    validation_digest = write_jsonl(output_root / "validation.jsonl", validation)
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph_seed": config["graph_seed"],
        "graph": graph,
        "train_n": len(train),
        "validation_n": len(validation),
        "train_unique_seeds": len({row["spec"]["prg_seed"] for row in train}),
        "validation_unique_seeds": len({row["spec"]["prg_seed"] for row in validation}),
        "seed_overlap": len(
            {row["spec"]["prg_seed"] for row in train}
            & {row["spec"]["prg_seed"] for row in validation}
        ),
        "train_sha256": train_digest,
        "validation_sha256": validation_digest,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("generated_data/parity_goldreich_bit_attention/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
