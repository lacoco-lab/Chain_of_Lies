#!/usr/bin/env python3
"""Generate deterministic synthetic data for the Goldreich toy transformer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


def predicate(values: list[int]) -> int:
    a, b, c, d, e = values
    return a ^ b ^ c ^ (d & e)


def fixed_graph(seed_bits: int, edge_count: int, graph_seed: int) -> list[list[int]]:
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(edge_count)]


def mask(seed: list[int], edge: list[int]) -> int:
    return predicate([seed[index] for index in edge])


def balanced_bits(length: int, wanted: int, rng: random.Random) -> list[int]:
    bits = [rng.randrange(2) for _ in range(length - 1)]
    bits.append(wanted ^ sum(bits) % 2)
    return bits


def encrypted_trajectory(
    bits: list[int], seed: list[int], graph: list[list[int]]
) -> tuple[list[int], list[int], int]:
    state = 0
    masks, encrypted = [], []
    for step, bit in enumerate(bits):
        current_mask = mask(seed, graph[step])
        state ^= bit
        masks.append(current_mask)
        encrypted.append(state ^ current_mask)
    return masks, encrypted, state


def seed_bits_for_length(config: dict, length: int) -> int:
    configured = int(config["seed_bits_by_length"][str(length)])
    # Goldreich's five-local predicate requires at least five seed positions.
    # This floor matters only for the added N=4 easy-regime point; all reported
    # N>=8 configurations already satisfy ceil(N**0.7) >= 5.
    expected = max(5, math.ceil(length ** float(config["seed_length_exponent"])))
    if configured != expected:
        raise ValueError(
            f"N={length} has {configured} seed bits, expected ceil(N^exponent)={expected}"
        )
    return configured


def split_seed_values(
    seed_bits: int, config: dict, length: int
) -> dict[str, list[int]]:
    """Choose deterministic, disjoint seed subsets without enumerating large spaces."""
    available = 1 << seed_bits
    requested_validation = int(config["validation_seed_n"])
    requested_test = int(config["test_seed_n"])
    validation_n = min(requested_validation, available // 8)
    test_n = min(requested_test, available // 8)
    train_n = min(int(config["train_seed_n"]), available - validation_n - test_n)
    selected_n = train_n + validation_n + test_n
    rng = random.Random(f"goldreich-toy-v2-seed-partition:0:n{length}:r{seed_bits}")
    if selected_n == available:
        selected = list(range(available))
        rng.shuffle(selected)
    else:
        chosen: set[int] = set()
        selected = []
        while len(selected) < selected_n:
            candidate = rng.getrandbits(seed_bits)
            if candidate not in chosen:
                chosen.add(candidate)
                selected.append(candidate)
    return {
        "train": selected[:train_n],
        "validation": selected[train_n : train_n + validation_n],
        "test": selected[train_n + validation_n :],
    }


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update((line + "\n").encode())
            handle.write(line + "\n")
            count += 1
    return count, digest.hexdigest()


def generate_rows(
    *,
    length: int,
    seed_bits: int,
    split: str,
    n: int,
    seeds: list[int],
    graph: list[list[int]],
    random_seed: str,
) -> Iterable[dict]:
    rng = random.Random(random_seed)
    order = seeds.copy()
    rng.shuffle(order)
    for index in range(n):
        seed_integer = order[index % len(order)]
        seed = [int(bit) for bit in f"{seed_integer:0{seed_bits}b}"]
        wanted = index % 2
        bits = balanced_bits(length, wanted, rng)
        masks, states, answer = encrypted_trajectory(bits, seed, graph)
        yield {
            "id": f"toy_n{length}_{split}_{index:07d}",
            "length": length,
            "split": split,
            "input_bits": "".join(map(str, bits)),
            "seed_bits": "".join(map(str, seed)),
            "masks": "".join(map(str, masks)),
            "encrypted_states": "".join(map(str, states)),
            "answer": answer,
        }


def generate(
    config_path: Path, output_root: Path, only_length: int | None = None
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["seed"]) != 0:
        raise ValueError("This calibrated experiment requires seed 0")
    lengths = list(map(int, config["lengths"]))
    if only_length is not None:
        if only_length not in lengths:
            raise ValueError(
                f"Requested N={only_length}, but the configured lengths are {lengths}"
            )
        lengths = [only_length]
    files, length_metadata = {}, {}
    for length in lengths:
        seed_bits = seed_bits_for_length(config, length)
        graph = fixed_graph(seed_bits, length, int(config["graph_seed"]))
        seed_splits = split_seed_values(seed_bits, config, length)
        length_metadata[str(length)] = {
            "seed_bits": seed_bits,
            "mask_bits": length,
            "graph": graph,
            "seed_split_sizes": {key: len(value) for key, value in seed_splits.items()},
            "seed_split_sha256": {
                split: hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()
                for split, values in seed_splits.items()
            },
        }
        for split in ("train", "validation", "test"):
            n = int(config[f"{split}_n_per_length"])
            path = output_root / f"n{length}" / f"{split}.jsonl.gz"
            size, digest = write_jsonl_gz(
                path,
                generate_rows(
                    length=length,
                    seed_bits=seed_bits,
                    split=split,
                    n=n,
                    seeds=seed_splits[split],
                    graph=graph,
                    random_seed=f"goldreich-toy-v2:0:n{length}:{split}",
                ),
            )
            files[f"n{length}/{split}"] = {
                "path": str(path),
                "n": size,
                "sha256": digest,
            }
    manifest = {
        "schema_version": 2,
        "experiment_name": config["experiment_name"],
        "seed": 0,
        "graph_seed": config["graph_seed"],
        "lengths": lengths,
        "seed_length_exponent": config["seed_length_exponent"],
        "length_metadata": length_metadata,
        "files": files,
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
        default=Path("generated_data/parity_goldreich_toy_transformer/seed_0"),
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Generate only this configured length (useful for independent Condor jobs)",
    )
    args = parser.parse_args()
    result = generate(args.config, args.output_root, args.length)
    print(
        json.dumps(
            {
                "status": "generated",
                "output_root": str(args.output_root),
                "lengths": {
                    length: {
                        "seed_bits": metadata["seed_bits"],
                        "mask_bits": metadata["mask_bits"],
                        "seed_split_sizes": metadata["seed_split_sizes"],
                    }
                    for length, metadata in result["length_metadata"].items()
                },
                "file_count": len(result["files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
