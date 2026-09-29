#!/usr/bin/env python3
"""Validate formulas, balance, hashes, and held-out seeds in toy-transformer data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from generate_data import encrypted_trajectory, fixed_graph, seed_bits_for_length


def read_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate(
    config_path: Path, data_root: Path, only_length: int | None = None
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root_manifest_path = data_root / "manifest.json"
    root_manifest = (
        json.loads(root_manifest_path.read_text(encoding="utf-8"))
        if root_manifest_path.is_file()
        else None
    )
    lengths = list(map(int, config["lengths"]))
    if only_length is not None:
        if only_length not in lengths:
            raise ValueError(
                f"Requested N={only_length}, but the configured lengths are {lengths}"
            )
        lengths = [only_length]
    report = {}
    for length in lengths:
        # Generation can run once for every configured length (one root
        # manifest) or as independent Condor jobs (one manifest per nN).
        if root_manifest is not None and str(length) in root_manifest.get(
            "length_metadata", {}
        ):
            manifest = root_manifest
            manifest_path = root_manifest_path
        else:
            manifest_path = data_root / f"n{length}" / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"No root or per-length manifest found for N={length}: "
                    f"checked {root_manifest_path} and {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["schema_version"]) != 2:
            raise ValueError(
                f"Expected one-bit-per-step schema version 2 in {manifest_path}"
            )
        seed_bits = seed_bits_for_length(config, length)
        graph = fixed_graph(seed_bits, length, int(config["graph_seed"]))
        metadata = manifest["length_metadata"][str(length)]
        if int(metadata["seed_bits"]) != seed_bits or metadata["graph"] != graph:
            raise ValueError(f"Graph or seed length mismatch for N={length}")
        split_seeds = {split: set() for split in ("train", "validation", "test")}
        for split in split_seeds:
            path = data_root / f"n{length}" / f"{split}.jsonl.gz"
            digest = hashlib.sha256()
            counts = Counter()
            n = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    digest.update(line.encode())
                    row = json.loads(line)
                    bits = list(map(int, row["input_bits"]))
                    seed = list(map(int, row["seed_bits"]))
                    masks, states, answer = encrypted_trajectory(bits, seed, graph)
                    if (
                        len(bits) != length
                        or len(seed) != seed_bits
                        or len(masks) != length
                        or len(states) != length
                    ):
                        raise ValueError(f"Invalid shape in {row['id']}")
                    if (
                        row["masks"] != "".join(map(str, masks))
                        or row["encrypted_states"] != "".join(map(str, states))
                        or int(row["answer"]) != answer
                    ):
                        raise ValueError(f"Invalid target in {row['id']}")
                    if row["split"] != split or int(row["length"]) != length:
                        raise ValueError(f"Invalid metadata in {row['id']}")
                    split_seeds[split].add(row["seed_bits"])
                    counts[answer] += 1
                    n += 1
            expected = int(config[f"{split}_n_per_length"])
            key = f"n{length}/{split}"
            if n != expected or counts != {0: expected // 2, 1: expected // 2}:
                raise ValueError(f"Invalid size or balance in {key}: {n}, {counts}")
            if digest.hexdigest() != manifest["files"][key]["sha256"]:
                raise ValueError(f"Digest mismatch in {key}")
            report[key] = {
                "n": n,
                "answer_zeros": counts[0],
                "answer_ones": counts[1],
                "unique_seeds_seen": len(split_seeds[split]),
            }
        if any(
            split_seeds[a] & split_seeds[b]
            for a, b in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ):
            raise ValueError(f"Seed partitions overlap for N={length}")
        for split, values in split_seeds.items():
            if len(values) != int(metadata["seed_split_sizes"][split]):
                raise ValueError(
                    f"Not every configured {split} seed appears for N={length}"
                )
        report[f"n{length}/seed_partitions_disjoint"] = True
    report["seed_partitions_disjoint"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_toy_transformer/seed_0"),
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Validate only this configured length (useful for independent Condor jobs)",
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.data_root, args.length), indent=2))


if __name__ == "__main__":
    main()
