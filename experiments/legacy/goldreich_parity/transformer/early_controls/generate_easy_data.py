#!/usr/bin/env python3
"""Generate exhaustive, deterministic, disjoint small-N PARITY splits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


def write_split(
    path: Path, length: int, split: str, values: list[int]
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for index, value in enumerate(values):
            bits = f"{value:0{length}b}"
            row = {
                "id": f"parity_control_n{length}_{split}_{index:06d}",
                "length": length,
                "split": split,
                "input_bits": bits,
                "answer": value.bit_count() % 2,
            }
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode())
            handle.write(line)
    return {"path": str(path), "n": len(values), "sha256": digest.hexdigest()}


def partition_domain(length: int, seed: int) -> dict[str, list[int]]:
    by_label = {
        label: [value for value in range(1 << length) if value.bit_count() % 2 == label]
        for label in (0, 1)
    }
    for label, values in by_label.items():
        random.Random(f"parity-control-easy:{seed}:n{length}:label{label}").shuffle(
            values
        )
    per_label = 1 << (length - 1)
    train_end = per_label // 2
    validation_end = train_end + per_label // 4
    parts = {"train": [], "validation": [], "test": []}
    for values in by_label.values():
        parts["train"].extend(values[:train_end])
        parts["validation"].extend(values[train_end:validation_end])
        parts["test"].extend(values[validation_end:])
    for split, values in parts.items():
        random.Random(f"parity-control-easy:{seed}:n{length}:{split}").shuffle(values)
    sets = {split: set(values) for split, values in parts.items()}
    if any(
        sets[a] & sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError(f"Generated splits overlap at N={length}")
    if len(set().union(*sets.values())) != 1 << length:
        raise RuntimeError(f"Generated splits do not cover the N={length} domain")
    return parts


def generate(config_path: Path, output_root: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    files, domains = {}, {}
    for length in map(int, config["lengths"]):
        parts = partition_domain(length, seed)
        domains[str(length)] = {
            "domain_size": 1 << length,
            "split_sizes": {split: len(values) for split, values in parts.items()},
            "exhaustive_union": True,
            "pairwise_disjoint": True,
        }
        for split, values in parts.items():
            key = f"n{length}/{split}"
            files[key] = write_split(
                output_root / f"n{length}" / f"{split}.jsonl.gz",
                length,
                split,
                values,
            )
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "partition": "complete domain, stratified 1/2 train, 1/4 validation, 1/4 test",
        "domains": domains,
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
        "--config", type=Path, default=Path(__file__).with_name("easy_config.json")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated_data/parity_cot_controls_easy/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
