#!/usr/bin/env python3
"""Generate exhaustive, balanced, disjoint PARITY data for N=4,8,16."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

EASY_LENGTHS = (4, 8, 16)


def partition(length: int, seed: int) -> dict[str, list[int]]:
    parts = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        values = [
            value for value in range(1 << length) if value.bit_count() % 2 == label
        ]
        random.Random(f"standard-controls:{seed}:n{length}:label{label}").shuffle(
            values
        )
        half, three_quarters = len(values) // 2, 3 * len(values) // 4
        parts["train"].extend(values[:half])
        parts["validation"].extend(values[half:three_quarters])
        parts["test"].extend(values[three_quarters:])
    for split, values in parts.items():
        random.Random(f"standard-controls:{seed}:n{length}:{split}").shuffle(values)
    sets = {name: set(values) for name, values in parts.items()}
    if (
        any(
            sets[a] & sets[b]
            for a, b in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        )
        or len(set().union(*sets.values())) != 1 << length
    ):
        raise RuntimeError(f"Invalid exhaustive partition at N={length}")
    return parts


def generate(config_path: Path, output_root: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "lengths": list(EASY_LENGTHS),
        "partition": "complete domain; balanced 1/2 train, 1/4 validation, 1/4 test",
        "files": {},
    }
    for length in EASY_LENGTHS:
        for split, values in partition(length, seed).items():
            path = output_root / f"n{length}" / f"{split}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
                for index, value in enumerate(values):
                    row = {
                        "id": f"standard_control_n{length}_{split}_{index:06d}",
                        "input_bits": f"{value:0{length}b}",
                        "answer": value.bit_count() % 2,
                    }
                    line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    handle.write(line)
                    digest.update(line.encode())
            manifest["files"][f"n{length}/{split}"] = {
                "rows": len(values),
                "sha256": digest.hexdigest(),
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
        default=Path("generated_data/parity_standard_cot_controls_easy/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
