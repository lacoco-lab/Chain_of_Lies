#!/usr/bin/env python3
"""Fail-closed structural and adapter-checksum verification for the paper release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

TASKS = ("multiplication", "iterated_addition", "s5_state_tracking", "plain_parity")
MODELS = ("qwen", "llama")
SEEDS = (0, 1, 2)
CONDITIONS = {
    "vanilla",
    "filler",
    "public_only_cot",
    "piggyback",
    "invisible",
    "filler_plus_public_cot",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, check_hashes: bool) -> dict[str, object]:
    status = json.loads((root / "RELEASE_STATUS.json").read_text())
    rows = list(csv.DictReader((root / "adapter_manifest.csv").open()))
    expected = len(TASKS) * len(MODELS) * len(SEEDS) * len(CONDITIONS)
    if len(rows) != expected or status["final_adapter_count"] != expected:
        raise ValueError(
            f"Adapter count mismatch: manifest={len(rows)}, expected={expected}"
        )

    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        key = (row["task"], row["model"], int(row["seed"]), row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate adapter row: {key}")
        seen.add(key)
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Missing or wrong-sized adapter: {path}")
        if check_hashes and _sha256(path) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")

    for task in TASKS:
        for model in MODELS:
            for seed in SEEDS:
                cell = root / "tasks" / task / "models" / model / f"seed_{seed}"
                actual = {path.name for path in cell.iterdir() if path.is_dir()}
                if actual != CONDITIONS:
                    raise ValueError(f"Condition mismatch in {cell}: {sorted(actual)}")

    required = (
        "README.md",
        "documentation/SCALING_REGIME_APPENDIX_HANDOFF.md",
        "documentation/Figures/figure_accuracy_plot_data.csv",
        "documentation/Figures/figure_chain_plot_data.csv",
        "documentation/Figures/figure_accuracy_by_length_appendix.pdf",
        "documentation/Figures/figure_chain_length_by_input_length_main.pdf",
        "integrity/filler_only_all_24_audit.json",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(
            f"Release must be self-contained; found symlinks: {symlinks[:5]}"
        )

    return {
        "complete": True,
        "adapter_count": len(rows),
        "adapter_hashes_checked": check_hashes,
        "tasks": list(TASKS),
        "conditions_per_model_seed": sorted(CONDITIONS),
        "full_cot_baseline": status["full_cot_baseline"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path(__file__).resolve().parent
    )
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), not args.skip_hashes), indent=2))


if __name__ == "__main__":
    main()
