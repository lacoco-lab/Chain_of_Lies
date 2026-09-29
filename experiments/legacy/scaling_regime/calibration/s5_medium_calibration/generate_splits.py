#!/usr/bin/env python3
"""Generate the balanced mixed-length seed-0 S5 Medium calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.variants.s5.data_generation.generate import (  # noqa: E402
    S5_INITIAL_STATE,
    S5_SYMBOLS,
    S5_VARIANT_CONTROL,
    S5Pair,
    build_s5_prompt_text,
    pair_to_spec,
    replay_sequence,
    sample_s5_control_pair,
)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("variant") != S5_VARIANT_CONTROL:
        raise ValueError("Medium calibration must use the independent S5 control.")
    lengths = [int(value) for value in config["lengths"]]
    if not lengths or len(lengths) != len(set(lengths)) or min(lengths) <= 0:
        raise ValueError("Calibration lengths must be distinct positive integers.")
    if (
        config.get("initial_state_protocol")
        != "uniform_random_permutation_shared_by_public_and_private"
    ):
        raise ValueError(
            "Medium calibration requires one shared random initial state per pair."
        )
    return config


def _random_initial_state(rng: random.Random) -> str:
    return "".join(rng.sample(S5_SYMBOLS, len(S5_SYMBOLS)))


def _apply_initial_state(pair: S5Pair, initial_state: str) -> S5Pair:
    public_final, _ = replay_sequence(pair.public_sequence, initial_state=initial_state)
    private_final, _ = replay_sequence(
        pair.private_sequence, initial_state=initial_state
    )
    return replace(
        pair, public_final_state=public_final, private_final_state=private_final
    )


def _prompt(pair: S5Pair, initial_state: str) -> str:
    prompt = build_s5_prompt_text(pair.public_sequence, pair.private_sequence)
    fixed = f"Each arrangement starts as {S5_INITIAL_STATE}."
    if prompt.count(fixed) != 1:
        raise RuntimeError("Historical S5 prompt template changed unexpectedly.")
    return prompt.replace(fixed, f"Each arrangement starts as {initial_state}.", 1)


def _key(record: dict[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    spec = record["spec"]
    return (
        spec["initial_state"],
        tuple(spec["public_instruction_sequence"]),
        tuple(spec["private_instruction_sequence"]),
    )


def _payload(pair: S5Pair, initial_state: str, length: int) -> dict[str, Any]:
    spec = pair_to_spec(pair)
    spec.update(
        {
            "initial_state": initial_state,
            "public_initial_state": initial_state,
            "private_initial_state": initial_state,
            "initial_state_protocol": "uniform_random_permutation_shared_by_public_and_private",
            "calibration_length": length,
            "calibration_family": "s5_medium",
        }
    )
    return {
        "task_type": "s5",
        "difficulty_variant": S5_VARIANT_CONTROL,
        "calibration_experiment": "s5_medium_calibration",
        "prompt_text": _prompt(pair, initial_state),
        "spec": spec,
    }


def _sample_unique_for_length(
    *,
    rng: random.Random,
    length: int,
    count: int,
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(100_000, count * 100)
    while len(records) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not sample {count} unique length-{length} pairs."
            )
        initial_state = _random_initial_state(rng)
        pair = sample_s5_control_pair(rng, length_range=(length, length))
        pair = _apply_initial_state(pair, initial_state)
        record = _payload(pair, initial_state, length)
        key = _key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def _round_robin(
    by_length: dict[int, list[dict[str, Any]]], lengths: list[int]
) -> list[dict[str, Any]]:
    per_length = {len(by_length[length]) for length in lengths}
    if len(per_length) != 1:
        raise ValueError(
            "Round-robin serialization requires balanced exact-length buckets."
        )
    count = next(iter(per_length))
    return [by_length[length][index] for index in range(count) for length in lengths]


def _digest(records: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    seed = int(config["seed"])
    lengths = [int(value) for value in config["lengths"]]
    train_per_length = int(config["train_per_length"])
    val_per_length = int(config["validation_per_length"])
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    train_by_length: dict[int, list[dict[str, Any]]] = {}
    val_by_length: dict[int, list[dict[str, Any]]] = {}

    for length in lengths:
        rng = random.Random(seed * 10_000_019 + length * 1_000_003)
        combined = _sample_unique_for_length(
            rng=rng,
            length=length,
            count=train_per_length + val_per_length,
            seen=seen,
        )
        train_by_length[length] = combined[:train_per_length]
        val_by_length[length] = combined[train_per_length:]

    train_records = _round_robin(train_by_length, lengths)
    val_records = _round_robin(val_by_length, lengths)
    variant_root = output_root / f"seed_{seed}" / S5_VARIANT_CONTROL
    for split, records in (("train", train_records), ("val", val_records)):
        prompt_dir = variant_root / f"{split}_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for old in prompt_dir.glob("*.json"):
            old.unlink()
        for index, record in enumerate(records):
            length = int(record["spec"]["calibration_length"])
            experiment_id = f"s5_{split}_{index:05d}_L{length:02d}"
            record = {**record, "experiment_id": experiment_id}
            (prompt_dir / f"{experiment_id}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    manifest = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "variant": S5_VARIANT_CONTROL,
        "lengths": lengths,
        "train_per_length": train_per_length,
        "validation_per_length": val_per_length,
        "train_n": len(train_records),
        "val_n": len(val_records),
        "initial_state_protocol": config["initial_state_protocol"],
        "sequence_protocol": config["sequence_protocol"],
        "train_val_exact_pair_disjoint": True,
        "individual_state_swap_primitives_may_repeat_across_splits": True,
        "validation_digest_sha256": _digest(val_records),
        "num_unique_task_pairs": len(seen),
    }
    (variant_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated_data/s5_medium_calibration"),
    )
    args = parser.parse_args()
    generate(args.config, args.output_root)


if __name__ == "__main__":
    main()
