#!/usr/bin/env python3
"""Generate the isolated random-initial replication of the original S5 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.variants.s5.data_generation.generate import (  # noqa: E402
    S5_INITIAL_STATE,
    S5_SYMBOLS,
    S5_VARIANT_CONTROL,
    S5_VARIANT_PIGGYBACK,
    S5Pair,
    build_s5_prompt_text,
    normalized_sequence_hamming_distance,
    pair_to_spec,
    replay_sequence,
    sample_s5_control_pair,
    sample_s5_piggyback_pair,
    sequence_hamming_distance,
)

VARIANTS = (S5_VARIANT_PIGGYBACK, S5_VARIANT_CONTROL)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("initial_state_protocol") != "uniform_random_permutation_shared_by_public_and_private":
        raise ValueError("This replication requires one shared random permutation per task pair.")
    if int(config["min_steps"]) != 10 or int(config["max_steps"]) != 19:
        raise ValueError("This replication intentionally preserves the original 10..19 step range.")
    return config


def _random_initial_state(rng: random.Random) -> str:
    return "".join(rng.sample(S5_SYMBOLS, len(S5_SYMBOLS)))


def _with_initial_state(pair: S5Pair, initial_state: str) -> S5Pair:
    public_final, _ = replay_sequence(pair.public_sequence, initial_state=initial_state)
    private_final, _ = replay_sequence(pair.private_sequence, initial_state=initial_state)
    return replace(
        pair,
        public_final_state=public_final,
        private_final_state=private_final,
    )


def _prompt_with_initial_state(pair: S5Pair, initial_state: str) -> str:
    prompt = build_s5_prompt_text(pair.public_sequence, pair.private_sequence)
    fixed_sentence = f"Each arrangement starts as {S5_INITIAL_STATE}."
    random_sentence = f"Each arrangement starts as {initial_state}."
    if prompt.count(fixed_sentence) != 1:
        raise RuntimeError("The historical S5 prompt template changed unexpectedly.")
    return prompt.replace(fixed_sentence, random_sentence, 1)


def _payload(variant: str, experiment_id: str, pair: S5Pair, initial_state: str) -> dict[str, Any]:
    spec = pair_to_spec(pair)
    spec.update(
        {
            "initial_state": initial_state,
            "public_initial_state": initial_state,
            "private_initial_state": initial_state,
            "initial_state_protocol": "uniform_random_permutation_shared_by_public_and_private",
        }
    )
    return {
        "task_type": "s5",
        "experiment_id": experiment_id,
        "difficulty_variant": variant,
        "replication": "s5_random_initial_replication",
        "prompt_text": _prompt_with_initial_state(pair, initial_state),
        "spec": spec,
    }


def _sampler(variant: str, length_range: tuple[int, int]) -> Callable[[random.Random], S5Pair]:
    if variant == S5_VARIANT_PIGGYBACK:
        return lambda rng: sample_s5_piggyback_pair(
            rng,
            length_range=length_range,
            change_range=(1, 1),
            change_position="last",
        )
    if variant == S5_VARIANT_CONTROL:
        return lambda rng: sample_s5_control_pair(rng, length_range=length_range)
    raise ValueError(f"Unsupported variant: {variant}")


def _record_key(record: dict[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    spec = record["spec"]
    return (
        spec["initial_state"],
        tuple(spec["public_instruction_sequence"]),
        tuple(spec["private_instruction_sequence"]),
    )


def _digest(records: list[dict[str, Any]]) -> str:
    canonical = "\n".join(json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(record["spec"]["sequence_length"]) for record in records]
    differences = [int(record["spec"]["num_differing_positions"]) for record in records]
    normalized = [float(record["spec"]["normalized_hamming_distance"]) for record in records]
    initial_counts = {
        state: sum(record["spec"]["initial_state"] == state for record in records)
        for state in sorted({record["spec"]["initial_state"] for record in records})
    }
    return {
        "avg_instruction_sequence_length": sum(lengths) / len(lengths),
        "min_instruction_sequence_length": min(lengths),
        "max_instruction_sequence_length": max(lengths),
        "avg_num_differing_positions": sum(differences) / len(differences),
        "avg_normalized_hamming_distance": sum(normalized) / len(normalized),
        "num_unique_initial_states": len(initial_counts),
        "initial_state_counts": initial_counts,
    }


def generate_variant(
    *,
    variant: str,
    seed: int,
    train_n: int,
    val_n: int,
    length_range: tuple[int, int],
    output_root: Path,
) -> dict[str, Any]:
    # Variant-specific seed streams make either variant independently reproducible.
    variant_offset = 0 if variant == S5_VARIANT_PIGGYBACK else 1_000_003
    rng = random.Random(seed + variant_offset)
    sample_pair = _sampler(variant, length_range)
    total = train_n + val_n
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    attempts = 0
    max_attempts = max(200_000, total * 50)

    while len(records) < total:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not generate {total} unique examples for {variant}.")
        initial_state = _random_initial_state(rng)
        pair = _with_initial_state(sample_pair(rng), initial_state)
        split_name = "train" if len(records) < train_n else "val"
        split_index = len(records) if split_name == "train" else len(records) - train_n
        record = _payload(variant, f"s5_{split_name}_{split_index:05d}", pair, initial_state)
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    variant_root = output_root / f"seed_{seed}" / variant
    train_records = records[:train_n]
    val_records = records[train_n:]
    for split_name, split_records in (("train", train_records), ("val", val_records)):
        prompt_dir = variant_root / f"{split_name}_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for old_path in prompt_dir.glob("*.json"):
            old_path.unlink()
        for record in split_records:
            (prompt_dir / f"{record['experiment_id']}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    manifest = {
        "experiment_name": "s5_random_initial_replication",
        "variant_name": variant,
        "seed": seed,
        "train_n": train_n,
        "val_n": val_n,
        "length_range": list(length_range),
        "initial_state_protocol": "uniform_random_permutation_shared_by_public_and_private",
        "piggyback_change_range": [1, 1] if variant == S5_VARIANT_PIGGYBACK else None,
        "piggyback_change_position": "last" if variant == S5_VARIANT_PIGGYBACK else None,
        "train_val_exact_task_disjoint": not bool(
            {_record_key(record) for record in train_records}
            & {_record_key(record) for record in val_records}
        ),
        "validation_digest_sha256": _digest(val_records),
        "diagnostics": {
            "all": _diagnostics(records),
            "train": _diagnostics(train_records),
            "val": _diagnostics(val_records),
        },
        "samples": records[:3],
    }
    (variant_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    default_config = Path(__file__).with_name("config.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated_data/s5_random_initial_replication"),
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        manifest = generate_variant(
            variant=variant,
            seed=args.seed,
            train_n=int(config["train_n"]),
            val_n=int(config["val_n"]),
            length_range=(int(config["min_steps"]), int(config["max_steps"])),
            output_root=args.output_root,
        )
        diagnostics = dict(manifest["diagnostics"]["all"])
        diagnostics.pop("initial_state_counts", None)
        print(
            f"[random-initial S5] generated {variant}, seed={args.seed}: "
            f"{json.dumps(diagnostics, sort_keys=True)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
