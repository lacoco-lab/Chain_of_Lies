#!/usr/bin/env python3
"""Fail-closed validation of five-fact knowledge Hard-regime splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.variants.knowledge.data_generation.generate import (
    ATOMIC_NUMBER_FACTS,
)
from chain_of_lies.variants.steganography.hard_task_channels import validate_target
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.knowledge.generate_splits import (
    CONTROL,
    PIGGYBACK,
    STEG,
    load_config,
)

FACTS = dict(ATOMIC_NUMBER_FACTS)


def _records(root: Path, split: str, count: int) -> list[dict[str, Any]]:
    paths = sorted((root / f"{split}_prompts").glob("*.json"))
    if len(paths) != count:
        raise RuntimeError(
            f"Expected {count} records in {root}/{split}_prompts; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _key(record: dict[str, Any], side: str) -> tuple[str, ...]:
    return tuple(
        sorted(str(fact["entity"]) for fact in record["spec"][f"{side}_facts"])
    )


def _check(record: dict[str, Any], variant: str) -> None:
    if record.get("task_type") != "knowledge" or record.get("variant_name") != variant:
        raise RuntimeError("Knowledge record identity mismatch.")
    spec = record["spec"]
    for side in ("public", "private"):
        facts = spec[f"{side}_facts"]
        if len(facts) != 5 or len({fact["entity"] for fact in facts}) != 5:
            raise RuntimeError(
                "A knowledge question does not contain five distinct facts."
            )
        for fact in facts:
            if FACTS.get(fact["entity"]) != fact["fact_value"]:
                raise RuntimeError("Incorrect atomic-number fact.")
        if sum(int(fact["fact_value"]) for fact in facts) != spec[f"{side}_answer"]:
            raise RuntimeError("Incorrect five-fact answer.")
    if variant == PIGGYBACK:
        public = spec["public_facts"]
        private = spec["private_facts"]
        if public[:4] != private[:4] or public[4] == private[4]:
            raise RuntimeError(
                "Knowledge piggyback must share four facts and replace only the fifth."
            )
    if variant == STEG:
        validate_target(record)


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    global_manifest = json.loads(
        (data_root / "split_manifest.json").read_text(encoding="utf-8")
    )
    required_global = {
        "experiment": config["experiment_name"],
        "seeds": config["seeds"],
        "train_n": config["train_n"],
        "val_n": config["val_n"],
        "facts_per_question": 5,
        "fact_bank": config["fact_bank"],
        "cross_seed_component_overlap": 0,
    }
    if any(global_manifest.get(key) != value for key, value in required_global.items()):
        raise RuntimeError("Global knowledge leakage contract is missing.")
    seeds = [seed] if seed is not None else config["seeds"]
    all_keys: set[tuple[str, ...]] = set()
    reports = {}
    for current_seed in seeds:
        split_keys: dict[str, set[Any]] = {"train": set(), "val": set()}
        by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for variant in (PIGGYBACK, CONTROL, STEG):
            by_variant[variant] = {}
            for split, count in (
                ("train", config["train_n"]),
                ("val", config["val_n"]),
            ):
                records = _records(
                    data_root / f"seed_{current_seed}" / variant, split, int(count)
                )
                by_variant[variant][split] = records
                for record in records:
                    _check(record, variant)
                    split_keys[split].update(
                        (_key(record, "public"), _key(record, "private"))
                    )
        expected_split_components = {
            "train": 5 * int(config["train_n"]),
            "val": 5 * int(config["val_n"]),
        }
        for split, expected_count in expected_split_components.items():
            if len(split_keys[split]) != expected_count:
                raise RuntimeError(
                    f"Expected {expected_count} unique knowledge questions in seed {current_seed} {split}; "
                    f"found {len(split_keys[split])}."
                )
        for split in ("train", "val"):
            pig = by_variant[PIGGYBACK][split]
            control = by_variant[CONTROL][split]
            if [(_key(r, "public"), r["spec"]["public_answer"]) for r in pig] != [
                (_key(r, "public"), r["spec"]["public_answer"]) for r in control
            ]:
                raise RuntimeError(
                    "Piggyback/control public knowledge records are not matched."
                )
        if split_keys["train"] & split_keys["val"]:
            raise RuntimeError(
                f"Knowledge train/eval component overlap in seed {current_seed}."
            )
        seed_keys = split_keys["train"] | split_keys["val"]
        if seed_keys & all_keys:
            raise RuntimeError(
                f"Knowledge cross-seed component overlap at seed {current_seed}."
            )
        all_keys.update(seed_keys)
        reports[str(current_seed)] = {
            "train_unique_components": len(split_keys["train"]),
            "val_unique_components": len(split_keys["val"]),
            "train_val_overlap": 0,
        }
    if seed is None:
        expected_global = (
            5 * (int(config["train_n"]) + int(config["val_n"])) * len(config["seeds"])
        )
        if len(all_keys) != expected_global:
            raise RuntimeError(
                f"Expected {expected_global} global knowledge questions; found {len(all_keys)}."
            )
        digest = hashlib.sha256(
            "\n".join(sorted(repr(key) for key in all_keys)).encode()
        ).hexdigest()
        if len(all_keys) != global_manifest.get(
            "unique_components"
        ) or digest != global_manifest.get("component_digest_sha256"):
            raise RuntimeError("Knowledge global component count/digest mismatch.")
    result = {
        "valid": True,
        "validated_seeds": seeds,
        "cross_seed_component_overlap": 0,
        "seed_reports": reports,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate(args.config, args.data_root, args.seed)
