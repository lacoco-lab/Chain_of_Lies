#!/usr/bin/env python3
"""Independently validate multiplication-only piggyback splits and leakage invariants."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.hard_regime.multiplication_only_piggyback.generate_splits import (
    CONTROL,
    PIGGYBACK,
    _digest_keys,
    load_config,
)
from experiments.hard_regime.range_calibration.generate_splits import (
    _audit_variant,
    _canonical_question,
    _wrap_shift,
)

QUESTION_RE = re.compile(r"^(\d+) \* (\d+)$")


def _records(root: Path, split: str, expected: int) -> list[dict[str, Any]]:
    paths = sorted((root / f"{split}_prompts").glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} records in {root}/{split}_prompts, found {len(paths)}.")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _parts(question: str, low: int, high: int) -> tuple[int, int]:
    match = QUESTION_RE.fullmatch(question)
    if match is None:
        raise RuntimeError(f"Question is not multiplication-only: {question!r}")
    left, right = (int(value) for value in match.groups())
    if not (low <= left <= high and low <= right <= high):
        raise RuntimeError(f"Operand outside {low}..{high}: {question!r}")
    return left, right


def _validate_record(
    record: dict[str, Any], variant: str, low: int, high: int, shifts: tuple[int, ...]
) -> None:
    if record.get("task_type") != "arithmetic":
        raise RuntimeError("Unexpected task type.")
    if record.get("difficulty_variant") != variant.removeprefix("arith_"):
        raise RuntimeError("Difficulty variant mismatch.")
    spec = record.get("spec") or {}
    if spec.get("question_format") != "multiplication_only":
        raise RuntimeError("Missing multiplication-only format marker.")
    public = str(spec.get("public_question"))
    private = str(spec.get("private_question"))
    public_parts = _parts(public, low, high)
    private_parts = _parts(private, low, high)
    if spec.get("public_answer") != public_parts[0] * public_parts[1]:
        raise RuntimeError("Incorrect public answer.")
    if spec.get("private_answer") != private_parts[0] * private_parts[1]:
        raise RuntimeError("Incorrect private answer.")
    prompt = str(record.get("prompt_text", ""))
    if f"Main question: {public}" not in prompt or f"Other question: {private}" not in prompt:
        raise RuntimeError("Prompt/spec mismatch.")
    if "adding the offset" in prompt.lower() or re.search(r"\d+ \* \d+ \+ \d+", prompt):
        raise RuntimeError("Offset language or an affine question leaked into the prompt.")
    if variant == PIGGYBACK:
        if public_parts[0] != private_parts[0]:
            raise RuntimeError("Piggyback pair does not share its multiplier.")
        allowed = {
            _wrap_shift(public_parts[1], shift, low, high)
            for shift in shifts
        }
        if private_parts[1] not in allowed:
            raise RuntimeError("Piggyback factor does not use a configured shift.")


def _validate_seed(
    config: dict[str, Any], data_root: Path, seed: int
) -> tuple[dict[str, Any], set[tuple[str, int, int, int]]]:
    low, high = (int(value) for value in config["operand_range"])
    section = config["piggyback"]
    shifts = tuple(int(value) for value in section["shifts"])
    expected = {"train": int(config["train_n"]), "val": int(config["val_n"])}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audits: dict[str, Any] = {}
    seed_root = data_root / f"seed_{seed}"
    seed_manifest_path = seed_root / "split_manifest.json"
    if not seed_manifest_path.exists():
        raise RuntimeError(f"Missing seed manifest: {seed_manifest_path}")
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    seed_required = {
        "experiment": config["experiment_name"],
        "seed": seed,
        "question_format": "multiplication_only",
        "public_records_matched_across_variants": True,
    }
    if any(seed_manifest.get(key) != value for key, value in seed_required.items()):
        raise RuntimeError(f"Frozen seed manifest fields changed in {seed_manifest_path}.")

    for variant in section["variants"]:
        variant_root = seed_root / variant
        audit = _audit_variant(variant_root, expected["train"], expected["val"])
        manifest_path = variant_root / "split_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "experiment": config["experiment_name"],
            "question_format": "multiplication_only",
            "operand_range": config["operand_range"],
            "seed": seed,
            "variant": variant,
            "train_n": config["train_n"],
            "val_n": config["val_n"],
            "train_val_individual_question_overlap": 0,
            "train_val_commutative_equivalent_overlap": 0,
            "train_val_exact_pair_overlap": 0,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"Frozen manifest fields changed in {manifest_path}.")
        for digest in (
            "train_question_digest_sha256", "val_question_digest_sha256",
            "train_record_digest_sha256", "val_record_digest_sha256",
        ):
            if manifest.get(digest) != audit.get(digest):
                raise RuntimeError(f"Digest mismatch for {digest} in {manifest_path}.")
        records[variant] = {}
        for split, count in expected.items():
            records[variant][split] = _records(variant_root, split, count)
            for record in records[variant][split]:
                _validate_record(record, variant, low, high, shifts)
        audits[variant] = audit

    for split in ("train", "val"):
        pig_public = [record["spec"]["public_question"] for record in records[PIGGYBACK][split]]
        control_public = [record["spec"]["public_question"] for record in records[CONTROL][split]]
        if pig_public != control_public:
            raise RuntimeError(f"Piggyback/control public records differ in seed {seed} {split}.")

    train_keys = {
        _canonical_question(record["spec"][field])
        for variant in section["variants"]
        for record in records[variant]["train"]
        for field in ("public_question", "private_question")
    }
    val_keys = {
        _canonical_question(record["spec"][field])
        for variant in section["variants"]
        for record in records[variant]["val"]
        for field in ("public_question", "private_question")
    }
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Suite-wide train/evaluation overlap in seed {seed}: {len(overlap)}")
    expected_train_keys = 3 * expected["train"]
    expected_val_keys = 3 * expected["val"]
    if len(train_keys) != expected_train_keys or len(val_keys) != expected_val_keys:
        raise RuntimeError("A component is reused across matched variants within a split.")
    return ({
        "valid": True,
        "seed": seed,
        "global_train_unique_questions": len(train_keys),
        "global_val_unique_questions": len(val_keys),
        "global_train_val_question_overlap": 0,
        "variant_audits": audits,
    }, train_keys | val_keys)


def validate(config_path: Path, data_root: Path, seed: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    global_manifest_path = data_root / "split_manifest.json"
    if not global_manifest_path.exists():
        raise RuntimeError(f"Missing global manifest: {global_manifest_path}")
    global_manifest = json.loads(global_manifest_path.read_text(encoding="utf-8"))
    global_required = {
        "experiment": config["experiment_name"],
        "question_format": "multiplication_only",
        "operand_range": config["operand_range"],
        "seeds": config["seeds"],
        "train_n": config["train_n"],
        "val_n": config["val_n"],
        "cross_seed_component_disjointness_required": True,
        "cross_seed_question_overlap": 0,
    }
    if any(global_manifest.get(key) != value for key, value in global_required.items()):
        raise RuntimeError(f"Frozen global manifest fields changed in {global_manifest_path}.")
    seeds = [seed] if seed is not None else [int(value) for value in config["seeds"]]
    if any(value not in config["seeds"] for value in seeds):
        raise ValueError(f"Unconfigured seed requested: {seed}")
    reports: dict[str, Any] = {}
    keyed_seeds: list[tuple[int, set[tuple[str, int, int, int]]]] = []
    for current_seed in seeds:
        seed_manifest_path = data_root / f"seed_{current_seed}" / "split_manifest.json"
        seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
        if global_manifest.get("seed_manifests", {}).get(str(current_seed)) != seed_manifest:
            raise RuntimeError(f"Seed {current_seed} manifest does not match the global manifest.")
        report, keys = _validate_seed(config, data_root, current_seed)
        reports[f"seed_{current_seed}"] = report
        keyed_seeds.append((current_seed, keys))
    if seed is None:
        for index, (left_seed, left) in enumerate(keyed_seeds):
            for right_seed, right in keyed_seeds[index + 1 :]:
                overlap = left & right
                if overlap:
                    raise RuntimeError(
                        f"Cross-seed component overlap in seeds {left_seed}/{right_seed}: {len(overlap)}"
                    )
        all_keys = set().union(*(keys for _, keys in keyed_seeds))
        if global_manifest.get("cross_seed_unique_questions") != len(all_keys):
            raise RuntimeError("Global unique-question count does not match regenerated records.")
        if global_manifest.get("cross_seed_question_digest_sha256") != _digest_keys(all_keys):
            raise RuntimeError("Global question digest does not match regenerated records.")
    result = {
        "valid": True,
        "experiment": config["experiment_name"],
        "question_format": "multiplication_only",
        "operand_range": config["operand_range"],
        "validated_seeds": seeds,
        "cross_seed_component_overlap": 0 if seed is None else "certified_by_global_manifest",
        "seed_reports": reports,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate(args.config, args.data_root, args.seed)


if __name__ == "__main__":
    main()
