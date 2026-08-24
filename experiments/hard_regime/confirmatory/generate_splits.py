#!/usr/bin/env python3
"""Generate the final fixed 500..1000 invisible-steganography suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.hard_regime.range_calibration.generate_splits import (
    _audit_variant,
    _canonical_question,
    _clear_prompts,
    _steganography_record,
)


STEGANOGRAPHY = "arith_steg_local_invisible"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}.")
    if config.get("operand_range") != [500, 1000]:
        raise ValueError("The confirmatory Hard regime is frozen to operands 500..1000.")
    if config.get("seeds") != [0, 1, 2]:
        raise ValueError("The confirmatory seeds are frozen to 0, 1, and 2.")
    if config.get("calibration_exclusion", {}).get("seed") != 314159:
        raise ValueError("The frozen range-selection exclusion seed is 314159.")
    return config


def _rng(seed: int, family: str) -> random.Random:
    material = f"hard-confirmatory-v1:{family}:{seed}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _calibration_rng(family: str) -> random.Random:
    """Reproduce the frozen r500_1000 calibration stream for exclusion."""
    material = f"314159:{family}:r500_1000:strict-v1".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _multiplication(parts: tuple[int, int]) -> tuple[str, int]:
    left, right = parts
    return f"{left} * {right}", left * right


def _digest_questions(keys: set[tuple[str, int, int, int]]) -> str:
    encoded = "\n".join(repr(key) for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_reserved_keys(config: dict[str, Any], family: str) -> set[tuple[str, int, int, int]]:
    """Return every component used to select the range at seed 314159."""
    low, high = (int(value) for value in config["operand_range"])
    total = int(config["train_n"]) + int(config["val_n"])
    rng = _calibration_rng(family)
    used: set[tuple[str, int, int, int]] = set()
    accepted = 0
    if family == "steganography":
        for _attempt in range(total * 2000):
            if accepted == total:
                break
            public = _multiplication((rng.randint(low, high), rng.randint(low, high)))
            private = _multiplication((rng.randint(low, high), rng.randint(low, high)))
            questions = {public[0], private[0]}
            keys = {_canonical_question(question) for question in questions}
            if len(questions) != 2 or len(keys) != 2 or keys & used:
                continue
            used.update(keys)
            accepted += 1
    else:
        raise ValueError(f"Unknown family: {family}")
    if accepted != total:
        raise RuntimeError(f"Could not reconstruct the calibration exclusion set for {family}.")
    # Small unit-test configs deliberately change the split sizes; the full
    # experiment must reproduce the frozen calibration exclusion digest.
    if int(config["train_n"]) == 10000 and int(config["val_n"]) == 1000:
        expected = config["calibration_exclusion"][f"{family}_question_digest_sha256"]
        observed = _digest_questions(used)
        if observed != expected:
            raise RuntimeError(
                f"Calibration exclusion reconstruction drifted for {family}: "
                f"expected {expected}, got {observed}."
            )
    return used


def _write_variant_manifest(
    *,
    config: dict[str, Any],
    family: str,
    seed: int,
    variant_root: Path,
    variant: str,
) -> dict[str, Any]:
    audit = _audit_variant(variant_root, int(config["train_n"]), int(config["val_n"]))
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "family": family,
        "operand_range": config["operand_range"],
        "seed": seed,
        "variant": variant,
        "train_n": config["train_n"],
        "val_n": config["val_n"],
        "strict_component_disjointness_required": True,
        "cross_seed_component_disjointness_required": True,
        **audit,
    }
    _atomic_json(variant_root / "split_manifest.json", manifest)
    return manifest


def generate_steganography(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    low, high = (int(value) for value in config["operand_range"])
    section = config["steganography"]
    train_n, val_n = int(config["train_n"]), int(config["val_n"])
    total = train_n + val_n
    calibration_keys = calibration_reserved_keys(config, "steganography")
    globally_used = set(calibration_keys)
    confirmatory_keys: set[tuple[str, int, int, int]] = set()
    seed_manifests: dict[str, Any] = {}

    for seed in config["seeds"]:
        rng = _rng(int(seed), "steganography")
        examples: list[tuple[tuple[str, int], tuple[str, int]]] = []
        for _attempt in range(total * 2000):
            if len(examples) == total:
                break
            public = _multiplication((rng.randint(low, high), rng.randint(low, high)))
            private = _multiplication((rng.randint(low, high), rng.randint(low, high)))
            questions = {public[0], private[0]}
            keys = {_canonical_question(question) for question in questions}
            if len(questions) != 2 or len(keys) != 2 or keys & globally_used:
                continue
            globally_used.update(keys)
            confirmatory_keys.update(keys)
            examples.append((public, private))
        if len(examples) != total:
            raise RuntimeError(f"Could not generate {total} strict steganography examples for seed {seed}.")

        variant = str(section["variant"])
        variant_root = output_root / "steganography" / f"seed_{seed}" / variant
        for split, start, count in (("train", 0, train_n), ("val", train_n, val_n)):
            prompt_dir = variant_root / f"{split}_prompts"
            _clear_prompts(prompt_dir)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            for index, (public, private) in enumerate(examples[start : start + count]):
                experiment_id = f"steg_{split}_{index:05d}"
                _atomic_json(
                    prompt_dir / f"{experiment_id}.json",
                    _steganography_record(experiment_id, public, private),
                )
        variant_manifest = _write_variant_manifest(
            config=config,
            family="steganography",
            seed=int(seed),
            variant_root=variant_root,
            variant=variant,
        )
        seed_root = output_root / "steganography" / f"seed_{seed}"
        seed_manifest = {"family": "steganography", "seed": seed, "variant_manifest": variant_manifest}
        _atomic_json(seed_root / "split_manifest.json", seed_manifest)
        seed_manifests[str(seed)] = seed_manifest

    return {
        "family": "steganography",
        "cross_seed_unique_questions": len(confirmatory_keys),
        "cross_seed_question_digest_sha256": _digest_questions(confirmatory_keys),
        "cross_seed_question_overlap": 0,
        "calibration_seed": 314159,
        "calibration_reserved_questions": len(calibration_keys),
        "calibration_confirmatory_question_overlap": 0,
        "seed_manifests": seed_manifests,
    }


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "operand_range": config["operand_range"],
        "seeds": config["seeds"],
        "strict_component_disjointness_required": True,
        "cross_seed_component_disjointness_required": True,
        "families": {"steganography": generate_steganography(config, output_root)},
    }
    _atomic_json(output_root / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
