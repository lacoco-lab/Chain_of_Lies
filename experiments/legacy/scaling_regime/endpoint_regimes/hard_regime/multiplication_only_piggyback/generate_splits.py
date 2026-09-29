#!/usr/bin/env python3
"""Generate fresh, globally disjoint multiplication-only piggyback splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.variants.arithmetic.data_generation.generate import (
    build_multiplication_prompt_text,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.range_calibration.generate_splits import (
    _audit_variant,
    _canonical_question,
    _clear_prompts,
    _wrap_shift,
)

PIGGYBACK = "arith_piggyback"
CONTROL = "arith_piggyback_control"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}.")
    if config.get("operand_range") != [500, 1000]:
        raise ValueError("The experiment is frozen to operands 500..1000.")
    if config.get("seeds") != [0, 1, 2]:
        raise ValueError("The experiment is frozen to seeds 0, 1, and 2.")
    section = config.get("piggyback", {})
    if (
        section.get("question_format") != "multiplication_only"
        or "offset_range" in section
    ):
        raise ValueError(
            "This experiment must contain multiplication only and no offset range."
        )
    if section.get("variants") != [PIGGYBACK, CONTROL]:
        raise ValueError("The frozen piggyback and matched-control variants changed.")
    if section.get("modes") != ["answer_only", "public_cot"]:
        raise ValueError("The frozen supervision modes changed.")
    return config


def _rng(seed: int) -> random.Random:
    material = f"hard-multiplication-only-piggyback-v1:{seed}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _multiplication(left: int, right: int) -> tuple[str, int]:
    return f"{left} * {right}", left * right


def _prompt_text(public_question: str, private_question: str) -> str:
    return build_multiplication_prompt_text(public_question, private_question)


def _record(
    variant: str,
    experiment_id: str,
    public: tuple[str, int],
    private: tuple[str, int],
) -> dict[str, Any]:
    return {
        "task_type": "arithmetic",
        "experiment_id": experiment_id,
        "difficulty_variant": variant.removeprefix("arith_"),
        "prompt_text": _prompt_text(public[0], private[0]),
        "spec": {
            "public_question": public[0],
            "private_question": private[0],
            "public_answer": public[1],
            "private_answer": private[1],
            "question_format": "multiplication_only",
        },
    }


def _digest_keys(keys: set[tuple[str, int, int, int]]) -> str:
    encoded = "\n".join(repr(key) for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _variant_manifest(
    config: dict[str, Any], seed: int, variant: str, variant_root: Path
) -> dict[str, Any]:
    audit = _audit_variant(variant_root, int(config["train_n"]), int(config["val_n"]))
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "family": "piggyback",
        "question_format": "multiplication_only",
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


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    low, high = (int(value) for value in config["operand_range"])
    section = config["piggyback"]
    shifts = tuple(int(value) for value in section["shifts"])
    train_n, val_n = int(config["train_n"]), int(config["val_n"])
    total = train_n + val_n
    globally_used: set[tuple[str, int, int, int]] = set()
    seed_manifests: dict[str, Any] = {}

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        rng = _rng(seed)
        examples: list[tuple[tuple[str, int], tuple[str, int], tuple[str, int]]] = []
        for _attempt in range(total * 2500):
            if len(examples) == total:
                break
            shared = rng.randint(low, high)
            public_factor = rng.randint(low, high)
            private_factor = _wrap_shift(public_factor, rng.choice(shifts), low, high)
            public = _multiplication(shared, public_factor)
            piggyback = _multiplication(shared, private_factor)
            control = _multiplication(rng.randint(low, high), rng.randint(low, high))
            questions = {public[0], piggyback[0], control[0]}
            keys = {_canonical_question(question) for question in questions}
            if (
                len(questions) != 3
                or len(keys) != 3
                or keys & globally_used
                or control[1] in {public[1], piggyback[1]}
            ):
                continue
            globally_used.update(keys)
            examples.append((public, piggyback, control))
        if len(examples) != total:
            raise RuntimeError(
                f"Could not generate {total} globally disjoint examples for seed {seed}; "
                f"accepted {len(examples)}."
            )

        seed_root = output_root / f"seed_{seed}"
        manifests: dict[str, Any] = {}
        for variant in section["variants"]:
            variant_root = seed_root / variant
            for split, start, count in (("train", 0, train_n), ("val", train_n, val_n)):
                prompt_dir = variant_root / f"{split}_prompts"
                _clear_prompts(prompt_dir)
                prompt_dir.mkdir(parents=True, exist_ok=True)
                for index, (public, piggyback, control) in enumerate(
                    examples[start : start + count]
                ):
                    private = piggyback if variant == PIGGYBACK else control
                    experiment_id = f"mul_{split}_{index:05d}"
                    _atomic_json(
                        prompt_dir / f"{experiment_id}.json",
                        _record(variant, experiment_id, public, private),
                    )
            manifests[variant] = _variant_manifest(config, seed, variant, variant_root)

        seed_manifest = {
            "schema_version": 1,
            "experiment": config["experiment_name"],
            "seed": seed,
            "question_format": "multiplication_only",
            "public_records_matched_across_variants": True,
            "variant_manifests": manifests,
        }
        _atomic_json(seed_root / "split_manifest.json", seed_manifest)
        seed_manifests[str(seed)] = seed_manifest

    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "question_format": "multiplication_only",
        "operand_range": config["operand_range"],
        "seeds": config["seeds"],
        "train_n": train_n,
        "val_n": val_n,
        "cross_seed_component_disjointness_required": True,
        "cross_seed_unique_questions": len(globally_used),
        "cross_seed_question_overlap": 0,
        "cross_seed_question_digest_sha256": _digest_keys(globally_used),
        "seed_manifests": seed_manifests,
    }
    _atomic_json(output_root / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
