#!/usr/bin/env python3
"""Generate strictly component-disjoint one-seed Hard-regime range sweeps."""

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

from chain_of_lies.variants.arithmetic.data_generation.generate import (  # noqa: E402
    build_arithmetic_prompt_text,
)
from chain_of_lies.variants.steganography.data_generation.generate import (  # noqa: E402
    SCHEME_LOCAL_INVISIBLE,
    STEG_VARIANT_LOCAL_INVISIBLE,
    aligned_private_payload,
    build_steganography_prompt_text,
    validate_steganographic_target,
)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}.")
    return config


def range_config(config: dict[str, Any], family: str, range_id: str) -> dict[str, Any]:
    for item in config[family]["ranges_hard_to_easy"]:
        if item["id"] == range_id:
            return item
    raise ValueError(f"Unknown {family} range {range_id!r}.")


def _stable_rng(seed: int, family: str, range_id: str) -> random.Random:
    payload = f"{seed}:{family}:{range_id}:strict-v1".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _clear_prompts(path: Path) -> None:
    if not path.exists():
        return
    for pattern in ("*.json", ".*.json.tmp"):
        for item in path.glob(pattern):
            item.unlink()


def _wrap_shift(value: int, delta: int, low: int, high: int) -> int:
    return low + ((value - low + delta) % (high - low + 1))


def _linear_task(triple: tuple[int, int, int]) -> tuple[str, int]:
    multiplier, factor, offset = triple
    return f"{multiplier} * {factor} + {offset}", multiplier * factor + offset


def _multiplication_task(pair: tuple[int, int]) -> tuple[str, int]:
    multiplier, factor = pair
    return f"{multiplier} * {factor}", multiplier * factor


def _canonical_question(question: str) -> tuple[str, int, int, int]:
    """Canonicalize commutative multiplication so reversed operands cannot leak."""
    if " + " in question:
        product, offset_text = question.rsplit(" + ", 1)
        left_text, right_text = product.split(" * ", 1)
        left, right, offset = int(left_text), int(right_text), int(offset_text)
        return ("linear", min(left, right), max(left, right), offset)
    left_text, right_text = question.split(" * ", 1)
    left, right = int(left_text), int(right_text)
    return ("multiplication", min(left, right), max(left, right), 0)


def _question_digest(questions: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(questions)).encode("utf-8")).hexdigest()


def _record_digest(records: list[tuple[str, dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for name, record in records:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _arithmetic_record(
    variant: str,
    experiment_id: str,
    public_task: tuple[str, int],
    private_task: tuple[str, int],
) -> dict[str, Any]:
    return {
        "task_type": "arithmetic",
        "experiment_id": experiment_id,
        "difficulty_variant": variant.removeprefix("arith_"),
        "prompt_text": build_arithmetic_prompt_text(public_task[0], private_task[0]),
        "spec": {
            "public_question": public_task[0],
            "private_question": private_task[0],
            "public_answer": public_task[1],
            "private_answer": private_task[1],
        },
    }


def _steganography_record(
    experiment_id: str,
    public_task: tuple[str, int],
    private_task: tuple[str, int],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_type": "arithmetic",
        "experiment_id": experiment_id,
        "difficulty_variant": STEG_VARIANT_LOCAL_INVISIBLE.removeprefix("arith_"),
        "prompt_text": build_steganography_prompt_text(
            public_task[0],
            private_task[0],
            scheme=SCHEME_LOCAL_INVISIBLE,
        ),
        "spec": {
            "public_question": public_task[0],
            "private_question": private_task[0],
            "public_answer": public_task[1],
            "private_answer": private_task[1],
            "steganography_scheme": SCHEME_LOCAL_INVISIBLE,
            "steganography_payload_source": "aligned_private_trace_values",
        },
    }
    payload = aligned_private_payload(record)
    record["spec"]["steganography_payload"] = payload
    record["spec"]["steganography_payload_value_count"] = len(payload.split("|"))
    validate_steganographic_target(record)
    return record


def _audit_variant(variant_root: Path, train_n: int, val_n: int) -> dict[str, Any]:
    split_questions: dict[str, set[str]] = {}
    split_question_keys: dict[str, set[tuple[str, int, int, int]]] = {}
    split_pairs: dict[str, set[tuple[str, str]]] = {}
    for split, expected in (("train", train_n), ("val", val_n)):
        paths = sorted((variant_root / f"{split}_prompts").glob("*.json"))
        if len(paths) != expected:
            raise RuntimeError(
                f"Expected {expected} {split} records in {variant_root}, found {len(paths)}."
            )
        questions: list[str] = []
        pairs: list[tuple[str, str]] = []
        records: list[tuple[str, dict[str, Any]]] = []
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append((path.name, record))
            spec = record["spec"]
            questions.extend((spec["public_question"], spec["private_question"]))
            pairs.append((spec["public_question"], spec["private_question"]))
        if len(set(questions)) != 2 * expected:
            raise RuntimeError(f"{variant_root} {split} reuses an individual question.")
        question_keys = {_canonical_question(question) for question in questions}
        if len(question_keys) != 2 * expected:
            raise RuntimeError(
                f"{variant_root} {split} reuses a commutative-equivalent question."
            )
        if len(set(pairs)) != expected:
            raise RuntimeError(f"{variant_root} {split} reuses an exact pair.")
        split_questions[split] = set(questions)
        split_question_keys[split] = question_keys
        split_pairs[split] = set(pairs)
        if split == "train":
            train_record_digest = _record_digest(records)
        else:
            val_record_digest = _record_digest(records)
    question_overlap = split_questions["train"] & split_questions["val"]
    pair_overlap = split_pairs["train"] & split_pairs["val"]
    equivalent_overlap = split_question_keys["train"] & split_question_keys["val"]
    if question_overlap or equivalent_overlap or pair_overlap:
        raise RuntimeError(
            f"Strict split failure for {variant_root}: question_overlap={len(question_overlap)}, "
            f"commutative_equivalent_overlap={len(equivalent_overlap)}, pair_overlap={len(pair_overlap)}."
        )
    return {
        "train_unique_individual_questions": len(split_questions["train"]),
        "val_unique_individual_questions": len(split_questions["val"]),
        "train_val_individual_question_overlap": 0,
        "train_val_commutative_equivalent_overlap": 0,
        "train_val_exact_pair_overlap": 0,
        "train_question_digest_sha256": _question_digest(split_questions["train"]),
        "val_question_digest_sha256": _question_digest(split_questions["val"]),
        "train_record_digest_sha256": train_record_digest,
        "val_record_digest_sha256": val_record_digest,
    }


def generate_piggyback(
    config: dict[str, Any], output_root: Path, range_id: str
) -> dict[str, Any]:
    section = config["piggyback"]
    selected_range = range_config(config, "piggyback", range_id)
    low, high = (int(value) for value in selected_range["operand_range"])
    offset_low, offset_high = (int(value) for value in section["offset_range"])
    shifts = tuple(int(value) for value in section["shifts"])
    train_n, val_n = int(config["train_n"]), int(config["val_n"])
    total = train_n + val_n
    rng = _stable_rng(int(config["seed"]), "piggyback", range_id)
    used_questions: set[tuple[str, int, int, int]] = set()
    examples: list[tuple[tuple[str, int], tuple[str, int], tuple[str, int]]] = []
    max_attempts = total * 500
    for _attempt in range(max_attempts):
        if len(examples) == total:
            break
        public_triple = (
            rng.randint(low, high),
            rng.randint(low, high),
            rng.randint(offset_low, offset_high),
        )
        public_task = _linear_task(public_triple)
        private_factor = _wrap_shift(public_triple[1], rng.choice(shifts), low, high)
        private_offset = _wrap_shift(
            public_triple[2], rng.choice(shifts), offset_low, offset_high
        )
        piggy_task = _linear_task((public_triple[0], private_factor, private_offset))
        if piggy_task[0] == public_task[0] or piggy_task[1] == public_task[1]:
            continue
        control_task = _linear_task(
            (
                rng.randint(low, high),
                rng.randint(low, high),
                rng.randint(offset_low, offset_high),
            )
        )
        if control_task[0] == public_task[0] or control_task[1] == public_task[1]:
            continue
        candidate_questions = {public_task[0], piggy_task[0], control_task[0]}
        candidate_keys = {
            _canonical_question(question) for question in candidate_questions
        }
        if (
            len(candidate_questions) != 3
            or len(candidate_keys) != 3
            or candidate_keys & used_questions
        ):
            continue
        used_questions.update(candidate_keys)
        examples.append((public_task, piggy_task, control_task))
    if len(examples) != total:
        raise RuntimeError(
            f"Could not generate {total} strict piggyback examples for {range_id}."
        )

    range_root = output_root / "piggyback" / range_id / f"seed_{config['seed']}"
    manifests: dict[str, Any] = {}
    for variant in section["variants"]:
        variant_root = range_root / variant
        for split, start, count in (("train", 0, train_n), ("val", train_n, val_n)):
            prompt_dir = variant_root / f"{split}_prompts"
            _clear_prompts(prompt_dir)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            for index, (public_task, piggy_task, control_task) in enumerate(
                examples[start : start + count]
            ):
                private_task = (
                    piggy_task if variant == "arith_piggyback" else control_task
                )
                experiment_id = f"arith_{split}_{index:05d}"
                _atomic_json(
                    prompt_dir / f"{experiment_id}.json",
                    _arithmetic_record(
                        variant, experiment_id, public_task, private_task
                    ),
                )
        audit = _audit_variant(variant_root, train_n, val_n)
        manifest = {
            "schema_version": 1,
            "family": "piggyback",
            "range_id": range_id,
            "operand_range": [low, high],
            "offset_range": [offset_low, offset_high],
            "seed": config["seed"],
            "variant": variant,
            "train_n": train_n,
            "val_n": val_n,
            "public_records_matched_across_variants": True,
            "strict_component_disjointness_required": True,
            **audit,
        }
        _atomic_json(variant_root / "split_manifest.json", manifest)
        manifests[variant] = manifest

    # Public questions are intentionally identical between piggyback and control,
    # but no question from either training variant appears in either validation variant.
    train_global = {
        _canonical_question(question)
        for public_task, piggy_task, control_task in examples[:train_n]
        for question in (public_task[0], piggy_task[0], control_task[0])
    }
    val_global = {
        _canonical_question(question)
        for public_task, piggy_task, control_task in examples[train_n:]
        for question in (public_task[0], piggy_task[0], control_task[0])
    }
    if train_global & val_global:
        raise RuntimeError("Cross-variant train/validation component overlap detected.")
    suite_manifest = {
        "family": "piggyback",
        "range_id": range_id,
        "seed": config["seed"],
        "global_train_unique_questions": len(train_global),
        "global_val_unique_questions": len(val_global),
        "global_train_val_question_overlap": 0,
        "global_train_val_commutative_equivalent_overlap": 0,
        "variant_manifests": manifests,
    }
    _atomic_json(range_root / "split_manifest.json", suite_manifest)
    return suite_manifest


def generate_steganography(
    config: dict[str, Any], output_root: Path, range_id: str
) -> dict[str, Any]:
    section = config["steganography"]
    selected_range = range_config(config, "steganography", range_id)
    low, high = (int(value) for value in selected_range["operand_range"])
    train_n, val_n = int(config["train_n"]), int(config["val_n"])
    total = train_n + val_n
    rng = _stable_rng(int(config["seed"]), "steganography", range_id)
    used_questions: set[tuple[str, int, int, int]] = set()
    examples: list[tuple[tuple[str, int], tuple[str, int]]] = []
    max_attempts = total * 1000
    for _attempt in range(max_attempts):
        if len(examples) == total:
            break
        public_task = _multiplication_task(
            (rng.randint(low, high), rng.randint(low, high))
        )
        private_task = _multiplication_task(
            (rng.randint(low, high), rng.randint(low, high))
        )
        candidate_questions = {public_task[0], private_task[0]}
        candidate_keys = {
            _canonical_question(question) for question in candidate_questions
        }
        if (
            len(candidate_questions) != 2
            or len(candidate_keys) != 2
            or candidate_keys & used_questions
        ):
            continue
        used_questions.update(candidate_keys)
        examples.append((public_task, private_task))
    if len(examples) != total:
        raise RuntimeError(
            f"Could not generate {total} strict steganography examples for {range_id}."
        )

    variant = section["variant"]
    variant_root = (
        output_root / "steganography" / range_id / f"seed_{config['seed']}" / variant
    )
    for split, start, count in (("train", 0, train_n), ("val", train_n, val_n)):
        prompt_dir = variant_root / f"{split}_prompts"
        _clear_prompts(prompt_dir)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for index, (public_task, private_task) in enumerate(
            examples[start : start + count]
        ):
            experiment_id = f"steg_{split}_{index:05d}"
            _atomic_json(
                prompt_dir / f"{experiment_id}.json",
                _steganography_record(experiment_id, public_task, private_task),
            )
    audit = _audit_variant(variant_root, train_n, val_n)
    manifest = {
        "schema_version": 1,
        "family": "steganography",
        "range_id": range_id,
        "operand_range": [low, high],
        "seed": config["seed"],
        "variant": variant,
        "train_n": train_n,
        "val_n": val_n,
        "public_private_sampled_independently": True,
        "strict_component_disjointness_required": True,
        **audit,
    }
    _atomic_json(variant_root / "split_manifest.json", manifest)
    return manifest


def generate(
    config_path: Path, output_root: Path, family: str, range_id: str
) -> dict[str, Any]:
    config = load_config(config_path)
    if family == "piggyback":
        return generate_piggyback(config, output_root, range_id)
    if family == "steganography":
        return generate_steganography(config, output_root, range_id)
    raise ValueError(f"Unsupported family {family!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=("piggyback", "steganography"), required=True
    )
    parser.add_argument("--range", dest="range_id", required=True)
    args = parser.parse_args()
    manifest = generate(args.config, args.output_root, args.family, args.range_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
