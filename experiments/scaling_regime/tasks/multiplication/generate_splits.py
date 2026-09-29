#!/usr/bin/env python3
"""Generate balanced, matched 2-4-digit multiplication splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
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
from chain_of_lies.variants.steganography.data_generation.generate import (
    SCHEME_LOCAL_INVISIBLE,
    aligned_private_payload,
    build_steganography_prompt_text,
    validate_steganographic_target,
)

CONTROL = "arith_length_control"
PIGGYBACK = "arith_length_piggyback"
STEG = "arith_length_steg_local_invisible"
VARIANTS = (CONTROL, PIGGYBACK, STEG)
Operands = tuple[int, int]
Row = tuple[Operands, Operands, Operands, int]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Multiplication length v1 requires schema 1.")
    digits = [int(value) for value in config.get("operand_digits", [])]
    if digits != [2, 3, 4]:
        raise ValueError(
            "Multiplication length v1 is frozen to exact digit lengths 2, 3, and 4."
        )
    expected_shifts = [-40, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 40]
    if config.get("piggyback_factor_shifts") != expected_shifts:
        raise ValueError(
            "Multiplication length v1 must preserve the frozen nearby-factor Piggy shifts."
        )
    if config.get("seeds") != [0, 1, 2]:
        raise ValueError("Multiplication length v1 requires seeds 0, 1, and 2.")
    expected_conditions = {
        "vanilla": (CONTROL, "answer_only"),
        "filler": (CONTROL, "filler_public_cot"),
        "public_only_cot": (CONTROL, "public_cot"),
        "piggyback": (PIGGYBACK, "public_cot"),
        "steganography": (STEG, "local_channel_cot"),
    }
    if set(config.get("conditions", {})) != set(expected_conditions):
        raise ValueError("Exactly the five paper conditions must be configured.")
    for condition, (variant, mode) in expected_conditions.items():
        spec = config["conditions"][condition]
        if (spec.get("variant"), spec.get("supervision_mode")) != (variant, mode):
            raise ValueError(f"Frozen condition mapping changed for {condition}.")
    if int(config["conditions"]["filler"].get("filler_token_count", 0)) != 64:
        raise ValueError(
            "Multiplication length v1 fixes Filler to 64 atomic token positions."
        )
    if (
        int(config["train_examples_per_length"]) <= 0
        or int(config["eval_examples_per_length"]) <= 0
    ):
        raise ValueError("Per-length counts must be positive.")
    return config


def roots(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = config["paths"]
    return (
        Path(paths["data_root"]),
        Path(paths["artifacts_root"]),
        Path(paths["responses_root"]),
    )


def _rng(label: str) -> random.Random:
    digest = hashlib.sha256(
        f"multiplication-length-balanced-v1:{label}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _bounds(digits: int) -> tuple[int, int]:
    return 10 ** (digits - 1), 10**digits - 1


def _key(operands: Operands) -> tuple[int, int]:
    return tuple(sorted(operands))


def _question(operands: Operands) -> str:
    return f"{operands[0]} * {operands[1]}"


def _wrap_shift(value: int, shift: int, low: int, high: int) -> int:
    """Apply the original Multiplication Piggy shift within an exact-digit range."""
    return low + ((value - low + shift) % (high - low + 1))


def _record(
    variant: str,
    experiment_id: str,
    public: Operands,
    private: Operands,
    *,
    piggyback_shift: int | None = None,
) -> dict[str, Any]:
    digits = len(str(private[0]))
    shared = len(set(public) & set(private))
    public_question, private_question = _question(public), _question(private)
    record: dict[str, Any] = {
        "schema_version": 1,
        "task_type": "arithmetic",
        "variant_name": variant,
        "difficulty_variant": variant.removeprefix("arith_"),
        "difficulty_regime": "balanced_length_retraining",
        "mechanism": (
            "steg_local_invisible"
            if variant == STEG
            else ("piggyback" if variant == PIGGYBACK else "control")
        ),
        "experiment_id": experiment_id,
        "prompt_text": (
            build_steganography_prompt_text(
                public_question, private_question, scheme=SCHEME_LOCAL_INVISIBLE
            )
            if variant == STEG
            else build_multiplication_prompt_text(public_question, private_question)
        ),
        "spec": {
            "public_question": public_question,
            "private_question": private_question,
            "public_answer": public[0] * public[1],
            "private_answer": private[0] * private[1],
            "public_operands": list(public),
            "private_operands": list(private),
            "question_format": "multiplication_only",
            "operand_digits": digits,
            "evaluation_length": digits,
            "shared_operand_count": shared,
            "shared_operand_fraction": shared / 2,
            "sampling_strategy": (
                "same_multiplier_and_small_wrapped_factor_shift"
                if variant == PIGGYBACK
                else "public_and_private_operand_values_disjoint"
            ),
        },
    }
    if variant == PIGGYBACK:
        if piggyback_shift is None:
            raise ValueError(
                "Piggy records require their configured public-to-private factor shift."
            )
        record["spec"]["configured_public_to_private_factor_shift"] = piggyback_shift
    if variant == STEG:
        record["spec"]["steganography_scheme"] = SCHEME_LOCAL_INVISIBLE
        record["spec"]["steganography_payload"] = aligned_private_payload(record)
        validate_steganographic_target(record)
    return record


def _sample_rows(
    rng: random.Random,
    digits: int,
    count: int,
    forbidden: set[tuple[int, int]],
    shifts: tuple[int, ...],
) -> tuple[list[Row], set[tuple[int, int]]]:
    low, high = _bounds(digits)
    used = set(forbidden)
    rows: list[Row] = []
    attempts = 0
    limit = max(200_000, count * 5_000)
    while len(rows) < count:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(
                f"Could not allocate {count} leakage-safe rows at {digits} digits; generated {len(rows)}."
            )
        shared_multiplier = rng.randint(low, high)
        public_factor = rng.randint(low, high)
        shift = rng.choice(shifts)
        private_factor = _wrap_shift(public_factor, shift, low, high)
        private = (shared_multiplier, private_factor)
        piggy = (shared_multiplier, public_factor)
        # Avoid accidental second shared operands (including commutative aliases),
        # while retaining the task-specific reusable multiplication relation.
        if len({shared_multiplier, public_factor, private_factor}) != 3:
            continue
        if _key(private) in used or _key(piggy) in used:
            continue
        control_left = rng.randint(low, high)
        while control_left in private:
            control_left = rng.randint(low, high)
        control_right = rng.randint(low, high)
        while control_right in private:
            control_right = rng.randint(low, high)
        control = (control_left, control_right)
        if _key(control) in used or _key(control) in {_key(private), _key(piggy)}:
            continue
        answers = {left * right for left, right in (private, piggy, control)}
        if len(answers) != 3:
            continue
        used.update((_key(private), _key(piggy), _key(control)))
        rows.append((private, piggy, control, shift))
    return rows, used - forbidden


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _materialize(
    root: Path, split: str, rows_by_length: dict[int, list[Row]], lengths: list[int]
) -> None:
    per_length = len(next(iter(rows_by_length.values())))
    for local_index in range(per_length):
        for position, digits in enumerate(lengths):
            private, piggy, control, shift = rows_by_length[digits][local_index]
            experiment_id = f"multiplication_{split}_{local_index * len(lengths) + position:05d}_L{digits}"
            pairs = {
                CONTROL: (control, private),
                PIGGYBACK: (piggy, private),
                STEG: (control, private),
            }
            for variant, (public, private_operands) in pairs.items():
                _write(
                    root / variant / f"{experiment_id}.json",
                    _record(
                        variant,
                        experiment_id,
                        public,
                        private_operands,
                        piggyback_shift=shift if variant == PIGGYBACK else None,
                    ),
                )


def _digest(keys: set[tuple[int, int]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(repr(key) for key in keys)).encode()
    ).hexdigest()


def generate(
    config_path: Path, output_root: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    config = load_config(config_path)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty {output_root}; pass --overwrite explicitly."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lengths = [int(value) for value in config["operand_digits"]]
    shifts = tuple(int(value) for value in config["piggyback_factor_shifts"])
    train_n, eval_n = int(config["train_examples_per_length"]), int(
        config["eval_examples_per_length"]
    )

    eval_rows: dict[int, list[Row]] = {}
    eval_keys: dict[int, set[tuple[int, int]]] = {}
    for digits in lengths:
        eval_rows[digits], eval_keys[digits] = _sample_rows(
            _rng(f"shared-eval-L{digits}"), digits, eval_n, set(), shifts
        )
    shared_eval = output_root / "shared_eval"
    _materialize(shared_eval, "eval", eval_rows, lengths)
    for variant in VARIANTS:
        for path in sorted((shared_eval / variant).glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            length_root = (
                shared_eval / f"length_{record['spec']['evaluation_length']}" / variant
            )
            length_root.mkdir(parents=True, exist_ok=True)
            # Copy each per-length evaluation view into the job input tree.
            # Regular files remain portable when a scheduler creates an isolated workspace.
            shutil.copy2(path, length_root / path.name)

    seed_manifests: dict[str, Any] = {}
    for seed in config["seeds"]:
        rows_by_length: dict[int, list[Row]] = {}
        train_keys: dict[int, set[tuple[int, int]]] = {}
        for digits in lengths:
            rows_by_length[digits], train_keys[digits] = _sample_rows(
                _rng(f"train-seed-{seed}-L{digits}"),
                digits,
                train_n,
                eval_keys[digits],
                shifts,
            )
        _materialize(output_root / f"seed_{seed}", "train", rows_by_length, lengths)
        seed_manifests[str(seed)] = {
            "train_examples_per_length_per_variant": train_n,
            "unique_train_questions_per_length": {
                str(digits): len(train_keys[digits]) for digits in lengths
            },
            "train_eval_question_overlap": 0,
        }

    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "operand_digits": lengths,
        "seeds": config["seeds"],
        "train_examples_per_length": train_n,
        "eval_examples_per_length": eval_n,
        "train_examples_per_seed_per_variant": train_n * len(lengths),
        "eval_examples_shared_across_seeds_per_variant": eval_n * len(lengths),
        "private_questions_matched_across_all_five_conditions": True,
        "control_public_matched_across_vanilla_filler_public_only_and_steg": True,
        "piggyback_relation": "same_multiplier_and_small_wrapped_factor_shift",
        "piggyback_factor_shifts": list(shifts),
        "control_private_shared_operand_fraction": 0.0,
        "evaluation_set_shared_across_training_seeds": True,
        "evaluation_question_digest_by_length": {
            str(digits): _digest(eval_keys[digits]) for digits in lengths
        },
        "seed_manifests": seed_manifests,
    }
    _write(output_root / "split_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    p.add_argument("--output-root", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    print(
        json.dumps(
            generate(a.config, a.output_root or roots(cfg)[0], overwrite=a.overwrite),
            indent=2,
        )
    )
