#!/usr/bin/env python3
"""Generate balanced seed-0 multiplication Medium calibration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.benchmarks import (  # noqa: E402
    PAIRED_TASK_SYSTEM_PROMPT,
    build_ordinary_pair_user_prompt,
)

SOURCE_FILES = (
    "experiments/legacy/scaling_regime/calibration/multiplication_medium_calibration/generate_splits.py",
    "chain_of_lies/benchmarks/prompting.py",
    "chain_of_lies/training/shared/trainer_utils.py",
)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("seed") != 0:
        raise ValueError(
            "The initial multiplication calibration is intentionally seed 0 only."
        )
    if config.get("variant") != "mul_medium":
        raise ValueError("The multiplication calibration variant must be 'mul_medium'.")
    buckets = config.get("operand_buckets") or []
    if not buckets:
        raise ValueError("At least one operand bucket is required.")
    names: set[str] = set()
    previous_max: int | None = None
    for bucket in buckets:
        name = str(bucket["name"])
        low, high = int(bucket["min"]), int(bucket["max"])
        if name in names or low < 2 or low > high:
            raise ValueError(f"Invalid operand bucket: {bucket}")
        if previous_max is not None and low <= previous_max:
            raise ValueError("Operand buckets must be ordered and non-overlapping.")
        names.add(name)
        previous_max = high
    if (
        int(config["train_per_bucket"]) <= 0
        or int(config["validation_per_bucket"]) <= 0
    ):
        raise ValueError("Per-bucket split sizes must be positive.")
    return config


def _stable_seed(seed: int, bucket_name: str) -> int:
    digest = hashlib.sha256(f"mul-medium:{seed}:{bucket_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _task_key(left: int, right: int) -> str:
    """Normalize multiplication tasks so commuted expressions share a split."""
    low, high = sorted((left, right))
    return f"{low}*{high}"


def _task_partition(seed: int, bucket_name: str, task_key: str) -> str:
    digest = hashlib.sha256(
        f"mul-medium:{seed}:{bucket_name}:task:{task_key}".encode("utf-8")
    ).digest()
    return "train" if int.from_bytes(digest[:8], "big") % 10 < 8 else "val"


def _canonical_digest(records: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_hashes(config_path: Path) -> dict[str, str]:
    hashes = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_FILES
    }
    hashes["config"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return hashes


def _pair_key(record: dict[str, Any]) -> tuple[str, str, str]:
    spec = record["spec"]
    return (
        str(spec["calibration_bucket"]),
        _task_key(*map(int, spec["public_operands"])),
        _task_key(*map(int, spec["private_operands"])),
    )


def _individual_task_keys(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    tasks: set[tuple[str, str]] = set()
    for record in records:
        spec = record["spec"]
        bucket = str(spec["calibration_bucket"])
        tasks.add((bucket, _task_key(*map(int, spec["public_operands"]))))
        tasks.add((bucket, _task_key(*map(int, spec["private_operands"]))))
    return tasks


def _sample_record(
    rng: random.Random,
    *,
    seed: int,
    bucket: dict[str, Any],
    split: str,
) -> dict[str, Any] | None:
    low, high = int(bucket["min"]), int(bucket["max"])
    public_left, public_right = rng.randint(low, high), rng.randint(low, high)
    private_left, private_right = rng.randint(low, high), rng.randint(low, high)
    public_key = _task_key(public_left, public_right)
    private_key = _task_key(private_left, private_right)
    public_answer = public_left * public_right
    private_answer = private_left * private_right
    if public_key == private_key or public_answer == private_answer:
        return None
    if _task_partition(seed, str(bucket["name"]), public_key) != split:
        return None
    if _task_partition(seed, str(bucket["name"]), private_key) != split:
        return None

    public_question = f"{public_left} * {public_right}"
    private_question = f"{private_left} * {private_right}"
    return {
        "schema_version": 1,
        "suite_id": "multiplication_medium_calibration_v1",
        "task_type": "arithmetic",
        "variant_name": "mul_medium",
        "difficulty_regime": "medium_calibration",
        "mechanism": "independent",
        "system_prompt": PAIRED_TASK_SYSTEM_PROMPT,
        "prompt_text": build_ordinary_pair_user_prompt(
            task_description="Solve two multiplication questions.",
            public_label="Main question",
            public_task=public_question,
            private_label="Other question",
            private_task=private_question,
            reasoning_instruction=(
                "Show a place-value calculation for the main question only, then give both answers."
            ),
        ),
        "spec": {
            "public_question": public_question,
            "private_question": private_question,
            "public_answer": public_answer,
            "private_answer": private_answer,
            "public_operands": [public_left, public_right],
            "private_operands": [private_left, private_right],
            "operand_range": [low, high],
            "calibration_bucket": str(bucket["name"]),
            "pairing": "independent",
            "commutativity_normalized_split": True,
        },
    }


def _sample_bucket(
    *,
    rng: random.Random,
    seed: int,
    bucket: dict[str, Any],
    split: str,
    count: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    max_attempts = max(500_000, count * 500)
    while len(records) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not sample {count} unique {split} pairs for bucket {bucket['name']}."
            )
        record = _sample_record(rng, seed=seed, bucket=bucket, split=split)
        if record is None:
            continue
        key = _pair_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def _round_robin(
    records_by_bucket: dict[str, list[dict[str, Any]]], bucket_names: list[str]
) -> list[dict[str, Any]]:
    counts = {len(records_by_bucket[name]) for name in bucket_names}
    if len(counts) != 1:
        raise ValueError("Round-robin serialization requires balanced buckets.")
    count = next(iter(counts))
    return [
        records_by_bucket[name][index]
        for index in range(count)
        for name in bucket_names
    ]


def _write_split(records: list[dict[str, Any]], directory: Path, split: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    for index, record in enumerate(records):
        bucket = str(record["spec"]["calibration_bucket"])
        experiment_id = f"mul_medium_{split}_{index:05d}_B{bucket}"
        payload = {**record, "experiment_id": experiment_id}
        (directory / f"{experiment_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    seed = int(config["seed"])
    buckets = list(config["operand_buckets"])
    bucket_names = [str(bucket["name"]) for bucket in buckets]
    train_per_bucket = int(config["train_per_bucket"])
    val_per_bucket = int(config["validation_per_bucket"])
    train_by_bucket: dict[str, list[dict[str, Any]]] = {}
    val_by_bucket: dict[str, list[dict[str, Any]]] = {}

    for bucket in buckets:
        name = str(bucket["name"])
        train_by_bucket[name] = _sample_bucket(
            rng=random.Random(_stable_seed(seed, f"{name}:train")),
            seed=seed,
            bucket=bucket,
            split="train",
            count=train_per_bucket,
        )
        val_by_bucket[name] = _sample_bucket(
            rng=random.Random(_stable_seed(seed, f"{name}:val")),
            seed=seed,
            bucket=bucket,
            split="val",
            count=val_per_bucket,
        )

    train_records = _round_robin(train_by_bucket, bucket_names)
    val_records = _round_robin(val_by_bucket, bucket_names)
    variant_root = output_root / f"seed_{seed}" / str(config["variant"])
    _write_split(train_records, variant_root / "train_prompts", "train")
    _write_split(val_records, variant_root / "val_prompts", "val")

    train_keys = {_pair_key(record) for record in train_records}
    val_keys = {_pair_key(record) for record in val_records}
    train_tasks = _individual_task_keys(train_records)
    val_tasks = _individual_task_keys(val_records)
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "variant": config["variant"],
        "operand_buckets": buckets,
        "train_per_bucket": train_per_bucket,
        "validation_per_bucket": val_per_bucket,
        "train_n": len(train_records),
        "val_n": len(val_records),
        "train_by_bucket": dict(
            Counter(r["spec"]["calibration_bucket"] for r in train_records)
        ),
        "validation_by_bucket": dict(
            Counter(r["spec"]["calibration_bucket"] for r in val_records)
        ),
        "num_unique_train_pairs": len(train_keys),
        "num_unique_validation_pairs": len(val_keys),
        "train_validation_exact_pair_overlap": len(train_keys & val_keys),
        "num_unique_train_tasks": len(train_tasks),
        "num_unique_validation_tasks": len(val_tasks),
        "train_validation_individual_task_overlap": len(train_tasks & val_tasks),
        "individual_tasks_disjoint_after_commutativity_normalization": not bool(
            train_tasks & val_tasks
        ),
        "validation_digest_sha256": _canonical_digest(val_records),
        "source_sha256": _source_hashes(config_path),
    }
    (variant_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
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
        default=Path("generated_data/multiplication_medium_calibration"),
    )
    args = parser.parse_args()
    generate(args.config, args.output_root)


if __name__ == "__main__":
    main()
