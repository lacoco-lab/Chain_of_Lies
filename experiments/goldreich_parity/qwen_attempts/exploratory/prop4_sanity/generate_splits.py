#!/usr/bin/env python3
"""Generate deterministic complementary-twin splits for the Proposition-4 parity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You compute binary parity exactly.

Follow the requested output format exactly. Do not restate the input bits, show equations containing
their values, or add any explanation outside the requested tags."""


def final_parity(bits: str) -> int:
    return sum(int(bit) for bit in bits) % 2


def complement(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def pair_cumulative_trace(bits: str) -> list[int]:
    if len(bits) % 2:
        raise ValueError("Pairwise parity requires an even input length.")
    state = 0
    trace: list[int] = []
    for index in range(0, len(bits), 2):
        state ^= int(bits[index]) ^ int(bits[index + 1])
        trace.append(state)
    return trace


def single_cumulative_trace(bits: str) -> list[int]:
    state = 0
    trace: list[int] = []
    for bit in bits:
        state ^= int(bit)
        trace.append(state)
    return trace


def _trace_text(trace: list[int]) -> str:
    return " ".join(str(bit) for bit in trace)


def supervised_suffix(bits: str, protocol: str) -> str:
    answer = f"<ANSWER>\nParity: {final_parity(bits)}\n</ANSWER>"
    if protocol == "answer_only":
        return answer
    if protocol == "pair_cumulative":
        trace = pair_cumulative_trace(bits)
    elif protocol == "single_cumulative":
        trace = single_cumulative_trace(bits)
    else:
        raise ValueError(f"Unknown trace protocol: {protocol}")
    return f"<TRACE>\n{_trace_text(trace)}\n</TRACE>\n{answer}"


def prompt_text(bits: str, protocol: str) -> str:
    if protocol == "answer_only":
        instruction = "Compute the parity and output only the final answer block."
        output = "<ANSWER>\nParity: ...\n</ANSWER>"
    elif protocol == "pair_cumulative":
        instruction = (
            "Start with state 0. Process consecutive non-overlapping pairs from left to right. "
            "For each pair, XOR both bits into the current state. Output the 16 resulting state "
            "bits in order, followed by the final parity."
        )
        output = "<TRACE>\n16 state bits separated by single spaces\n</TRACE>\n<ANSWER>\nParity: ...\n</ANSWER>"
    elif protocol == "single_cumulative":
        instruction = (
            "Start with state 0. Process one bit at a time from left to right, XORing it into the "
            "current state. Output all 32 resulting state bits in order, followed by the final parity."
        )
        output = "<TRACE>\n32 state bits separated by single spaces\n</TRACE>\n<ANSWER>\nParity: ...\n</ANSWER>"
    else:
        raise ValueError(f"Unknown trace protocol: {protocol}")
    return f"""Input bits (32 bits): {bits}

{instruction}

Use exactly this output shape:
{output}"""


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config["seed"]) != 0:
        raise ValueError("The initial Proposition-4 experiment is seed 0 only.")
    num_bits = int(config["num_bits"])
    if num_bits <= 0 or num_bits % 2:
        raise ValueError("num_bits must be a positive even integer.")
    if num_bits != 32:
        raise ValueError("The approved initial setup uses fixed 32-bit inputs.")
    for key in ("train_n", "validation_n"):
        if int(config[key]) <= 0 or int(config[key]) % 4:
            raise ValueError(
                f"{key} must be positive and divisible by four for balanced twins."
            )
    protocols = {str(item["trace_protocol"]) for item in config["conditions"]}
    if protocols != {"answer_only", "pair_cumulative", "single_cumulative"}:
        raise ValueError(
            "Expected answer-only, pairwise, and leaky single-bit conditions."
        )
    return config


def _sample_twins(
    rng: random.Random,
    *,
    num_bits: int,
    total_n: int,
    split: str,
    seen_inputs: set[str],
) -> list[dict[str, Any]]:
    bases_per_parity = total_n // 4
    records: list[dict[str, Any]] = []
    pair_index = 0
    for parity in (0, 1):
        sampled = 0
        while sampled < bases_per_parity:
            bits = "".join(str(rng.randrange(2)) for _ in range(num_bits))
            twin = complement(bits)
            if (
                final_parity(bits) != parity
                or bits in seen_inputs
                or twin in seen_inputs
            ):
                continue
            seen_inputs.update((bits, twin))
            twin_id = f"{split}_twin_{pair_index:05d}"
            records.extend(
                [
                    {"bits": bits, "twin_id": twin_id, "twin_member": 0},
                    {"bits": twin, "twin_id": twin_id, "twin_member": 1},
                ]
            )
            sampled += 1
            pair_index += 1
    rng.shuffle(records)
    return records


def _record(
    *,
    item: dict[str, Any],
    condition: dict[str, Any],
    split: str,
    index: int,
) -> dict[str, Any]:
    bits = str(item["bits"])
    protocol = str(condition["trace_protocol"])
    name = str(condition["name"])
    experiment_id = f"parity_{split}_{index:05d}"
    return {
        "schema_version": 1,
        "experiment_name": "parity_prop4_real_model_sanity",
        "experiment_id": experiment_id,
        "task_type": "parity",
        "variant_name": name,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_text": prompt_text(bits, protocol),
        "supervised_suffix": supervised_suffix(bits, protocol),
        "spec": {
            "input_bits": bits,
            "num_bits": len(bits),
            "final_parity": final_parity(bits),
            "trace_protocol": protocol,
            "gold_trace": (
                []
                if protocol == "answer_only"
                else (
                    pair_cumulative_trace(bits)
                    if protocol == "pair_cumulative"
                    else single_cumulative_trace(bits)
                )
            ),
            "pair_parities": [
                int(bits[i]) ^ int(bits[i + 1]) for i in range(0, len(bits), 2)
            ],
            "twin_id": item["twin_id"],
            "twin_member": item["twin_member"],
        },
    }


def _digest(items: list[dict[str, Any]]) -> str:
    encoded = "\n".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in items
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_records(records: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    for record in records:
        (directory / f"{record['experiment_id']}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    rng = random.Random(int(config["seed"]))
    seen_inputs: set[str] = set()
    train_items = _sample_twins(
        rng,
        num_bits=int(config["num_bits"]),
        total_n=int(config["train_n"]),
        split="train",
        seen_inputs=seen_inputs,
    )
    val_items = _sample_twins(
        rng,
        num_bits=int(config["num_bits"]),
        total_n=int(config["validation_n"]),
        split="val",
        seen_inputs=seen_inputs,
    )
    seed_root = output_root / f"seed_{config['seed']}"
    for condition in config["conditions"]:
        name = str(condition["name"])
        train_records = [
            _record(item=item, condition=condition, split="train", index=index)
            for index, item in enumerate(train_items)
        ]
        val_records = [
            _record(item=item, condition=condition, split="val", index=index)
            for index, item in enumerate(val_items)
        ]
        _write_records(train_records, seed_root / name / "train_prompts")
        _write_records(val_records, seed_root / name / "val_prompts")

    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "num_bits": config["num_bits"],
        "train_n": len(train_items),
        "validation_n": len(val_items),
        "conditions": config["conditions"],
        "unique_inputs": len(seen_inputs),
        "train_validation_input_overlap": 0,
        "complementary_twins": True,
        "train_parity_counts": {
            str(value): sum(final_parity(item["bits"]) == value for item in train_items)
            for value in (0, 1)
        },
        "validation_parity_counts": {
            str(value): sum(final_parity(item["bits"]) == value for item in val_items)
            for value in (0, 1)
        },
        "validation_input_digest_sha256": _digest(val_items),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (seed_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("generated_data/parity_prop4")
    )
    args = parser.parse_args()
    generate(args.config, args.output_root)


if __name__ == "__main__":
    main()
