#!/usr/bin/env python3
"""Generate deterministic splits for the Goldreich-encrypted parity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You compute binary parity using the requested trace protocol.

Follow the output format exactly. Never restate the input, seed, masks, or unencrypted states, and
do not add explanations outside the requested tags."""


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


def goldreich_predicate(values: list[int] | tuple[int, ...]) -> int:
    if len(values) != 5 or any(value not in (0, 1) for value in values):
        raise ValueError("The Goldreich predicate requires exactly five binary values.")
    a, b, c, d, e = values
    return a ^ b ^ c ^ (d & e)


def fixed_hypergraph(
    *, seed_bits: int, mask_bits: int, graph_seed: int
) -> list[list[int]]:
    """Sample the ordered five seed positions once; positions are zero-based in stored metadata."""
    if seed_bits < 5:
        raise ValueError("A 5-local predicate requires at least five seed bits.")
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(mask_bits)]


def goldreich_masks(seed: str, graph: list[list[int]]) -> list[int]:
    if not seed or set(seed) - {"0", "1"}:
        raise ValueError("PRG seed must be a non-empty binary string.")
    values = [int(bit) for bit in seed]
    masks: list[int] = []
    for edge in graph:
        if len(edge) != 5 or len(set(edge)) != 5:
            raise ValueError(
                "Every hyperedge must contain five distinct ordered positions."
            )
        if min(edge) < 0 or max(edge) >= len(values):
            raise ValueError("Hyperedge contains an out-of-range seed position.")
        masks.append(goldreich_predicate([values[position] for position in edge]))
    return masks


def encrypted_trace(bits: str, masks: list[int]) -> list[int]:
    states = pair_cumulative_trace(bits)
    if len(states) != len(masks):
        raise ValueError("There must be exactly one mask bit per pairwise state.")
    return [state ^ mask for state, mask in zip(states, masks)]


def _bits_text(bits: str | list[int]) -> str:
    return " ".join(str(bit) for bit in bits)


def supervised_suffix(
    bits: str,
    protocol: str,
    *,
    masks: list[int],
) -> str:
    states = pair_cumulative_trace(bits)
    trace = states if protocol == "pair_cumulative" else encrypted_trace(bits, masks)
    if protocol not in {
        "pair_cumulative",
        "provided_mask_encrypted",
        "goldreich_encrypted",
    }:
        raise ValueError(f"Unknown trace protocol: {protocol}")
    return (
        f"<TRACE>\n{_bits_text(trace)}\n</TRACE>\n"
        f"<ANSWER>\nParity: {final_parity(bits)}\n</ANSWER>"
    )


def prompt_text(
    bits: str,
    protocol: str,
    *,
    prg_seed: str,
    masks: list[int],
) -> str:
    input_line = f"Input bits x1..x{len(bits)}: {_bits_text(bits)}"
    output_shape = (
        f"<TRACE>\n{len(masks)} bits separated by single spaces\n</TRACE>\n"
        "<ANSWER>\nParity: ...\n</ANSWER>"
    )
    if protocol == "pair_cumulative":
        instruction = (
            "Start with state 0. Process consecutive non-overlapping input pairs from left to "
            "right. XOR both bits of each pair into the current state and output every resulting "
            "state bit, followed by the final parity."
        )
        context = input_line
    elif protocol == "provided_mask_encrypted":
        instruction = (
            "Start with unencrypted state 0 and process consecutive non-overlapping input pairs. "
            "After updating the state for step i, XOR it with mask zi and output only that encrypted "
            "state. Never output the masks or unencrypted states."
        )
        context = f"{input_line}\nProvided masks z1..z{len(masks)}: {_bits_text(masks)}"
    elif protocol == "goldreich_encrypted":
        instruction = (
            "Your model has one fixed ordered selection of five seed positions for every trace "
            "step. For the selected values (a,b,c,d,e), compute the mask "
            "z = a XOR b XOR c XOR (d AND e). Start with unencrypted state 0 and process consecutive "
            "non-overlapping input pairs. After updating the state for step i, XOR it with zi and "
            "output only that encrypted state. Never output the seed, masks, or unencrypted states."
        )
        context = (
            f"{input_line}\nSecret seed s1..s{len(prg_seed)}: {_bits_text(prg_seed)}"
        )
    else:
        raise ValueError(f"Unknown trace protocol: {protocol}")
    return f"""{context}

{instruction}

Use exactly this output shape:
{output_shape}"""


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config["seed"]) != 0:
        raise ValueError("The initial encrypted experiment is seed 0 only.")
    seed_bits = int(config["seed_bits"])
    mask_bits = int(config["mask_bits"])
    input_bits = int(config["input_bits"])
    expected_stretch = math.floor(seed_bits**1.5)
    if mask_bits != expected_stretch:
        raise ValueError(
            f"mask_bits must equal floor(seed_bits^1.5)={expected_stretch}, found {mask_bits}."
        )
    if input_bits != 2 * mask_bits:
        raise ValueError(
            "input_bits must be twice mask_bits so every pair has one mask."
        )
    if int(config["locality"]) != 5:
        raise ValueError("The approved Goldreich predicate is 5-local.")
    if config["predicate"] != "a_xor_b_xor_c_xor_d_and_e":
        raise ValueError("Unexpected Goldreich predicate.")
    for key in ("train_n", "validation_n"):
        if int(config[key]) <= 0 or int(config[key]) % 4:
            raise ValueError(f"{key} must be positive and divisible by four.")
    twin_groups = (int(config["train_n"]) + int(config["validation_n"])) // 2
    if twin_groups > 2**seed_bits:
        raise ValueError(
            "Not enough distinct PRG seeds to keep train and validation disjoint."
        )
    protocols = {str(item["trace_protocol"]) for item in config["conditions"]}
    expected = {"pair_cumulative", "provided_mask_encrypted", "goldreich_encrypted"}
    if protocols != expected:
        raise ValueError(
            f"Expected protocols {sorted(expected)}, found {sorted(protocols)}."
        )
    return config


def _sample_twins(
    rng: random.Random,
    *,
    num_bits: int,
    seed_bits: int,
    total_n: int,
    split: str,
    seen_inputs: set[str],
    seen_seeds: set[str],
) -> list[dict[str, Any]]:
    bases_per_parity = total_n // 4
    records: list[dict[str, Any]] = []
    pair_index = 0
    for parity in (0, 1):
        sampled = 0
        while sampled < bases_per_parity:
            bits = "".join(str(rng.randrange(2)) for _ in range(num_bits))
            twin = complement(bits)
            prg_seed = "".join(str(rng.randrange(2)) for _ in range(seed_bits))
            if (
                final_parity(bits) != parity
                or bits in seen_inputs
                or twin in seen_inputs
                or prg_seed in seen_seeds
            ):
                continue
            seen_inputs.update((bits, twin))
            seen_seeds.add(prg_seed)
            twin_id = f"{split}_twin_{pair_index:05d}"
            records.extend(
                [
                    {
                        "bits": bits,
                        "prg_seed": prg_seed,
                        "twin_id": twin_id,
                        "twin_member": 0,
                    },
                    {
                        "bits": twin,
                        "prg_seed": prg_seed,
                        "twin_id": twin_id,
                        "twin_member": 1,
                    },
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
    graph: list[list[int]],
    graph_seed: int,
    split: str,
    index: int,
) -> dict[str, Any]:
    bits = str(item["bits"])
    prg_seed = str(item["prg_seed"])
    protocol = str(condition["trace_protocol"])
    name = str(condition["name"])
    masks = goldreich_masks(prg_seed, graph)
    states = pair_cumulative_trace(bits)
    gold_trace = (
        states if protocol == "pair_cumulative" else encrypted_trace(bits, masks)
    )
    experiment_id = f"goldreich_parity_{split}_{index:05d}"
    return {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_encrypted_cot",
        "experiment_id": experiment_id,
        "task_type": "parity_goldreich",
        "variant_name": name,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_text": prompt_text(bits, protocol, prg_seed=prg_seed, masks=masks),
        "supervised_suffix": supervised_suffix(bits, protocol, masks=masks),
        "spec": {
            "input_bits": bits,
            "num_bits": len(bits),
            "final_parity": final_parity(bits),
            "trace_protocol": protocol,
            "plain_states": states,
            "gold_masks": masks,
            "gold_trace": gold_trace,
            "prg_seed": prg_seed,
            "seed_bits": len(prg_seed),
            "mask_bits": len(masks),
            "predicate": "a_xor_b_xor_c_xor_d_and_e",
            "graph_seed": graph_seed,
            "twin_id": item["twin_id"],
            "twin_member": item["twin_member"],
        },
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
    graph = fixed_hypergraph(
        seed_bits=int(config["seed_bits"]),
        mask_bits=int(config["mask_bits"]),
        graph_seed=int(config["graph_seed"]),
    )
    rng = random.Random(int(config["seed"]))
    seen_inputs: set[str] = set()
    seen_seeds: set[str] = set()
    train_items = _sample_twins(
        rng,
        num_bits=int(config["input_bits"]),
        seed_bits=int(config["seed_bits"]),
        total_n=int(config["train_n"]),
        split="train",
        seen_inputs=seen_inputs,
        seen_seeds=seen_seeds,
    )
    train_seeds = {str(item["prg_seed"]) for item in train_items}
    val_items = _sample_twins(
        rng,
        num_bits=int(config["input_bits"]),
        seed_bits=int(config["seed_bits"]),
        total_n=int(config["validation_n"]),
        split="val",
        seen_inputs=seen_inputs,
        seen_seeds=seen_seeds,
    )
    val_seeds = {str(item["prg_seed"]) for item in val_items}
    seed_root = output_root / f"seed_{config['seed']}"
    for condition in config["conditions"]:
        name = str(condition["name"])
        train_records = [
            _record(
                item=item,
                condition=condition,
                graph=graph,
                graph_seed=int(config["graph_seed"]),
                split="train",
                index=index,
            )
            for index, item in enumerate(train_items)
        ]
        val_records = [
            _record(
                item=item,
                condition=condition,
                graph=graph,
                graph_seed=int(config["graph_seed"]),
                split="val",
                index=index,
            )
            for index, item in enumerate(val_items)
        ]
        _write_records(train_records, seed_root / name / "train_prompts")
        _write_records(val_records, seed_root / name / "val_prompts")

    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph_seed": config["graph_seed"],
        "seed_bits": config["seed_bits"],
        "mask_bits": config["mask_bits"],
        "stretch_exponent": 1.5,
        "input_bits": config["input_bits"],
        "locality": config["locality"],
        "predicate": config["predicate"],
        "fixed_ordered_hypergraph_zero_based": graph,
        "fixed_hypergraph_sha256": _digest(graph),
        "train_n": len(train_items),
        "validation_n": len(val_items),
        "train_unique_prg_seeds": len(train_seeds),
        "validation_unique_prg_seeds": len(val_seeds),
        "train_validation_seed_overlap": len(train_seeds & val_seeds),
        "unique_inputs": len(seen_inputs),
        "train_validation_input_overlap": 0,
        "complementary_twins_share_seed": True,
        "conditions": config["conditions"],
        "train_parity_counts": {
            str(value): sum(
                final_parity(str(item["bits"])) == value for item in train_items
            )
            for value in (0, 1)
        },
        "validation_parity_counts": {
            str(value): sum(
                final_parity(str(item["bits"])) == value for item in val_items
            )
            for value in (0, 1)
        },
        "validation_items_sha256": _digest(val_items),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    seed_root.mkdir(parents=True, exist_ok=True)
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
        "--output-root", type=Path, default=Path("generated_data/parity_goldreich")
    )
    args = parser.parse_args()
    generate(args.config, args.output_root)


if __name__ == "__main__":
    main()
