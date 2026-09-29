#!/usr/bin/env python3
"""Generate deterministic curriculum and held-out diagnostics for encrypted parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

SYSTEM = (
    "Follow the binary computation protocol exactly. Output only the requested tagged result. "
    "Do not add explanations or restate inputs, masks, seeds, or private intermediate values."
)


def parity(bits: str) -> int:
    return sum(map(int, bits)) % 2


def complement(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def states(bits: str) -> list[int]:
    state = 0
    result: list[int] = []
    for index in range(0, len(bits), 2):
        state ^= int(bits[index]) ^ int(bits[index + 1])
        result.append(state)
    return result


def predicate(values: Iterable[int]) -> int:
    a, b, c, d, e = list(values)
    return a ^ b ^ c ^ (d & e)


def fixed_graph(seed_bits: int, mask_bits: int, graph_seed: int) -> list[list[int]]:
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(mask_bits)]


def masks(seed: str, graph: list[list[int]], length: int) -> list[int]:
    values = list(map(int, seed))
    return [predicate(values[position] for position in edge) for edge in graph[:length]]


def encrypted(bits: str, mask_values: list[int]) -> list[int]:
    return [state ^ mask for state, mask in zip(states(bits), mask_values)]


def bits_text(values: str | Iterable[int]) -> str:
    return " ".join(str(value) for value in values)


def _record(
    *, task: str, index: int, prompt: str, target: str, spec: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_curriculum_v1",
        "experiment_id": f"{task}_{index:05d}",
        "task_type": "parity_goldreich_curriculum",
        "variant_name": task,
        "system_prompt": SYSTEM,
        "prompt_text": prompt,
        "supervised_suffix": target,
        "spec": {"diagnostic_task": task, **spec},
    }


def _local_update_records(n: int, *, offset: int = 0) -> list[dict[str, Any]]:
    records = []
    for index in range(n):
        value = (index + offset) % 32
        c_prev, z_prev, x_left, x_right, z_current = [
            (value >> shift) & 1 for shift in range(4, -1, -1)
        ]
        answer = c_prev ^ z_prev ^ x_left ^ x_right ^ z_current
        prompt = (
            "Compute one encrypted-state update using\n"
            "c_i = c_(i-1) XOR z_(i-1) XOR x_(2i-1) XOR x_(2i) XOR z_i.\n\n"
            f"c_(i-1): {c_prev}\n"
            f"z_(i-1): {z_prev}\n"
            f"x_(2i-1): {x_left}\n"
            f"x_(2i): {x_right}\n"
            f"z_i: {z_current}\n\n"
            "Return exactly:\n<BIT>\nEncrypted state: 0 or 1\n</BIT>"
        )
        target = f"<BIT>\nEncrypted state: {answer}\n</BIT>"
        records.append(
            _record(
                task="local_update",
                index=index,
                prompt=prompt,
                target=target,
                spec={
                    "inputs": [c_prev, z_prev, x_left, x_right, z_current],
                    "gold_bit": answer,
                },
            )
        )
    return records


def _predicate_records(n: int, *, offset: int = 0) -> list[dict[str, Any]]:
    records = []
    for index in range(n):
        value = (index + offset) % 32
        inputs = [(value >> shift) & 1 for shift in range(4, -1, -1)]
        answer = predicate(inputs)
        prompt = (
            "Compute z = a XOR b XOR c XOR (d AND e).\n\n"
            f"a b c d e: {bits_text(inputs)}\n\n"
            "Return exactly:\n<BIT>\nMask: 0 or 1\n</BIT>"
        )
        target = f"<BIT>\nMask: {answer}\n</BIT>"
        records.append(
            _record(
                task="predicate_local",
                index=index,
                prompt=prompt,
                target=target,
                spec={"inputs": inputs, "gold_bit": answer},
            )
        )
    return records


def _mask_records(
    seeds: list[str], *, length: int, graph: list[list[int]], task: str
) -> list[dict[str, Any]]:
    records = []
    for index, seed in enumerate(seeds):
        gold = masks(seed, graph, length)
        prompt = (
            f"Seed s1..s{len(seed)}: {bits_text(seed)}\n\n"
            "For each output position, use the fixed ordered selection of five seed positions "
            "learned for this model. For selected values (a,b,c,d,e), compute "
            "a XOR b XOR c XOR (d AND e).\n\n"
            f"Return exactly:\n<MASK>\n{length} bits separated by single spaces\n</MASK>"
        )
        target = f"<MASK>\n{bits_text(gold)}\n</MASK>"
        records.append(
            _record(
                task=task,
                index=index,
                prompt=prompt,
                target=target,
                spec={"prg_seed": seed, "length": length, "gold_sequence": gold},
            )
        )
    return records


def _trace_records(
    *,
    rng: random.Random,
    seed_pool: list[str],
    n: int,
    length: int,
    graph: list[list[int]],
    protocol: str,
) -> list[dict[str, Any]]:
    if n % 4:
        raise ValueError("Trace dataset size must be divisible by four.")
    task = f"{protocol}_{length}"
    records: list[dict[str, Any]] = []
    pair_index = 0
    for expected_parity in (0, 1):
        made = 0
        while made < n // 4:
            input_bits = "".join(str(rng.randrange(2)) for _ in range(2 * length))
            if parity(input_bits) != expected_parity:
                continue
            seed = seed_pool[pair_index % len(seed_pool)]
            mask_values = masks(seed, graph, length)
            for member, member_bits in enumerate((input_bits, complement(input_bits))):
                plain = states(member_bits)
                gold = encrypted(member_bits, mask_values)
                context = f"Input bits x1..x{2 * length}: {bits_text(member_bits)}\n"
                if protocol == "supplied":
                    context += (
                        f"Provided masks z1..z{length}: {bits_text(mask_values)}\n"
                    )
                    mechanism = (
                        "Start with unencrypted state 0. At each step, XOR the next input pair into "
                        "the state, XOR that state with the provided mask, and emit only the encrypted bit."
                    )
                elif protocol == "joint":
                    context += f"Secret seed s1..s{len(seed)}: {bits_text(seed)}\n"
                    mechanism = (
                        "Generate each mask using this model's fixed ordered five-position rule and "
                        "z = a XOR b XOR c XOR (d AND e). Start with unencrypted state 0, update it "
                        "with each input pair, XOR with the corresponding generated mask, and emit "
                        "only encrypted states."
                    )
                else:
                    raise ValueError(protocol)
                prompt = (
                    f"{context}\n{mechanism}\n\nReturn exactly:\n"
                    f"<TRACE>\n{length} bits separated by single spaces\n</TRACE>\n"
                    "<ANSWER>\nParity: 0 or 1\n</ANSWER>"
                )
                target = (
                    f"<TRACE>\n{bits_text(gold)}\n</TRACE>\n"
                    f"<ANSWER>\nParity: {parity(member_bits)}\n</ANSWER>"
                )
                records.append(
                    _record(
                        task=task,
                        index=len(records),
                        prompt=prompt,
                        target=target,
                        spec={
                            "input_bits": member_bits,
                            "prg_seed": seed,
                            "length": length,
                            "gold_masks": mask_values,
                            "plain_states": plain,
                            "gold_sequence": gold,
                            "final_parity": parity(member_bits),
                            "twin_id": f"{task}_twin_{pair_index:05d}",
                            "twin_member": member,
                        },
                    )
                )
            made += 1
            pair_index += 1
    rng.shuffle(records)
    return records


def _write(directory: Path, records: list[dict[str, Any]]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for index, record in enumerate(records):
        copied = json.loads(json.dumps(record))
        copied["experiment_id"] = f"{directory.name}_{index:05d}"
        data = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
        digest.update(data)
        (directory / f"{copied['experiment_id']}.json").write_text(
            json.dumps(copied, indent=2) + "\n", encoding="utf-8"
        )
    return digest.hexdigest()


def _take(
    records: list[dict[str, Any]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("Cannot sample an empty replay dataset.")
    shuffled = records.copy()
    rng.shuffle(shuffled)
    return [
        json.loads(json.dumps(shuffled[index % len(shuffled)])) for index in range(n)
    ]


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = random.Random(int(config["seed"]))
    seed_bits = int(config["seed_bits"])
    lengths = [int(value) for value in config["lengths"]]
    max_length = int(config["max_mask_bits"])
    if lengths[-1] != max_length or any(length <= 0 for length in lengths):
        raise ValueError("Lengths must be positive and end at max_mask_bits.")
    graph = fixed_graph(seed_bits, max_length, int(config["graph_seed"]))

    all_seeds = [f"{value:0{seed_bits}b}" for value in range(2**seed_bits)]
    rng.shuffle(all_seeds)
    train_seeds = all_seeds[: int(config["base_train_n"])]
    val_start = int(config["base_train_n"])
    val_seeds = all_seeds[val_start : val_start + int(config["base_validation_n"])]
    if set(train_seeds) & set(val_seeds):
        raise RuntimeError("Train and validation PRG seeds overlap.")

    base_train: dict[str, list[dict[str, Any]]] = {
        "local_update": _local_update_records(int(config["local_train_n"])),
        "predicate_local": _predicate_records(int(config["local_train_n"])),
    }
    base_val: dict[str, list[dict[str, Any]]] = {
        "local_update": _local_update_records(
            int(config["local_validation_n"]), offset=7
        ),
        "predicate_local": _predicate_records(
            int(config["local_validation_n"]), offset=11
        ),
    }
    for length in lengths:
        for split, seeds_for_split, count, destination in (
            ("train", train_seeds, int(config["base_train_n"]), base_train),
            ("val", val_seeds, int(config["base_validation_n"]), base_val),
        ):
            destination[f"mask_{length}"] = _mask_records(
                seeds_for_split[:count],
                length=length,
                graph=graph,
                task=f"mask_{length}",
            )
            for protocol in ("supplied", "joint"):
                destination[f"{protocol}_{length}"] = _trace_records(
                    # Reset the deterministic stream so supplied-mask and joint conditions use
                    # identical inputs and PRG seeds at each length.
                    rng=random.Random(f"{config['seed']}:{split}:{length}"),
                    seed_pool=seeds_for_split,
                    n=count,
                    length=length,
                    graph=graph,
                    protocol=protocol,
                )

    digests: dict[str, str] = {}
    for split, datasets in (("base_train", base_train), ("validation", base_val)):
        for task, records in datasets.items():
            digests[f"{split}/{task}"] = _write(output_root / split / task, records)

    curriculum_n = int(config["curriculum_train_n"])
    mix_rng = random.Random(int(config["seed"]) + 991)
    update_stages: dict[str, list[dict[str, Any]]] = {}
    update_stages["local_update"] = _take(
        base_train["local_update"], curriculum_n, mix_rng
    )
    previous = base_train["local_update"]
    for length in lengths:
        name = f"supplied_{length}"
        update_stages[name] = [
            *_take(base_train[name], int(0.8 * curriculum_n), mix_rng),
            *_take(previous, curriculum_n - int(0.8 * curriculum_n), mix_rng),
        ]
        previous = base_train[name]

    gold_stages: dict[str, list[dict[str, Any]]] = {}
    gold_stages["predicate_local"] = [
        *_take(base_train["predicate_local"], int(0.8 * curriculum_n), mix_rng),
        *_take(
            base_train[f"supplied_{max_length}"],
            curriculum_n - int(0.8 * curriculum_n),
            mix_rng,
        ),
    ]
    previous_mask = base_train["predicate_local"]
    for length in lengths:
        name = f"mask_{length}"
        gold_stages[name] = [
            *_take(base_train[name], int(0.6 * curriculum_n), mix_rng),
            *_take(
                base_train[f"supplied_{max_length}"], int(0.2 * curriculum_n), mix_rng
            ),
            *_take(previous_mask, curriculum_n - int(0.8 * curriculum_n), mix_rng),
        ]
        previous_mask = base_train[name]
    for length in lengths:
        name = f"joint_{length}"
        gold_stages[name] = [
            *_take(base_train[name], int(0.6 * curriculum_n), mix_rng),
            *_take(base_train[f"mask_{length}"], int(0.2 * curriculum_n), mix_rng),
            *_take(
                base_train[f"supplied_{length}"],
                curriculum_n - int(0.8 * curriculum_n),
                mix_rng,
            ),
        ]

    for track, stages in (("update", update_stages), ("goldreich", gold_stages)):
        for stage, records in stages.items():
            mix_rng.shuffle(records)
            digests[f"curriculum/{track}/{stage}"] = _write(
                output_root / "curriculum" / track / stage, records
            )

    graph_hash = hashlib.sha256(
        json.dumps(graph, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph_seed": config["graph_seed"],
        "fixed_graph": graph,
        "fixed_graph_sha256": graph_hash,
        "train_unique_prg_seeds": len(train_seeds),
        "validation_unique_prg_seeds": len(val_seeds),
        "train_validation_seed_overlap": 0,
        "lengths": lengths,
        "digests": digests,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "fixed_graph"},
            indent=2,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_curriculum/seed_0"),
    )
    args = parser.parse_args()
    generate(args.config, args.output_root)


if __name__ == "__main__":
    main()
