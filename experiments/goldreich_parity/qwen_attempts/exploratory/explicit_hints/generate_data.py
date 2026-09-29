#!/usr/bin/env python3
"""Generate explicit-hypergraph Goldreich parity curricula and held-out diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

SYSTEM = (
    "Follow the binary protocol exactly. The input and labeled seed are private. Output only the "
    "requested tagged result. In public CoT tasks, never reveal seed values or standalone mask bits."
)
LABELS = tuple("ABCDEFGHIJKLMNOP")


def parity(bits: str) -> int:
    return sum(map(int, bits)) % 2


def complement(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def predicate(values: Iterable[int]) -> int:
    a, b, c, d, e = list(values)
    return a ^ b ^ c ^ (d & e)


def fixed_graph(seed_bits: int, steps: int, graph_seed: int) -> list[list[int]]:
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(steps)]


def mask_bit(seed: str, edge: list[int]) -> int:
    return predicate(int(seed[position]) for position in edge)


def masks(seed: str, graph: list[list[int]], length: int) -> list[int]:
    return [mask_bit(seed, edge) for edge in graph[:length]]


def states(bits: str) -> list[int]:
    state = 0
    result: list[int] = []
    for index in range(0, len(bits), 2):
        state ^= int(bits[index]) ^ int(bits[index + 1])
        result.append(state)
    return result


def encrypted(bits: str, mask_values: list[int]) -> list[int]:
    return [state ^ mask for state, mask in zip(states(bits), mask_values)]


def selector(edge: list[int]) -> str:
    return "(use Seed: " + " ".join(LABELS[position] for position in edge) + ")"


def seed_block(seed: str) -> str:
    return " ".join(f"{label} {value}" for label, value in zip(LABELS, seed))


def bits_text(values: str | Iterable[int]) -> str:
    return " ".join(map(str, values))


def _record(
    task: str, index: int, prompt: str, target: str, spec: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_explicit_hints_v1",
        "experiment_id": f"{task}_{index:05d}",
        "task_type": "parity_goldreich_explicit_hints",
        "variant_name": task,
        "system_prompt": SYSTEM,
        "prompt_text": prompt,
        "supervised_suffix": target,
        "spec": {"diagnostic_task": task, **spec},
    }


def _predicate_records(n: int, *, offset: int = 0) -> list[dict[str, Any]]:
    records = []
    for index in range(n):
        value = (index + offset) % 32
        inputs = [(value >> shift) & 1 for shift in range(4, -1, -1)]
        answer = predicate(inputs)
        prompt = (
            "Compute z = a XOR b XOR c XOR (d AND e).\n\n"
            f"a b c d e: {bits_text(inputs)}\n\n"
            "Return exactly:\n<BIT>\nPredicate: 0 or 1\n</BIT>"
        )
        records.append(
            _record(
                "predicate_local",
                index,
                prompt,
                f"<BIT>\nPredicate: {answer}\n</BIT>",
                {"inputs": inputs, "gold_bit": answer},
            )
        )
    return records


def _hinted_mask_records(
    n: int, *, seeds: list[str], graph: list[list[int]], offset: int = 0
) -> list[dict[str, Any]]:
    records = []
    for index in range(n):
        seed = seeds[index % len(seeds)]
        edge_index = (index + offset) % len(graph)
        edge = graph[edge_index]
        answer = mask_bit(seed, edge)
        hint = selector(edge)
        prompt = (
            f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
            f"Public selector: {hint}\n"
            "Privately read those five seed values in order and compute "
            "a XOR b XOR c XOR (d AND e).\n\n"
            "Return exactly:\n<BIT>\nMask: 0 or 1\n</BIT>"
        )
        records.append(
            _record(
                "hinted_mask_local",
                index,
                prompt,
                f"<BIT>\nMask: {answer}\n</BIT>",
                {
                    "prg_seed": seed,
                    "edge_index": edge_index,
                    "edge": edge,
                    "selector": hint,
                    "gold_bit": answer,
                },
            )
        )
    return records


def _hinted_update_records(
    n: int, *, seeds: list[str], graph: list[list[int]], rng: random.Random
) -> list[dict[str, Any]]:
    records = []
    for index in range(n):
        seed = seeds[index % len(seeds)]
        step = 1 + (index % (len(graph) - 1))
        previous_edge = graph[step - 1]
        current_edge = graph[step]
        c_previous, left, right = rng.randrange(2), rng.randrange(2), rng.randrange(2)
        z_previous = mask_bit(seed, previous_edge)
        z_current = mask_bit(seed, current_edge)
        answer = c_previous ^ z_previous ^ left ^ right ^ z_current
        previous_hint, current_hint = selector(previous_edge), selector(current_edge)
        prompt = (
            f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
            "Compute one encrypted-state update without revealing either mask.\n"
            "c_i = c_(i-1) XOR z_(i-1) XOR x_(2i-1) XOR x_(2i) XOR z_i\n\n"
            f"c_(i-1): {c_previous}\nPrevious selector: {previous_hint}\n"
            f"Current input pair: {left} {right}\nCurrent selector: {current_hint}\n\n"
            "Return exactly:\n<BIT>\nEncrypted state: 0 or 1\n</BIT>"
        )
        records.append(
            _record(
                "hinted_update_local",
                index,
                prompt,
                f"<BIT>\nEncrypted state: {answer}\n</BIT>",
                {
                    "prg_seed": seed,
                    "step": step,
                    "previous_edge": previous_edge,
                    "current_edge": current_edge,
                    "previous_selector": previous_hint,
                    "current_selector": current_hint,
                    "c_previous": c_previous,
                    "input_pair": [left, right],
                    "gold_bit": answer,
                },
            )
        )
    return records


def _hinted_start_records(
    n: int, *, seeds: list[str], graph: list[list[int]], rng: random.Random
) -> list[dict[str, Any]]:
    records = []
    edge = graph[0]
    hint = selector(edge)
    for index in range(n):
        seed = seeds[index % len(seeds)]
        left, right = rng.randrange(2), rng.randrange(2)
        answer = left ^ right ^ mask_bit(seed, edge)
        prompt = (
            f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
            "Compute the first encrypted cumulative parity state. The initial unencrypted state is 0.\n"
            f"First input pair: {left} {right}\nCurrent selector: {hint}\n\n"
            "Return exactly:\n<BIT>\nEncrypted state: 0 or 1\n</BIT>"
        )
        records.append(
            _record(
                "hinted_start_local",
                index,
                prompt,
                f"<BIT>\nEncrypted state: {answer}\n</BIT>",
                {
                    "prg_seed": seed,
                    "edge": edge,
                    "selector": hint,
                    "input_pair": [left, right],
                    "gold_bit": answer,
                },
            )
        )
    return records


def _hinted_trace_records(
    *, rng: random.Random, seeds: list[str], n: int, length: int, graph: list[list[int]]
) -> list[dict[str, Any]]:
    if n % 4:
        raise ValueError("Trace dataset size must be divisible by four.")
    task = f"hinted_joint_{length}"
    records: list[dict[str, Any]] = []
    pair_index = 0
    for expected_parity in (0, 1):
        made = 0
        while made < n // 4:
            input_bits = "".join(str(rng.randrange(2)) for _ in range(2 * length))
            if parity(input_bits) != expected_parity:
                continue
            seed = seeds[pair_index % len(seeds)]
            mask_values = masks(seed, graph, length)
            selectors = [selector(edge) for edge in graph[:length]]
            for member, member_bits in enumerate((input_bits, complement(input_bits))):
                gold = encrypted(member_bits, mask_values)
                prompt = (
                    f"<PRIVATE_INPUT>\n{bits_text(member_bits)}\n</PRIVATE_INPUT>\n"
                    f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
                    "For each public CoT line, emit the given public seed-position selector and "
                    "then the encrypted cumulative parity state. Privately compute its mask using "
                    "z = a XOR b XOR c XOR (d AND e). Never emit a mask or seed value.\n\n"
                    "Return exactly the public selector/state lines, followed by the answer."
                )
                trace = "\n".join(f"{hint} {bit}" for hint, bit in zip(selectors, gold))
                target = f"<COT>\n{trace}\n</COT>\n<ANSWER>\nParity: {parity(member_bits)}\n</ANSWER>"
                records.append(
                    _record(
                        task,
                        len(records),
                        prompt,
                        target,
                        {
                            "input_bits": member_bits,
                            "prg_seed": seed,
                            "length": length,
                            "selectors": selectors,
                            "gold_masks": mask_values,
                            "plain_states": states(member_bits),
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
        compact = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
        digest.update(compact)
        (directory / f"{copied['experiment_id']}.json").write_text(
            json.dumps(copied, indent=2) + "\n", encoding="utf-8"
        )
    return digest.hexdigest()


def _take(
    records: list[dict[str, Any]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    shuffled = records.copy()
    rng.shuffle(shuffled)
    return [
        json.loads(json.dumps(shuffled[index % len(shuffled)])) for index in range(n)
    ]


def _mixed_stage(
    current: list[dict[str, Any]],
    local_mask: list[dict[str, Any]],
    local_start: list[dict[str, Any]],
    local_update: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = [
        *_take(current, int(0.70 * n), rng),
        *_take(local_mask, int(0.10 * n), rng),
        *_take(local_start, int(0.05 * n), rng),
        *_take(local_update, int(0.10 * n), rng),
        *_take(previous, n - int(0.95 * n), rng),
    ]
    rng.shuffle(rows)
    return rows


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = random.Random(int(config["seed"]))
    lengths = [
        *map(int, config["core_lengths"]),
        *map(int, config["extension_lengths"]),
    ]
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["max_steps"]), int(config["graph_seed"])
    )
    all_seeds = [
        f"{value:0{config['seed_bits']}b}"
        for value in range(2 ** int(config["seed_bits"]))
    ]
    rng.shuffle(all_seeds)
    train_count, val_count = int(config["trace_train_n"]), int(
        config["trace_validation_n"]
    )
    train_seeds = all_seeds[:train_count]
    val_seeds = all_seeds[train_count : train_count + val_count]
    if set(train_seeds) & set(val_seeds):
        raise RuntimeError("Training and validation seeds overlap.")

    train: dict[str, list[dict[str, Any]]] = {
        "predicate_local": _predicate_records(int(config["local_train_n"])),
        "hinted_mask_local": _hinted_mask_records(
            int(config["local_train_n"]), seeds=train_seeds, graph=graph
        ),
        "hinted_start_local": _hinted_start_records(
            int(config["local_train_n"]),
            seeds=train_seeds,
            graph=graph,
            rng=random.Random(f"{config['seed']}:train:start"),
        ),
        "hinted_update_local": _hinted_update_records(
            int(config["local_train_n"]),
            seeds=train_seeds,
            graph=graph,
            rng=random.Random(f"{config['seed']}:train:update"),
        ),
    }
    validation: dict[str, list[dict[str, Any]]] = {
        "predicate_local": _predicate_records(
            int(config["local_validation_n"]), offset=11
        ),
        "hinted_mask_local": _hinted_mask_records(
            int(config["local_validation_n"]), seeds=val_seeds, graph=graph, offset=7
        ),
        "hinted_start_local": _hinted_start_records(
            int(config["local_validation_n"]),
            seeds=val_seeds,
            graph=graph,
            rng=random.Random(f"{config['seed']}:validation:start"),
        ),
        "hinted_update_local": _hinted_update_records(
            int(config["local_validation_n"]),
            seeds=val_seeds,
            graph=graph,
            rng=random.Random(f"{config['seed']}:validation:update"),
        ),
    }
    for length in lengths:
        task = f"hinted_joint_{length}"
        train[task] = _hinted_trace_records(
            rng=random.Random(f"{config['seed']}:train:{length}"),
            seeds=train_seeds,
            n=train_count,
            length=length,
            graph=graph,
        )
        validation[task] = _hinted_trace_records(
            rng=random.Random(f"{config['seed']}:validation:{length}"),
            seeds=val_seeds,
            n=val_count,
            length=length,
            graph=graph,
        )

    digests: dict[str, str] = {}
    for split, datasets in (("base_train", train), ("validation", validation)):
        for task, rows in datasets.items():
            digests[f"{split}/{task}"] = _write(output_root / split / task, rows)

    n = int(config["curriculum_train_n"])
    mix_rng = random.Random(int(config["seed"]) + 1731)
    core: dict[str, list[dict[str, Any]]] = {
        "predicate_local": _take(train["predicate_local"], n, mix_rng),
        "hinted_mask_local": [
            *_take(train["hinted_mask_local"], int(0.8 * n), mix_rng),
            *_take(train["predicate_local"], n - int(0.8 * n), mix_rng),
        ],
        "hinted_start_local": [
            *_take(train["hinted_start_local"], int(0.7 * n), mix_rng),
            *_take(train["hinted_mask_local"], int(0.2 * n), mix_rng),
            *_take(train["predicate_local"], n - int(0.9 * n), mix_rng),
        ],
        "hinted_update_local": [
            *_take(train["hinted_update_local"], int(0.6 * n), mix_rng),
            *_take(train["hinted_start_local"], int(0.2 * n), mix_rng),
            *_take(train["hinted_mask_local"], int(0.1 * n), mix_rng),
            *_take(train["predicate_local"], n - int(0.9 * n), mix_rng),
        ],
    }
    previous = train["hinted_update_local"]
    for length in map(int, config["core_lengths"]):
        task = f"hinted_joint_{length}"
        core[task] = _mixed_stage(
            train[task],
            train["hinted_mask_local"],
            train["hinted_start_local"],
            train["hinted_update_local"],
            previous,
            n,
            mix_rng,
        )
        previous = train[task]
    extension: dict[str, list[dict[str, Any]]] = {}
    for length in map(int, config["extension_lengths"]):
        task = f"hinted_joint_{length}"
        extension[task] = _mixed_stage(
            train[task],
            train["hinted_mask_local"],
            train["hinted_start_local"],
            train["hinted_update_local"],
            previous,
            n,
            mix_rng,
        )
        previous = train[task]
    for track, stages in (("core", core), ("extension", extension)):
        for task, rows in stages.items():
            mix_rng.shuffle(rows)
            digests[f"curriculum/{track}/{task}"] = _write(
                output_root / "curriculum" / track / task, rows
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
        "train_unique_seeds": len(train_seeds),
        "validation_unique_seeds": len(val_seeds),
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
        json.dumps({k: v for k, v in manifest.items() if k != "fixed_graph"}, indent=2)
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
        default=Path("generated_data/parity_goldreich_explicit_hints/seed_0"),
    )
    args = parser.parse_args()
    generate(args.config, args.output_root)


if __name__ == "__main__":
    main()
