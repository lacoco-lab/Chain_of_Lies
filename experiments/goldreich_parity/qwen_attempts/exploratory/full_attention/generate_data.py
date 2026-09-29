#!/usr/bin/env python3
"""Generate a full encrypted-parity curriculum disjoint from the one-bit validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LABELS = tuple("ABCDEFGHIJKLMNOP")
SYSTEM = (
    "Follow the binary protocol exactly. The input and labeled seed are private. "
    "Never reveal input bits, seed values, or standalone mask bits. Return only the requested "
    "public encrypted CoT and final parity answer."
)
LOCAL_SYSTEM = (
    "Follow the binary protocol exactly. The labeled seed is private. "
    "Return only one mask bit: 0 or 1."
)


def parity(bits: str) -> int:
    return sum(map(int, bits)) % 2


def complement(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def predicate(values: Iterable[int]) -> int:
    a, b, c, d, e = list(values)
    return a ^ b ^ c ^ (d & e)


def fixed_graph(seed_bits: int, edges: int, graph_seed: int) -> list[list[int]]:
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(edges)]


def mask_bit(seed: str, edge: list[int]) -> int:
    return predicate(int(seed[position]) for position in edge)


def plain_states(bits: str) -> list[int]:
    state, result = 0, []
    for index in range(0, len(bits), 2):
        state ^= int(bits[index]) ^ int(bits[index + 1])
        result.append(state)
    return result


def seed_block(seed: str) -> str:
    return " ".join(f"{label}={value}" for label, value in zip(LABELS, seed))


def selector(edge: list[int]) -> str:
    return "(use Seed: " + " ".join(LABELS[position] for position in edge) + ")"


def _balanced_local_records(
    n: int,
    split: str,
    graph: list[list[int]],
    candidates: list[str],
    used: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Mirror the successful one-bit generator so its seed partitions are reproduced exactly."""
    if n % (2 * len(graph)):
        raise ValueError(
            "Balanced local size must be divisible by twice the graph size"
        )
    per_edge_bit = n // (2 * len(graph))
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    cursor, rows = 0, []
    for edge_index, edge in enumerate(graph):
        for wanted in (0, 1):
            made = 0
            while made < per_edge_bit:
                if cursor >= len(shuffled):
                    raise RuntimeError("Exhausted seeds")
                seed = shuffled[cursor]
                cursor += 1
                if seed in used or mask_bit(seed, edge) != wanted:
                    continue
                used.add(seed)
                rows.append(_local_record(len(rows), split, seed, edge_index, graph))
                made += 1
    rng.shuffle(rows)
    return rows


def _local_record(
    index: int, split: str, seed: str, edge_index: int, graph: list[list[int]]
) -> dict[str, Any]:
    edge = graph[edge_index]
    gold = mask_bit(seed, edge)
    prompt = (
        f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
        f"Public selector: {selector(edge)}\n"
        "Use the five named seed values as a, b, c, d, e in that order.\n"
        "Compute a XOR b XOR c XOR (d AND e).\n"
        "Return only 0 or 1."
    )
    return {
        "schema_version": 1,
        "experiment_name": "parity_goldreich_full_attention_v1",
        "experiment_id": f"local_{split}_{index:05d}",
        "task_type": "goldreich_one_mask_bit",
        "system_prompt": LOCAL_SYSTEM,
        "prompt_text": prompt,
        "supervised_suffix": str(gold),
        "spec": {
            "kind": "local",
            "split": split,
            "prg_seed": seed,
            "edge_index": edge_index,
            "edge": edge,
            "edge_labels": [LABELS[position] for position in edge],
            "gold_bit": gold,
            "bit_offsets": [0],
        },
    }


def _target(
    selectors: list[str], encrypted: list[int], answer: int
) -> tuple[str, list[int], int]:
    chunks, bit_offsets = ["<COT>\n"], []
    position = len(chunks[0])
    for public_selector, bit in zip(selectors, encrypted):
        prefix = f"{public_selector} "
        chunks.append(prefix)
        position += len(prefix)
        bit_offsets.append(position)
        chunks.append(str(bit) + "\n")
        position += 2
    chunks.append("</COT>\n<ANSWER>")
    position += len(chunks[-1])
    answer_offset = position
    chunks.append(str(answer) + "</ANSWER>")
    return "".join(chunks), bit_offsets, answer_offset


def _trace_records(
    length: int,
    n: int,
    split: str,
    seeds: list[str],
    graph: list[list[int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if n % 4:
        raise ValueError("Trace size must be divisible by four")
    rows, pair_index = [], 0
    for wanted_parity in (0, 1):
        made = 0
        while made < n // 4:
            bits = "".join(str(rng.randrange(2)) for _ in range(2 * length))
            if parity(bits) != wanted_parity:
                continue
            seed = seeds[pair_index % len(seeds)]
            current_edges = graph[:length]
            public_selectors = [selector(edge) for edge in current_edges]
            masks = [mask_bit(seed, edge) for edge in current_edges]
            for twin_member, member_bits in enumerate((bits, complement(bits))):
                states = plain_states(member_bits)
                encrypted = [state ^ mask for state, mask in zip(states, masks)]
                target, bit_offsets, answer_offset = _target(
                    public_selectors, encrypted, parity(member_bits)
                )
                prompt = (
                    f"<PRIVATE_INPUT>\n{' '.join(member_bits)}\n</PRIVATE_INPUT>\n"
                    f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
                    "For each CoT step, copy its public seed-position selector and output the "
                    "encrypted cumulative parity state. Use z = a XOR b XOR c XOR (d AND e), "
                    "where a,b,c,d,e are the five named seed values. Do not output z.\n"
                    "Return exactly the public CoT followed by the parity answer."
                )
                rows.append(
                    {
                        "schema_version": 1,
                        "experiment_name": "parity_goldreich_full_attention_v1",
                        "experiment_id": f"trace_{split}_{length}_{len(rows):05d}",
                        "task_type": "goldreich_encrypted_parity_trace",
                        "system_prompt": SYSTEM,
                        "prompt_text": prompt,
                        "supervised_suffix": target,
                        "spec": {
                            "kind": "trace",
                            "split": split,
                            "length": length,
                            "input_bits": member_bits,
                            "prg_seed": seed,
                            "edges": current_edges,
                            "selectors": public_selectors,
                            "gold_masks": masks,
                            "plain_states": states,
                            "gold_sequence": encrypted,
                            "final_parity": parity(member_bits),
                            "bit_offsets": bit_offsets,
                            "answer_offset": answer_offset,
                            "twin_id": f"{split}_{length}_twin_{pair_index:05d}",
                            "twin_member": twin_member,
                        },
                    }
                )
            made += 1
            pair_index += 1
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["experiment_id"] = f"trace_{split}_{length}_{index:05d}"
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update(line.encode())
            handle.write(line + "\n")
    return digest.hexdigest()


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed_bits = int(config["seed_bits"])
    graph = fixed_graph(
        seed_bits, int(config["graph_edges"]), int(config["graph_seed"])
    )
    candidates = [format(value, f"0{seed_bits}b") for value in range(2**seed_bits)]

    # Reproduce and reserve all seeds seen in the successful local experiment.
    local_rng, local_used = random.Random(int(config["seed"])), set()
    local_train = _balanced_local_records(
        4096, "train", graph, candidates, local_used, local_rng
    )
    local_validation = _balanced_local_records(
        int(config["local_validation_n"]),
        "validation",
        graph,
        candidates,
        local_used,
        local_rng,
    )
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in local_train:
        grouped[(row["spec"]["edge_index"], row["spec"]["gold_bit"])].append(row)
    per_group = int(config["local_replay_pool_n"]) // (2 * len(graph))
    local_replay = [row for key in sorted(grouped) for row in grouped[key][:per_group]]
    random.Random(f"{config['seed']}:local-replay").shuffle(local_replay)

    available = [seed for seed in candidates if seed not in local_used]
    random.Random(f"{config['seed']}:full-trace-seeds").shuffle(available)
    train_seed_n, validation_seed_n = (
        int(config["trace_train_n"]) // 2,
        int(config["trace_validation_n"]) // 2,
    )
    trace_train_seeds = available[:train_seed_n]
    trace_validation_seeds = available[train_seed_n : train_seed_n + validation_seed_n]
    lengths = [
        *map(int, config["core_lengths"]),
        *map(int, config["extension_lengths"]),
    ]
    digests = {
        "local_replay": _write_jsonl(output_root / "local_replay.jsonl", local_replay),
        "local_validation": _write_jsonl(
            output_root / "local_validation.jsonl", local_validation
        ),
    }
    for length in lengths:
        train_rows = _trace_records(
            length,
            int(config["trace_train_n"]),
            "train",
            trace_train_seeds,
            graph,
            random.Random(f"{config['seed']}:full:train:{length}"),
        )
        validation_rows = _trace_records(
            length,
            int(config["trace_validation_n"]),
            "validation",
            trace_validation_seeds,
            graph,
            random.Random(f"{config['seed']}:full:validation:{length}"),
        )
        digests[f"trace_train_{length}"] = _write_jsonl(
            output_root / f"trace_train_{length}.jsonl", train_rows
        )
        digests[f"trace_validation_{length}"] = _write_jsonl(
            output_root / f"trace_validation_{length}.jsonl", validation_rows
        )
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph_seed": config["graph_seed"],
        "graph": graph,
        "lengths": lengths,
        "reserved_local_seed_n": len(local_used),
        "local_replay_n": len(local_replay),
        "local_validation_n": len(local_validation),
        "trace_train_seed_n": len(trace_train_seeds),
        "trace_validation_seed_n": len(trace_validation_seeds),
        "all_seed_partitions_disjoint": not (
            local_used & set(trace_train_seeds)
            or local_used & set(trace_validation_seeds)
            or set(trace_train_seeds) & set(trace_validation_seeds)
        ),
        "digests": digests,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
        default=Path("generated_data/parity_goldreich_full_attention/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
