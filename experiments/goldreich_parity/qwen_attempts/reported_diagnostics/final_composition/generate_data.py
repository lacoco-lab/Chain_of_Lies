#!/usr/bin/env python3
"""Generate the final balanced Goldreich composition curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LABELS = tuple("ABCDEFGHIJKLMNOP")
CALIBRATION_EXPERIMENT = "parity_goldreich_update_ablation_v1"
FINAL_EXPERIMENT = "parity_goldreich_final_v2_composition_curriculum"
# Digests written by the exact seed-0 generator used in the successful
# fresh-rank-32 diagnostic. Generation aborts if any calibrated row changes.
RANK32_CALIBRATION_DIGESTS = {
    "local_replay": "1b4255f3ba661e595a467718aa37f2eca90008a2ec6bf1eaf8e0c3774f1519ce",
    "local_validation": "ccd8fe4bc4784601ee855f018bb4cc0a86f482a73848e8bdb656ca7e30e9d6b8",
    "supplied_update_train": "4098041369241815223ea5597735c2944ce0decacdc0038b0de0304c042661be",
    "supplied_update_validation": "b7eb4fa7fa12acc2b5770fe543f4518feec8892f6a23b3b78e6942e9bdca664a",
    "derive_previous_train": "5c330017a4a6f1a5308e1540fffd0b28e7e614f11616a00caa9c138f032f5d59",
    "derive_previous_validation": "eb9a9e83ebedf3edd158fe13794345135a3015026e1a6aa0f3b182ee849f591e",
    "derive_current_train": "27eb60cafb1920cf8211269ec34fe0f896ea342dc104c0c18559e762b4169cd9",
    "derive_current_validation": "61410ba2c9c6a84ebce184dda3ff8bae00e0e1991b6498d746824fed26389f72",
}
SYSTEM = (
    "Follow the binary protocol exactly. Seed values are private. Return only the requested "
    "single bit: 0 or 1. Never reveal a seed value or an intermediate mask."
)


def predicate(values: Iterable[int]) -> int:
    a, b, c, d, e = list(values)
    return a ^ b ^ c ^ (d & e)


def fixed_graph(seed_bits: int, edge_count: int, graph_seed: int) -> list[list[int]]:
    rng = random.Random(graph_seed)
    return [rng.sample(range(seed_bits), 5) for _ in range(edge_count)]


def mask_bit(seed: str, edge: list[int]) -> int:
    return predicate(int(seed[position]) for position in edge)


def seed_block(seed: str) -> str:
    return " ".join(f"{label}={value}" for label, value in zip(LABELS, seed))


def selector(edge: list[int]) -> str:
    return "(use Seed: " + " ".join(LABELS[position] for position in edge) + ")"


def record(
    task: str,
    index: int,
    prompt: str,
    gold: int,
    spec: dict[str, Any],
    experiment_name: str = CALIBRATION_EXPERIMENT,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "experiment_id": f"{task}_{index:05d}",
        "task_type": task,
        "system_prompt": SYSTEM,
        "prompt_text": prompt,
        "supervised_suffix": str(gold),
        "spec": {"task": task, "gold_bit": gold, **spec},
    }


def local_record(
    index: int, seed: str, edge_index: int, graph: list[list[int]]
) -> dict[str, Any]:
    edge = graph[edge_index]
    gold = mask_bit(seed, edge)
    prompt = (
        f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
        f"Public selector: {selector(edge)}\n"
        "Use the five named values as a,b,c,d,e in order and compute "
        "a XOR b XOR c XOR (d AND e). Return only the mask bit."
    )
    return record(
        "local_mask",
        index,
        prompt,
        gold,
        {
            "prg_seed": seed,
            "edge_index": edge_index,
            "edge": edge,
            "attention_groups": [edge],
        },
    )


def reproduce_one_mask_splits(
    config: dict[str, Any], graph: list[list[int]]
) -> tuple[list[dict], list[dict], set[str]]:
    """Reproduce the successful rank-32 diagnostic's local replay exactly."""
    width = int(config["seed_bits"])
    candidates = [format(value, f"0{width}b") for value in range(2**width)]
    rng, used = random.Random(int(config["seed"])), set()

    def make(n: int) -> list[dict]:
        if n % (2 * len(graph)):
            raise ValueError("Local split must balance every edge and output")
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        cursor, rows = 0, []
        per_group = n // (2 * len(graph))
        for edge_index, edge in enumerate(graph):
            for wanted in (0, 1):
                made = 0
                while made < per_group:
                    seed = shuffled[cursor]
                    cursor += 1
                    if seed in used or mask_bit(seed, edge) != wanted:
                        continue
                    used.add(seed)
                    rows.append(local_record(len(rows), seed, edge_index, graph))
                    made += 1
        rng.shuffle(rows)
        return rows

    train = make(4096)
    validation = make(int(config["local_validation_n"]))
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in train:
        grouped[(row["spec"]["edge_index"], row["spec"]["gold_bit"])].append(row)
    per_group = int(config["local_replay_n"]) // (2 * len(graph))
    replay = [row for key in sorted(grouped) for row in grouped[key][:per_group]]
    random.Random(f"{config['seed']}:update-ablation-local").shuffle(replay)
    return replay, validation, used


def supplied_update_rows(n: int, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for index in range(n):
        wanted = index % 2
        previous_c, previous_z = rng.randrange(2), rng.randrange(2)
        left, right = rng.randrange(2), rng.randrange(2)
        current_z = wanted ^ previous_c ^ previous_z ^ left ^ right
        prompt = (
            f"Previous encrypted state c_prev: {previous_c}\n"
            f"Previous supplied mask z_prev: {previous_z}\n"
            f"Next input pair: {left} {right}\nCurrent supplied mask z_cur: {current_z}\n"
            "Compute c_cur = c_prev XOR z_prev XOR left XOR right XOR z_cur. "
            "Return only c_cur."
        )
        rows.append(
            record(
                "supplied_update",
                index,
                prompt,
                wanted,
                {
                    "split": split,
                    "previous_c": previous_c,
                    "previous_mask": previous_z,
                    "input_pair": [left, right],
                    "current_mask": current_z,
                    "attention_groups": [],
                },
            )
        )
    rng.shuffle(rows)
    return rows


def composition_rows(
    task: str,
    n: int,
    split: str,
    seeds: list[str],
    graph: list[list[int]],
    rng: random.Random,
) -> list[dict]:
    if task not in {"derive_previous", "derive_current", "full_update"}:
        raise ValueError(f"Unknown composition task: {task}")
    transitions = len(graph) - 1
    if n % (2 * transitions):
        raise ValueError("Composition size must balance every transition and output")
    rows, seed_index = [], 0
    per_group = n // (2 * transitions)
    for step in range(1, len(graph)):
        previous_edge, current_edge = graph[step - 1], graph[step]
        for wanted in (0, 1):
            for _ in range(per_group):
                seed = seeds[seed_index]
                seed_index += 1
                previous_z = mask_bit(seed, previous_edge)
                current_z = mask_bit(seed, current_edge)
                previous_c, left, right = (
                    rng.randrange(2),
                    rng.randrange(2),
                    rng.randrange(2),
                )
                if task == "derive_previous":
                    current_z = wanted ^ previous_c ^ previous_z ^ left ^ right
                    mask_lines = (
                        f"Previous selector: {selector(previous_edge)}\n"
                        f"Current supplied mask z_cur: {current_z}\n"
                    )
                    instruction = "Privately compute z_prev from the previous selector"
                    attention_groups = [previous_edge]
                elif task == "derive_current":
                    previous_z = wanted ^ previous_c ^ left ^ right ^ current_z
                    mask_lines = (
                        f"Previous supplied mask z_prev: {previous_z}\n"
                        f"Current selector: {selector(current_edge)}\n"
                    )
                    instruction = "Privately compute z_cur from the current selector"
                    attention_groups = [current_edge]
                else:
                    previous_c = wanted ^ previous_z ^ left ^ right ^ current_z
                    mask_lines = (
                        f"Previous selector: {selector(previous_edge)}\n"
                        f"Current selector: {selector(current_edge)}\n"
                    )
                    instruction = (
                        "Privately compute z_prev and z_cur from their selectors"
                    )
                    attention_groups = [previous_edge, current_edge]
                privacy_instruction = (
                    "do not output either mask"
                    if task == "full_update"
                    else "do not output the derived mask"
                )
                prompt = (
                    f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
                    f"Previous encrypted state c_prev: {previous_c}\n"
                    f"{mask_lines}Next input pair: {left} {right}\n"
                    f"{instruction}, then compute c_cur = c_prev XOR z_prev XOR left XOR right "
                    f"XOR z_cur. Return only c_cur; {privacy_instruction}."
                )
                rows.append(
                    record(
                        task,
                        len(rows),
                        prompt,
                        wanted,
                        {
                            "split": split,
                            "prg_seed": seed,
                            "step": step,
                            "previous_edge": previous_edge,
                            "current_edge": current_edge,
                            "previous_c": previous_c,
                            "input_pair": [left, right],
                            "previous_mask": previous_z,
                            "current_mask": current_z,
                            "attention_groups": attention_groups,
                        },
                        experiment_name=(
                            FINAL_EXPERIMENT
                            if task == "full_update"
                            else CALIBRATION_EXPERIMENT
                        ),
                    )
                )
    rng.shuffle(rows)
    return rows


def curriculum_rows(
    full_rows: list[dict[str, Any]], task: str, rng: random.Random
) -> list[dict[str, Any]]:
    """Turn paired full-update rows into progressively harder, truthful bridges.

    Every bridge uses exactly the same seed and adjacent selectors as the final
    endpoint.  The first target exposes only the XOR of the two masks during
    training; subsequent targets add the three public XOR terms one at a time.
    The endpoint itself remains answer-only and never emits a mask.
    """
    if task not in {"mask_delta", "full_zero", "full_one", "full_two"}:
        raise ValueError(f"Unknown curriculum task: {task}")
    rows = []
    for index, source in enumerate(full_rows):
        spec = source["spec"]
        seed = spec["prg_seed"]
        previous_edge = list(spec["previous_edge"])
        current_edge = list(spec["current_edge"])
        previous_z = int(spec["previous_mask"])
        current_z = int(spec["current_mask"])
        delta = previous_z ^ current_z
        common = (
            f"<PRIVATE_SEED>\n{seed_block(seed)}\n</PRIVATE_SEED>\n\n"
            f"Previous selector: {selector(previous_edge)}\n"
            f"Current selector: {selector(current_edge)}\n"
        )
        if task == "mask_delta":
            previous_c = left = right = 0
            gold = delta
            prompt = (
                common
                + "Privately compute z_prev and z_cur from their selectors, then compute "
                "d = z_prev XOR z_cur. Return only d; do not output either mask."
            )
        else:
            # Hardness is controlled only by how many public XOR terms are
            # non-constant. Later bridges are exactly balanced by construction.
            wanted = index % 2
            if task == "full_zero":
                previous_c = left = right = 0
                gold = delta
            else:
                left = rng.randrange(2) if task == "full_two" else 0
                right = 0
                previous_c = wanted ^ delta ^ left
                gold = wanted
            prompt = (
                common
                + f"Previous encrypted state c_prev: {previous_c}\n"
                + f"Next input pair: {left} {right}\n"
                + "Privately compute z_prev and z_cur from their selectors, then compute "
                "c_cur = c_prev XOR z_prev XOR left XOR right XOR z_cur. "
                "Return only c_cur; do not output either mask."
            )
        rows.append(
            record(
                task,
                index,
                prompt,
                gold,
                {
                    "split": spec["split"],
                    "prg_seed": seed,
                    "step": spec["step"],
                    "previous_edge": previous_edge,
                    "current_edge": current_edge,
                    "previous_c": previous_c,
                    "input_pair": [left, right],
                    "previous_mask": previous_z,
                    "current_mask": current_z,
                    "mask_delta": delta,
                    "attention_groups": [previous_edge, current_edge],
                },
                experiment_name=FINAL_EXPERIMENT,
            )
        )
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            copied = json.loads(json.dumps(row))
            copied["experiment_id"] = f"{copied['task_type']}_{index:05d}"
            line = json.dumps(copied, sort_keys=True, separators=(",", ":"))
            digest.update(line.encode())
            handle.write(line + "\n")
    return digest.hexdigest()


def generate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    local_replay, local_validation, reserved = reproduce_one_mask_splits(config, graph)
    width = int(config["seed_bits"])
    candidates = [
        format(value, f"0{width}b")
        for value in range(2**width)
        if format(value, f"0{width}b") not in reserved
    ]
    # These seed allocations and RNG labels are deliberately identical to the
    # successful fresh-rank-32 composition diagnostic.
    random.Random(f"{config['seed']}:update-ablation-seeds").shuffle(candidates)
    train_n = int(config["composition_train_n"])
    validation_n = int(config["composition_validation_n"])
    train_seeds = candidates[:train_n]
    validation_seeds = candidates[train_n : train_n + validation_n]
    datasets = {
        "local_replay": local_replay,
        "local_validation": local_validation,
        "supplied_update_train": supplied_update_rows(
            int(config["supplied_update_train_n"]),
            "train",
            random.Random("ablation:supplied:train:0"),
        ),
        "supplied_update_validation": supplied_update_rows(
            int(config["supplied_update_validation_n"]),
            "validation",
            random.Random("ablation:supplied:val:0"),
        ),
    }
    for task in ("derive_previous", "derive_current", "full_update"):
        train_rng = (
            random.Random(f"ablation:{task}:train:0")
            if task != "full_update"
            else random.Random("final:full_update:train")
        )
        validation_rng = (
            random.Random(f"ablation:{task}:val:0")
            if task != "full_update"
            else random.Random("final:full_update:validation")
        )
        datasets[f"{task}_train"] = composition_rows(
            task, train_n, "train", train_seeds, graph, train_rng
        )
        datasets[f"{task}_validation"] = composition_rows(
            task,
            validation_n,
            "validation",
            validation_seeds,
            graph,
            validation_rng,
        )
    # Directly supervise the missing two-mask composition before asking the
    # model to fold it into the encrypted-state update.  These rows are derived
    # from the endpoint rows, so seed/selector pairing cannot drift.
    for task in ("mask_delta", "full_zero", "full_one", "full_two"):
        for split in ("train", "validation"):
            datasets[f"{task}_{split}"] = curriculum_rows(
                datasets[f"full_update_{split}"],
                task,
                random.Random(f"final:{task}:{split}:0"),
            )
    digests = {
        name: write_jsonl(output_root / f"{name}.jsonl", rows)
        for name, rows in datasets.items()
    }
    observed_calibration = {name: digests[name] for name in RANK32_CALIBRATION_DIGESTS}
    calibration_matches = observed_calibration == RANK32_CALIBRATION_DIGESTS
    if bool(config.get("enforce_rank32_calibration", True)) and not calibration_matches:
        mismatches = {
            name: {"expected": expected, "observed": observed_calibration.get(name)}
            for name, expected in RANK32_CALIBRATION_DIGESTS.items()
            if observed_calibration.get(name) != expected
        }
        raise RuntimeError(f"Rank-32 calibration data drifted: {mismatches}")
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "graph": graph,
        "reserved_one_mask_seed_n": len(reserved),
        "composition_train_seed_n": len(train_seeds),
        "composition_validation_seed_n": len(validation_seeds),
        "all_seed_partitions_disjoint": not (
            reserved & set(train_seeds)
            or reserved & set(validation_seeds)
            or set(train_seeds) & set(validation_seeds)
        ),
        "paired_task_seeds": True,
        "full_curriculum": [
            "mask_delta",
            "full_zero",
            "full_one",
            "full_two",
            "full_update",
        ],
        "rank32_calibration_recipe": {
            "source_experiment": CALIBRATION_EXPERIMENT,
            "enforced": bool(config.get("enforce_rank32_calibration", True)),
            "digests_verified": calibration_matches,
            "digests": observed_calibration,
            "model_facing_datasets_reproduced_exactly": [
                "local_replay",
                "local_validation",
                "supplied_update_train",
                "supplied_update_validation",
                "derive_previous_train",
                "derive_previous_validation",
                "derive_current_train",
                "derive_current_validation",
            ],
            "training_rng": f"{config['seed']}:composition-rank",
        },
        "dataset_sizes": {name: len(rows) for name, rows in datasets.items()},
        "digests": digests,
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
        default=Path("generated_data/parity_goldreich_final/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_root), indent=2))


if __name__ == "__main__":
    main()
