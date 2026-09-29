#!/usr/bin/env python3
"""Validate trajectory formulas, privacy boundaries, formatting, and seed partitions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from generate_data import fixed_graph, mask, read_jsonl

TRACE_LINE = re.compile(
    r"\(use Input: x[A-Z]+ x[A-Z]+; use Seed: [A-P](?: [A-P]){4}\) C=([01])"
)
ANSWER_LINE = re.compile(r"ANSWER=([01])")


def parse_target(text: str) -> tuple[list[int], int] | None:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "COT:":
        return None
    answer = ANSWER_LINE.fullmatch(lines[-1])
    trace = [TRACE_LINE.fullmatch(line) for line in lines[1:-1]]
    if answer is None or not trace or any(match is None for match in trace):
        return None
    return [int(match.group(1)) for match in trace if match is not None], int(
        answer.group(1)
    )


def validate(config_path: Path, source_root: Path, data_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    graph = fixed_graph(
        int(config["seed_bits"]), int(config["graph_edges"]), int(config["graph_seed"])
    )
    if manifest["graph"] != graph or not manifest["train_validation_seeds_disjoint"]:
        raise ValueError("Manifest graph or seed partition is invalid")
    source_train = {
        row["spec"]["prg_seed"]
        for row in read_jsonl(source_root / "derive_previous_train.jsonl")
    }
    source_validation = {
        row["spec"]["prg_seed"]
        for row in read_jsonl(source_root / "derive_previous_validation.jsonl")
    }
    report, trajectory_seeds = {}, {"train": set(), "validation": set()}
    for length in map(int, config["lengths"]):
        for split in ("train", "validation"):
            name = f"trajectory_n{length}_{split}"
            rows = read_jsonl(data_root / f"{name}.jsonl")
            expected_n = int(config[f"{split}_n_per_length"])
            if len(rows) != expected_n:
                raise ValueError(f"Wrong row count for {name}")
            counts = Counter(int(row["spec"]["final_parity"]) for row in rows)
            if counts != {0: expected_n // 2, 1: expected_n // 2}:
                raise ValueError(f"Unbalanced final parity in {name}: {counts}")
            for row in rows:
                spec = row["spec"]
                bits = str(spec["input_bits"])
                seed = str(spec["prg_seed"])
                if len(bits) != length or len(seed) != int(config["seed_bits"]):
                    raise ValueError(
                        f"Wrong input or seed length in {row['experiment_id']}"
                    )
                if seed not in (
                    source_train if split == "train" else source_validation
                ):
                    raise ValueError(f"Seed escaped the calibrated {split} partition")
                trajectory_seeds[split].add(seed)
                masks = [mask(seed, graph[index]) for index in range(length // 2)]
                state, plain, encrypted = 0, [], []
                for index in range(0, length, 2):
                    state ^= int(bits[index]) ^ int(bits[index + 1])
                    step = index // 2
                    plain.append(state)
                    encrypted.append(state ^ masks[step])
                parity = sum(map(int, bits)) % 2
                if (
                    spec["masks"] != masks
                    or spec["plain_states"] != plain
                    or spec["encrypted_states"] != encrypted
                    or int(spec["final_parity"]) != parity
                ):
                    raise ValueError(f"Incorrect trajectory in {row['experiment_id']}")
                parsed = parse_target(str(row["supervised_suffix"]))
                if not parsed or parsed[0] != encrypted:
                    raise ValueError(
                        f"Incorrect target format in {row['experiment_id']}"
                    )
                if parsed[1] != parity:
                    raise ValueError(
                        f"Incorrect target answer in {row['experiment_id']}"
                    )
                target = str(row["supervised_suffix"])
                if seed in target or bits in target:
                    raise ValueError(
                        f"Private string leaked into target in {row['experiment_id']}"
                    )
            report[name] = {"n": len(rows), "zeros": counts[0], "ones": counts[1]}
    if trajectory_seeds["train"] & trajectory_seeds["validation"]:
        raise ValueError("Trajectory train and validation seeds overlap")
    positions = set(manifest["transition_positions"])
    for split in ("train", "validation"):
        rows = read_jsonl(data_root / f"transition_{split}.jsonl")
        expected_n = int(config[f"transition_{split}_n"])
        if len(rows) != expected_n:
            raise ValueError(f"Wrong row count for transition_{split}")
        task_counts, bit_counts = Counter(), Counter()
        for row in rows:
            spec = row["spec"]
            step = int(spec["step"])
            seed = str(spec["prg_seed"])
            bits = str(spec["input_bits"])
            if step not in positions:
                raise ValueError(
                    f"Invalid transition position in {row['experiment_id']}"
                )
            if seed not in (source_train if split == "train" else source_validation):
                raise ValueError(f"Transition seed escaped the {split} partition")
            previous_mask = mask(seed, graph[step - 1])
            current_mask = mask(seed, graph[step])
            left, right = map(int, bits[2 * step : 2 * step + 2])
            wanted = (
                int(spec["previous_c"]) ^ left ^ right ^ previous_mask ^ current_mask
            )
            if (
                spec["previous_mask"] != previous_mask
                or spec["current_mask"] != current_mask
                or int(spec["gold_bit"]) != wanted
                or row["supervised_suffix"] != str(wanted)
            ):
                raise ValueError(f"Incorrect transition in {row['experiment_id']}")
            task_counts[step] += 1
            bit_counts[(step, wanted)] += 1
        per_position = expected_n // len(positions)
        if any(task_counts[position] != per_position for position in positions):
            raise ValueError(
                f"Unbalanced transition positions in {split}: {task_counts}"
            )
        if any(
            bit_counts[(position, bit)] != per_position // 2
            for position in positions
            for bit in (0, 1)
        ):
            raise ValueError(f"Unbalanced transition bits in {split}: {bit_counts}")
        report[f"transition_{split}"] = {
            "n": len(rows),
            "positions": len(positions),
            "per_position": per_position,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_final/seed_0"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_trajectory/seed_0"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.source_root, args.data_root), indent=2))


if __name__ == "__main__":
    main()
