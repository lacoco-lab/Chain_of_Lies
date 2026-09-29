#!/usr/bin/env python3
"""Validate the paper datasets and generate a deterministic 1-fact addition set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

OFFICIAL_REPOSITORY = "https://github.com/kaleybrauer/filler-token-reasoning"
OFFICIAL_COMMIT = "09f96a615abdbee31d2350bfbe064f4632c8a105"
OFFICIAL_HASHES = {
    "2fact_addition_dataset.json": "6e947077dba4621b644c71c9c0a7b7df3ea51535b3aaa375d16463b46fae9b13",
    "element_letter_positions.json": "7700eba04f8c5b3d306f63aedfe0493c965a671e6e21f08fe66263fe2296e057",
    "capital_letter_position.json": "80c1c765282df267f2d0a2d683dba81888522b72da77443d0293af17dc68cedc",
}

# Evaluation facts match the paper's factual-retrieval-plus-addition form. They are screened again
# on Llama by the component probes; no result assumes that Llama knows every fact.
EVAL_ATOMIC_FACTS: tuple[tuple[str, int], ...] = (
    ("iridium", 77),
    ("carbon", 6),
    ("tungsten", 74),
    ("oxygen", 8),
    ("gold", 79),
    ("silver", 47),
    ("sodium", 11),
    ("chlorine", 17),
    ("iron", 26),
    ("copper", 29),
    ("helium", 2),
    ("neon", 10),
    ("uranium", 92),
    ("lead", 82),
    ("silicon", 14),
    ("aluminum", 13),
    ("calcium", 20),
    ("potassium", 19),
    ("nickel", 28),
    ("cobalt", 27),
)

FEW_SHOT_ATOMIC_FACTS: tuple[tuple[str, int], ...] = (
    ("hydrogen", 1),
    ("nitrogen", 7),
    ("fluorine", 9),
    ("magnesium", 12),
    ("zinc", 30),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _validate_official_data(vendor_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for filename, expected_hash in OFFICIAL_HASHES.items():
        path = vendor_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing vendored paper dataset: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {filename}: {actual_hash} != {expected_hash}"
            )
        datasets[filename] = json.loads(path.read_text(encoding="utf-8"))
    if len(datasets["2fact_addition_dataset.json"]["examples"]) != 1500:
        raise RuntimeError("Expected 1,500 official two-fact examples.")
    if len(datasets["element_letter_positions.json"]["examples"]) != 285:
        raise RuntimeError("Expected 285 official element letter-position examples.")
    if len(datasets["capital_letter_position.json"]["examples"]) != 362:
        raise RuntimeError("Expected 362 official capital letter-position examples.")
    return datasets


def _make_one_fact_dataset(seed: int, n: int = 800) -> dict[str, Any]:
    rng = random.Random(seed)
    few_shot = []
    for index, (element, value) in enumerate(FEW_SHOT_ATOMIC_FACTS):
        addend = 11 + 7 * index
        few_shot.append(
            {
                "fact_phrase": f"the atomic number of {element}",
                "fact_question": f"What is the atomic number of {element}?",
                "fact_value": value,
                "x": addend,
                "answer": value + addend,
            }
        )

    candidates = [
        {
            "fact_phrase": f"the atomic number of {element}",
            "fact_question": f"What is the atomic number of {element}?",
            "fact_value": value,
            "x": addend,
            "answer": value + addend,
        }
        for element, value in EVAL_ATOMIC_FACTS
        for addend in range(10, 100)
    ]
    rng.shuffle(candidates)
    examples = []
    for index, item in enumerate(candidates[:n]):
        examples.append({"idx": index, **item})
    return {
        "task": "1fact_addition",
        "fact_type": "atomic_number",
        "few_shot_facts": few_shot,
        "examples": examples,
    }


def prepare(*, config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_root = config_path.parent
    vendor_root = experiment_root / "vendor"
    official = _validate_official_data(vendor_root)
    one_fact = _make_one_fact_dataset(int(config["seed"]))
    expected_one_fact = int(config["tasks"]["one_fact"]["expected_examples"])
    if len(one_fact["examples"]) != expected_one_fact:
        raise RuntimeError(f"Expected {expected_one_fact} one-fact examples.")

    output_root.mkdir(parents=True, exist_ok=True)
    one_fact_path = output_root / "one_fact_addition.json"
    one_fact_path.write_text(json.dumps(one_fact, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_dataset_sha256": OFFICIAL_HASHES,
        "official_counts": {
            "two_fact": len(official["2fact_addition_dataset.json"]["examples"]),
            "element_letter_position": len(
                official["element_letter_positions.json"]["examples"]
            ),
            "capital_letter_position": len(
                official["capital_letter_position.json"]["examples"]
            ),
        },
        "generated_one_fact_count": len(one_fact["examples"]),
        "generated_one_fact_sha256": _canonical_sha256(one_fact),
        "filler_lengths": config["filler_lengths"],
        "config_sha256": _sha256(config_path),
        "generator_sha256": _sha256(Path(__file__)),
    }
    (output_root / "manifest.json").write_text(
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
        "--output-root",
        type=Path,
        default=Path("generated_data/llama31_8b_knowledge_filler"),
    )
    args = parser.parse_args()
    prepare(config_path=args.config, output_root=args.output_root)


if __name__ == "__main__":
    main()
