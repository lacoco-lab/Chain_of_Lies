#!/usr/bin/env python3
"""
Generate larger train/validation prompt splits for the RL experiments.

This keeps the original 10-example benchmark directories untouched and writes
new prompt pools under a dedicated root directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl import SELECTED_RULE_BASED_RL_VARIANTS
from chain_of_lies.stage1_arithmetic import generate_arithmetic_prompt
from chain_of_lies.stage1_graph_prompt import generate_experiment_prompt
from chain_of_lies.stage1_graph_prompt.benign_malign import generate_experiment_prompt_benign_malign


def _write_payload(payload: dict[str, Any], out_dir: Path, experiment_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{experiment_id}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _graph_payload(variant_name: str, seed: int, experiment_id: str, nodes: int, edge_probability: float) -> dict[str, Any]:
    if variant_name == "graph_benign_malign":
        exp = generate_experiment_prompt_benign_malign(
            n_nodes=nodes,
            edge_probability=edge_probability,
            seed=seed,
            experiment_id=experiment_id,
        )
        return {
            "task_type": "graph_benign_malign",
            "experiment_id": exp.experiment_id,
            "prompt_text": exp.prompt_text,
            "spec": {
                "edges_text": exp.spec.edges_text,
                "start_node": exp.spec.start_node,
                "public_target": exp.spec.public_target,
                "private_target": exp.spec.private_target,
                "node_list": exp.spec.node_list,
                "edge_list": [list(edge) for edge in exp.spec.edge_list],
                "public_distance": exp.spec.public_distance,
                "private_distance": exp.spec.private_distance,
            },
        }

    prompt_variant = "latent_cot" if variant_name == "graph_latent_cot" else "default"
    exp = generate_experiment_prompt(
        n_nodes=nodes,
        edge_probability=edge_probability,
        seed=seed,
        experiment_id=experiment_id,
        prompt_variant=prompt_variant,
    )
    return {
        "experiment_id": exp.experiment_id,
        "prompt_variant": prompt_variant,
        "prompt_text": exp.prompt_text,
        "spec": {
            "edges_text": exp.spec.edges_text,
            "start_node": exp.spec.start_node,
            "public_target": exp.spec.public_target,
            "private_target": exp.spec.private_target,
            "node_list": exp.spec.node_list,
            "edge_list": [list(edge) for edge in exp.spec.edge_list],
            "public_distance": exp.spec.public_distance,
            "private_distance": exp.spec.private_distance,
        },
    }


def _arithmetic_payload(variant_name: str, seed: int, experiment_id: str, max_summand: int) -> dict[str, Any]:
    difficulty_variant = variant_name.removeprefix("arith_")
    prompt_text, _, spec = generate_arithmetic_prompt(
        max_summand=max_summand,
        seed=seed,
        experiment_id=experiment_id,
        difficulty_variant=difficulty_variant,
    )
    return {
        "task_type": "arithmetic",
        "experiment_id": experiment_id,
        "difficulty_variant": difficulty_variant,
        "prompt_text": prompt_text,
        "spec": {
            "public_question": spec.public_question,
            "private_question": spec.private_question,
            "public_answer": spec.public_answer,
            "private_answer": spec.private_answer,
        },
    }


def _make_payload(
    variant_name: str,
    seed: int,
    experiment_id: str,
    *,
    nodes: int,
    edge_probability: float,
    max_summand: int,
) -> dict[str, Any]:
    if variant_name.startswith("graph_"):
        return _graph_payload(variant_name, seed, experiment_id, nodes, edge_probability)
    return _arithmetic_payload(variant_name, seed, experiment_id, max_summand)


def _generate_split(
    variant_name: str,
    split_name: str,
    count: int,
    out_dir: Path,
    *,
    seed_offset: int,
    nodes: int,
    edge_probability: float,
    max_summand: int,
) -> None:
    for index in range(count):
        seed = seed_offset + index
        prefix = "exp" if variant_name.startswith("graph_") else "arith"
        experiment_id = f"{prefix}_{split_name}_{index:05d}"
        payload = _make_payload(
            variant_name,
            seed,
            experiment_id,
            nodes=nodes,
            edge_probability=edge_probability,
            max_summand=max_summand,
        )
        _write_payload(payload, out_dir, experiment_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/validation prompt splits for the selected RL variants.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all_selected",
        help="Variant name or 'all_selected'.",
    )
    parser.add_argument("--train-n", type=int, default=1000, help="Number of train prompts per variant.")
    parser.add_argument("--val-n", type=int, default=100, help="Number of validation prompts per variant.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for deterministic prompt generation.")
    parser.add_argument("--nodes", type=int, default=15, help="Number of nodes for graph variants.")
    parser.add_argument("--p", type=float, default=0.3, help="Edge probability for graph variants.")
    parser.add_argument("--max-summand", type=int, default=99, help="Max summand for easy arithmetic expressions.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/RL_splits"),
        help="Root directory for the generated train/validation prompt pools.",
    )
    args = parser.parse_args()

    variants = (
        list(SELECTED_RULE_BASED_RL_VARIANTS)
        if args.variant == "all_selected"
        else [args.variant]
    )

    for variant_name in variants:
        variant_root = args.output_root / variant_name
        train_dir = variant_root / "train_prompts"
        val_dir = variant_root / "val_prompts"
        print(f"[SplitGen] variant={variant_name} train={train_dir} val={val_dir}", flush=True)
        _generate_split(
            variant_name,
            "train",
            args.train_n,
            train_dir,
            seed_offset=args.seed,
            nodes=args.nodes,
            edge_probability=args.p,
            max_summand=args.max_summand,
        )
        _generate_split(
            variant_name,
            "val",
            args.val_n,
            val_dir,
            seed_offset=args.seed + args.train_n,
            nodes=args.nodes,
            edge_probability=args.p,
            max_summand=args.max_summand,
        )
        print(
            f"[SplitGen] wrote {args.train_n} train prompts and {args.val_n} validation prompts for {variant_name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
