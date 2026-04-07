#!/usr/bin/env python3
"""Unified prompt generator for graph and arithmetic Chain-of-Lies variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.config import get_default_paths
from chain_of_lies.stage1_arithmetic import generate_arithmetic_prompt
from chain_of_lies.stage1_graph_prompt import generate_experiment_prompt
from chain_of_lies.stage1_graph_prompt.benign_malign import generate_experiment_prompt_benign_malign


VARIANT_CHOICES = (
    "graph_default",
    "graph_latent_cot",
    "graph_benign_malign",
    "arith_default",
    "arith_public_hard",
    "arith_private_hard",
    "arith_both_hard",
)


def _default_out_dir(variant: str) -> Path:
    paths = get_default_paths()
    mapping = {
        "graph_default": paths.prompts_dir,
        "graph_latent_cot": Path("data/Without_self_eval/prompts_latent_cot"),
        "graph_benign_malign": Path("data/Without_self_eval/prompts_benign_malign"),
        "arith_default": Path("data/Without_self_eval/prompts_arithmetic"),
        "arith_public_hard": Path("data/Without_self_eval/prompts_arithmetic_public_hard"),
        "arith_private_hard": Path("data/Without_self_eval/prompts_arithmetic_private_hard"),
        "arith_both_hard": Path("data/Without_self_eval/prompts_arithmetic_both_hard"),
    }
    return mapping[variant]


def _graph_payload(variant: str, *, seed: int | None, experiment_id: str, nodes: int, edge_probability: float) -> dict:
    if variant == "graph_benign_malign":
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
                "edge_list": [list(e) for e in exp.spec.edge_list],
                "public_distance": exp.spec.public_distance,
                "private_distance": exp.spec.private_distance,
            },
        }

    prompt_variant = "latent_cot" if variant == "graph_latent_cot" else "default"
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
            "edge_list": [list(e) for e in exp.spec.edge_list],
            "public_distance": exp.spec.public_distance,
            "private_distance": exp.spec.private_distance,
        },
    }


def _arith_payload(variant: str, *, seed: int | None, experiment_id: str, max_summand: int) -> dict:
    difficulty_variant = variant.removeprefix("arith_")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompts for any Chain-of-Lies variant.")
    parser.add_argument("--variant", type=str, required=True, choices=VARIANT_CHOICES)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--max-summand", type=int, default=99)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = get_default_paths()
    paths.ensure_dirs()
    out_dir = args.out_dir or _default_out_dir(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.n):
        seed = (args.seed + index) if args.seed is not None else None
        if args.variant.startswith("graph_"):
            experiment_id = f"exp_{index:04d}" if args.variant != "graph_benign_malign" else f"exp_bm_{index:04d}"
            payload = _graph_payload(
                args.variant,
                seed=seed,
                experiment_id=experiment_id,
                nodes=args.nodes,
                edge_probability=args.p,
            )
        else:
            experiment_id = f"arith_{index:04d}"
            payload = _arith_payload(
                args.variant,
                seed=seed,
                experiment_id=experiment_id,
                max_summand=args.max_summand,
            )
        out_file = out_dir / f"{payload['experiment_id']}.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_file}")

    print(f"Generated {args.n} prompt(s) for {args.variant} in {out_dir}")


if __name__ == "__main__":
    main()
