#!/usr/bin/env python3
"""
Run Stage 1 (graph with benign/malign labels):

- Generate connected graphs.
- Randomly label nodes as Benign or Malign (with constraints).
- Instruct the model that the public path must use only Benign nodes.

This is a separate experiment for prediction #3 and does not affect the
existing default / latent_cot graph prompts.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.config import get_default_paths
from chain_of_lies.stage1_graph_prompt.benign_malign import (
    generate_experiment_prompt_benign_malign,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Lies graph prompts with Benign/Malign node labels (prediction #3)."
    )
    parser.add_argument("--n", type=int, default=1, help="Number of experiments to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--nodes", type=int, default=15, help="Number of nodes in graph")
    parser.add_argument("--p", type=float, default=0.3, help="Edge probability")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/Without_self_eval/prompts_benign_malign"),
        help="Output directory (default: data/Without_self_eval/prompts_benign_malign)",
    )
    args = parser.parse_args()

    paths = get_default_paths()
    paths.ensure_dirs()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        seed = (args.seed + i) if args.seed is not None else None
        exp = generate_experiment_prompt_benign_malign(
            n_nodes=args.nodes,
            edge_probability=args.p,
            seed=seed,
            experiment_id=f"exp_bm_{i:04d}",
        )
        payload = {
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
        out_file = out_dir / f"{exp.experiment_id}.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_file}")

    print(f"Generated {args.n} benign/malign prompt(s) in {out_dir}")


if __name__ == "__main__":
    main()

