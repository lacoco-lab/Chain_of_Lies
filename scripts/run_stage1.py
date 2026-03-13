import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.config import get_default_paths
from chain_of_lies.stage1_graph_prompt import (
    PROMPT_VARIANT_DEFAULT,
    generate_experiment_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chain-of-Lies experiment prompts")
    parser.add_argument("--n", type=int, default=1, help="Number of experiments to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--nodes", type=int, default=15, help="Number of nodes in graph")
    parser.add_argument("--p", type=float, default=0.3, help="Edge probability")
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default=PROMPT_VARIANT_DEFAULT,
        choices=["default", "latent_cot"],
        help="default=covert CoT, answer both paths; latent_cot=first N words hidden",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: data/prompts)")
    args = parser.parse_args()

    paths = get_default_paths()
    paths.ensure_dirs()
    out_dir = args.out_dir or paths.prompts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        seed = (args.seed + i) if args.seed is not None else None
        exp = generate_experiment_prompt(
            n_nodes=args.nodes,
            edge_probability=args.p,
            seed=seed,
            experiment_id=f"exp_{i:04d}",
            prompt_variant=args.prompt_variant,
        )
        payload = {
            "experiment_id": exp.experiment_id,
            "prompt_variant": args.prompt_variant,
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

    print(f"Generated {args.n} prompt(s) in {out_dir}")


if __name__ == "__main__":
    main()
