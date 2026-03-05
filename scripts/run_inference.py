#!/usr/bin/env python3
"""
Batch inference: load prompts from a directory, run Qwen on each, save responses.

Designed for HTCondor: run in background, survive disconnect, resume if interrupted.
- Writes each response as soon as it is done (no batch buffer).
- Skips experiment_id if output file already exists (--resume).
- Logs progress to stdout and to a progress file so you can tail -f job.out on Condor.

Usage (local):
  python scripts/run_inference.py --model Qwen/Qwen2.5-7B-Instruct
  python scripts/run_inference.py --model Qwen/Qwen2.5-3B-Instruct --resume

Usage (Condor): same script; run.sh sets env and calls this. See condor/README.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Project root on front of path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.config import get_default_paths
from chain_of_lies.types import ExperimentPrompt, GraphSpec
from chain_of_lies.stage2_inference import run_inference


def load_prompt(path: Path) -> ExperimentPrompt:
    """Load experiment from JSON written by run_stage1."""
    data = json.loads(path.read_text(encoding="utf-8"))
    spec_data = data["spec"]
    spec = GraphSpec(
        edges_text=spec_data["edges_text"],
        start_node=spec_data["start_node"],
        public_target=spec_data["public_target"],
        private_target=spec_data["private_target"],
        node_list=spec_data.get("node_list", []),
        edge_list=[tuple(e) for e in spec_data.get("edge_list", [])],
        public_distance=spec_data.get("public_distance"),
        private_distance=spec_data.get("private_distance"),
    )
    return ExperimentPrompt(
        prompt_text=data["prompt_text"],
        spec=spec,
        experiment_id=data["experiment_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen inference on generated prompts")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--prompts-dir", type=Path, default=None, help="Directory of prompt JSONs (default: data/prompts)")
    parser.add_argument("--responses-dir", type=Path, default=None, help="Directory to write response JSONs (default: data/responses)")
    parser.add_argument("--resume", action="store_true", help="Skip experiments that already have a response file")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    paths = get_default_paths()
    paths.ensure_dirs()
    prompts_dir = args.prompts_dir or paths.prompts_dir
    responses_dir = (
        args.responses_dir
        or (Path(os.environ["RESPONSES_DIR"]) if os.environ.get("RESPONSES_DIR") else paths.responses_dir)
    )
    responses_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(prompts_dir.glob("*.json"))
    if not prompt_files:
        print("No prompt JSONs found in", prompts_dir, file=sys.stderr)
        sys.exit(1)

    progress_file = responses_dir / "progress.txt"
    total = len(prompt_files)
    done = 0
    skipped = 0

    for i, path in enumerate(prompt_files):
        exp = load_prompt(path)
        out_path = responses_dir / f"{exp.experiment_id}.json"
        if args.resume and out_path.exists():
            skipped += 1
            print(f"[{i+1}/{total}] Skip (exists): {exp.experiment_id}")
            continue
        print(f"[{i+1}/{total}] Running: {exp.experiment_id} ...", flush=True)
        try:
            response = run_inference(
                exp,
                model_id=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            out_path.write_text(
                json.dumps({
                    "experiment_id": response.experiment_id,
                    "model_id": response.model_id,
                    "raw_text": response.raw_text,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            done += 1
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"{response.experiment_id}\n")
            print(f"[{i+1}/{total}] Done: {exp.experiment_id}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{total}] ERROR {exp.experiment_id}: {e}", flush=True, file=sys.stderr)
            raise

    print(f"Finished. Completed {done}, skipped {skipped}, total {total}. Responses in {responses_dir}")
    if progress_file.exists():
        print(f"Progress log: {progress_file}")


if __name__ == "__main__":
    main()
