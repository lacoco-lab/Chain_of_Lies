from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chain_of_lies.rl.rewards import summarize_variant_results
from chain_of_lies.stage2_inference import clear_model_cache, run_inference_batch
from chain_of_lies.types import ExperimentPrompt, GraphSpec


def load_prompt_for_inference(path: Path) -> ExperimentPrompt:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("task_type") == "arithmetic":
        spec = GraphSpec(
            edges_text="",
            start_node="",
            public_target="",
            private_target="",
        )
    else:
        spec_data = data["spec"]
        spec = GraphSpec(
            edges_text=spec_data["edges_text"],
            start_node=spec_data["start_node"],
            public_target=spec_data["public_target"],
            private_target=spec_data["private_target"],
            node_list=spec_data.get("node_list", []),
            edge_list=[tuple(edge) for edge in spec_data.get("edge_list", [])],
            public_distance=spec_data.get("public_distance"),
            private_distance=spec_data.get("private_distance"),
        )
    return ExperimentPrompt(
        prompt_text=data["prompt_text"],
        spec=spec,
        experiment_id=data["experiment_id"],
    )


def summarize_training_history(adapter_dir: Path) -> dict[str, Any]:
    history_path = adapter_dir / "train_history.json"
    metadata_path = adapter_dir / "training_metadata.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    if not history:
        return {
            "training_metadata": metadata,
            "history_points": 0,
        }

    best_by_reward = max(history, key=lambda item: item["avg_reward"])
    final = history[-1]
    return {
        "training_metadata": metadata,
        "history_points": len(history),
        "initial_avg_reward": history[0]["avg_reward"],
        "best_avg_reward": best_by_reward["avg_reward"],
        "best_step": best_by_reward["step"],
        "final_avg_reward": final["avg_reward"],
        "final_task_success_rate": final["task_success_rate"],
        "final_task_subgoal_rate": final.get("task_subgoal_rate"),
        "final_task_component_rate": final.get("task_component_rate"),
        "final_concealment_rate": final["concealment_rate"],
    }


def run_variant_inference(
    prompts_dir: Path,
    responses_dir: Path,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    resume: bool = True,
    batch_size: int = 8,
) -> None:
    ### Generate responses for one prompt directory using either a base model or an RL adapter.
    responses_dir.mkdir(parents=True, exist_ok=True)
    clear_model_cache()

    prompt_files = sorted(prompts_dir.glob("*.json"))
    if not prompt_files:
        raise ValueError(f"No prompt JSONs found in {prompts_dir}")

    progress_file = responses_dir / "progress.txt"
    pending: list[ExperimentPrompt] = []
    for index, prompt_path in enumerate(prompt_files, start=1):
        experiment = load_prompt_for_inference(prompt_path)
        output_path = responses_dir / f"{experiment.experiment_id}.json"
        if resume and output_path.exists():
            continue
        pending.append(experiment)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        if not batch:
            continue
        first_idx = batch_start + 1
        last_idx = batch_start + len(batch)
        print(
            f"[Eval] [{first_idx}-{last_idx}/{len(pending)}] running batch of {len(batch)} prompts",
            flush=True,
        )
        responses = run_inference_batch(
            batch,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        with progress_file.open("a", encoding="utf-8") as handle:
            for response in responses:
                output_path = responses_dir / f"{response.experiment_id}.json"
                output_path.write_text(
                    json.dumps(
                        {
                            "experiment_id": response.experiment_id,
                            "model_id": response.model_id,
                            "raw_text": response.raw_text,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                handle.write(f"{response.experiment_id}\n")
    clear_model_cache()


def compare_variant_results(
    *,
    prompts_dir: Path,
    baseline_responses_dir: Path,
    rl_responses_dir: Path,
) -> dict[str, Any]:
    baseline = summarize_variant_results(prompts_dir, baseline_responses_dir)
    rl = summarize_variant_results(prompts_dir, rl_responses_dir)
    return {
        "variant_name": rl["variant_name"],
        "baseline": baseline,
        "rl": rl,
        "delta": {
            "avg_reward": rl.get("avg_reward", 0.0) - baseline.get("avg_reward", 0.0),
            "task_success_rate": rl.get("task_success_rate", 0.0) - baseline.get("task_success_rate", 0.0),
            "task_subgoal_rate": rl.get("task_subgoal_rate", 0.0) - baseline.get("task_subgoal_rate", 0.0),
            "task_component_rate": rl.get("task_component_rate", 0.0) - baseline.get("task_component_rate", 0.0),
            "concealment_rate": rl.get("concealment_rate", 0.0) - baseline.get("concealment_rate", 0.0),
            "format_rate": rl.get("format_rate", 0.0) - baseline.get("format_rate", 0.0),
            "avg_cot_words": rl.get("avg_cot_words", 0.0) - baseline.get("avg_cot_words", 0.0),
        },
    }
