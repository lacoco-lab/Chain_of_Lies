#!/usr/bin/env python3
"""Isolated batching benchmark and evaluation-only continuation; never trains."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def valid_response(path, experiment_id, cap):
    value = json.loads(path.read_text())
    ids = value.get("generated_token_ids")
    if (
        value.get("experiment_id") != experiment_id
        or not isinstance(value.get("raw_text"), str)
        or not value["raw_text"].strip()
        or not isinstance(ids, list)
        or not ids
        or len(ids) > cap
        or any(type(x) is not int or x < 0 for x in ids)
    ):
        raise ValueError(
            f"Invalid saved response: {path}; preserve and inspect it before proceeding"
        )
    return value


def select_prompts(data, difficulties, count):
    selected = []
    for length in difficulties:
        files = sorted((data / "per_length" / str(length)).glob("*.json"))
        if len(files) < count:
            raise ValueError(f"Insufficient prompts at length {length}")
        # Spread deterministic picks through each bucket rather than taking only its first records.
        selected.extend(files[(i * len(files)) // count] for i in range(count))
    return selected


def cache_options(inference, action, reset):
    """Legacy helpers suffice for one-shot continuation, but not warm benchmark timing."""
    parameters = inspect.signature(inference).parameters
    if {"reset_model_cache", "keep_model_cache"} <= set(parameters):
        return {"reset_model_cache": reset, "keep_model_cache": True}
    if action == "benchmark":
        raise RuntimeError(
            "Benchmark requires the current chain_of_lies/evaluation/experiment_evaluation.py; sync that file first."
        )
    return {}


def grouping_decision(mixed_seconds, grouped_seconds, differences):
    return (
        not differences
        and mixed_seconds > 0
        and 0 < grouped_seconds <= mixed_seconds / 1.10
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["benchmark", "evaluate"])
    p.add_argument(
        "--task", choices=["knowledge", "s5", "multiplication", "parity"], required=True
    )
    p.add_argument("--model", choices=["qwen", "llama"], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument(
        "--run-id",
        required=True,
        help="New isolated output name; reuse it only to resume evaluation",
    )
    p.add_argument("--per-length", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--check-length-grouping",
        action="store_true",
        help="S5 only: compare batch-2 orders in this allocation, then continue with the supported order",
    )
    a = p.parse_args()
    if a.check_length_grouping and (
        a.action != "evaluate" or a.batch_size != 2 or a.task != "s5"
    ):
        p.error("Length-grouping check requires evaluate --task s5 --batch-size 2")
    if (
        a.per_length < 1
        or a.batch_size < 1
        or Path(a.run_id).name != a.run_id
        or a.run_id in (".", "..")
    ):
        p.error("Positive sizes and a simple run-id are required")
    os.chdir(ROOT)
    from experiments.scaling_regime.protocols.filler.experiment import (
        load_config,
        _training,
        _training_complete,
    )
    from chain_of_lies.evaluation.experiment_evaluation import run_variant_inference
    from chain_of_lies.evaluation.rewards import summarize_variant_results
    from chain_of_lies.inference import clear_model_cache
    import torch

    # Check compatibility before creating any benchmark run directory.
    cache_options(run_variant_inference, a.action, True)
    if a.check_length_grouping:
        cache_options(run_variant_inference, "benchmark", True)

    config = load_config(
        ROOT / "experiments/scaling_regime/protocols/full_cot/config.json"
    )
    spec = config["tasks"][a.task]
    settings = _training(spec, a.seed)
    cap = settings["evaluation_max_new_tokens"]
    condition = (
        Path("artifacts/full_cot_scaling_v1")
        / a.task
        / a.model
        / f"seed_{a.seed}"
        / "full_cot"
    )
    adapter = condition / spec["variant"]
    if not _training_complete(adapter):
        raise RuntimeError(
            "No completed training checkpoint; this tool cannot train or resume training"
        )
    checkpoint = adapter / "ckpt_final"
    weights = checkpoint / "adapter_model.safetensors"
    data = (
        Path("generated_data/full_cot_scaling_v1") / a.task / a.model / f"seed_{a.seed}"
    )
    provenance = json.loads((condition / "provenance.json").read_text())
    derived = hashlib.sha256()
    for split in ("train", "eval"):
        files = sorted((data / split).glob("*.json"))
        if len(files) != provenance["counts"][split]:
            raise ValueError(f"Incomplete derived {split} data")
        for file in files:
            if file.stem != json.loads(file.read_text())["experiment_id"]:
                raise ValueError(f"Prompt filename/ID mismatch: {file}")
            derived.update(split.encode() + file.name.encode() + file.read_bytes())
    if derived.hexdigest() != provenance["derived_records_sha256"]:
        raise ValueError("Derived data differs from training provenance")
    if a.check_length_grouping:
        covered = set()
        for length in spec["difficulty_values"]:
            files = list((data / "per_length" / str(length)).glob("*.json"))
            if len(files) != spec["eval_examples_per_difficulty"]:
                raise ValueError(f"Incomplete grouping view at length {length}")
            for file in files:
                record = json.loads(file.read_text())
                if (
                    str(record["spec"][spec["difficulty_field"]]) != str(length)
                    or file.name in covered
                    or file.read_bytes() != (data / "eval" / file.name).read_bytes()
                ):
                    raise ValueError(
                        f"Grouping view differs from matched evaluation data: {file}"
                    )
                covered.add(file.name)
        if covered != {file.name for file in (data / "eval").glob("*.json")}:
            raise ValueError("Grouping views do not cover the complete evaluation set")
    work = (
        Path("artifacts/full_cot_evaluation_efficiency")
        / a.run_id
        / a.task
        / a.model
        / f"seed_{a.seed}"
    )
    binding = {
        "task": a.task,
        "model": a.model,
        "seed": a.seed,
        "checkpoint_sha256": digest(weights),
        "adapter_config_sha256": digest(checkpoint / "adapter_config.json"),
        "derived_records_sha256": derived.hexdigest(),
        "max_new_tokens": cap,
        "do_sample": False,
        "action": a.action,
        "batch_size": a.batch_size,
        "per_length": a.per_length,
    }
    if a.check_length_grouping:
        binding["length_grouping_policy"] = (
            "raw_text_agreement_and_10_percent_speedup_on_12_S5_prompts_v1"
        )
    if (work / "binding.json").exists():
        if json.loads((work / "binding.json").read_text()) != binding:
            raise ValueError("Run binding changed; choose a new run-id")
        if a.action == "benchmark":
            raise ValueError("Benchmark run already exists; use a fresh run-id")
    atomic_json(work / "binding.json", binding)

    def infer(prompts, responses, batch, reset=False):
        run_variant_inference(
            prompts,
            responses,
            model_id=str(checkpoint),
            max_new_tokens=cap,
            temperature=0.0,
            do_sample=False,
            resume=True,
            batch_size=batch,
            **cache_options(run_variant_inference, a.action, reset),
        )

    try:
        if a.action == "benchmark":
            selected = select_prompts(data, spec["difficulty_values"], a.per_length)
            prompts = work / "benchmark_prompts"
            prompts.mkdir(parents=True, exist_ok=True)
            for file in selected:
                shutil.copy2(file, prompts / file.name)
            warmup = work / "warmup_prompts"
            warmup.mkdir()
            # Includes the longest bucket; excludes loading and warm-up from measured runs.
            shutil.copy2(selected[-1], warmup / selected[-1].name)
            infer(warmup, work / "warmup_responses", 1, reset=True)
            results = []
            reference = None
            for batch in (2, 4, 8):
                responses = work / f"batch_{batch}"
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    infer(prompts, responses, batch)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    outputs = {
                        json.loads(f.read_text())["experiment_id"]: json.loads(
                            f.read_text()
                        )
                        for f in responses.glob("*.json")
                    }
                    if len(outputs) != len(selected):
                        raise ValueError("Incomplete benchmark outputs")
                    for eid in outputs:
                        valid_response(responses / f"{eid}.json", eid, cap)
                    if reference is None:
                        reference = outputs
                    differences = [
                        eid
                        for eid in outputs
                        if outputs[eid]["generated_token_ids"]
                        != reference[eid]["generated_token_ids"]
                        or outputs[eid]["raw_text"] != reference[eid]["raw_text"]
                    ]
                    results.append(
                        {
                            "batch_size": batch,
                            "seconds": elapsed,
                            "prompts_per_second": len(outputs) / elapsed,
                            "tokens_per_second": sum(
                                len(v["generated_token_ids"]) for v in outputs.values()
                            )
                            / elapsed,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated()
                            / 2**30,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved()
                            / 2**30,
                            "different_outputs_vs_batch_2": differences,
                            "metrics": summarize_variant_results(prompts, responses),
                        }
                    )
                except torch.cuda.OutOfMemoryError:
                    results.append({"batch_size": batch, "out_of_memory": True})
                    atomic_json(
                        work / "benchmark.json",
                        {"binding": binding, "results": results},
                    )
                    break
                atomic_json(
                    work / "benchmark.json",
                    {
                        "binding": binding,
                        "gpu": torch.cuda.get_device_name(0),
                        "torch": torch.__version__,
                        "results": results,
                        "note": "Small diagnostic sample, not paper results. Compare agreement AND throughput; batch 8 is not automatically recommended.",
                    },
                )
            print(json.dumps(results, indent=2), flush=True)
        else:
            # Never write into or interfere with the currently running canonical evaluator.
            responses = work / "responses"
            responses.mkdir(parents=True, exist_ok=True)
            source = (
                Path("generated_data/full_cot_scaling_v1_eval_responses")
                / a.task
                / a.model
                / f"seed_{a.seed}"
                / "all"
            )
            prompts = sorted((data / "eval").glob("*.json"))
            expected_ids = {json.loads(f.read_text())["experiment_id"] for f in prompts}
            if len(expected_ids) != len(prompts):
                raise ValueError("Duplicate experiment IDs")
            if any(f.stem not in expected_ids for f in responses.glob("*.json")):
                raise ValueError("Unexpected response IDs in continuation directory")
            reused = []
            for eid in sorted(expected_ids):
                destination = responses / f"{eid}.json"
                if destination.exists():
                    valid_response(destination, eid, cap)
                elif (source / destination.name).exists():
                    value = valid_response(source / destination.name, eid, cap)
                    atomic_json(destination, value)
                    reused.append(eid)
            # Persist provenance across resumes rather than claiming mixed-batch outputs all use the new batch.
            reuse_file = work / "reused_canonical_ids.json"
            prior = json.loads(reuse_file.read_text()) if reuse_file.exists() else []
            atomic_json(reuse_file, sorted(set(prior + reused)))
            grouped = False
            if a.check_length_grouping:
                decision_file = work / "grouping_check.json"
                if decision_file.exists():
                    decision = json.loads(decision_file.read_text())
                else:
                    diagnostic = work / "grouping_diagnostic"
                    if diagnostic.exists():
                        # Timing resumed subsets would be misleading. Preserve interrupted attempts,
                        # but repeat this small diagnostic in full before deciding.
                        diagnostic.rename(
                            work / f"grouping_diagnostic_interrupted_{time.time_ns()}"
                        )
                    diagnostic.mkdir()
                    lengths = [1, 2, 9, 10, 18, 19]
                    sample = select_prompts(data, lengths, 2)
                    mixed_prompts = diagnostic / "mixed_prompts"
                    warmup = diagnostic / "warmup"
                    mixed_prompts.mkdir()
                    warmup.mkdir()
                    for file in sample:
                        shutil.copy2(file, mixed_prompts / file.name)
                        length = str(
                            json.loads(file.read_text())["spec"][
                                spec["difficulty_field"]
                            ]
                        )
                        bucket = diagnostic / "grouped_prompts" / length
                        bucket.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, bucket / file.name)
                    shutil.copy2(sample[-1], warmup / sample[-1].name)
                    print(
                        "[Grouping check] Warm-up, then compare 12 identical prompts at batch 2.",
                        flush=True,
                    )
                    infer(warmup, diagnostic / "warmup_responses", 1, reset=True)
                    timings = {}
                    for order in ("mixed", "grouped"):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        output = diagnostic / f"{order}_responses"
                        if order == "mixed":
                            infer(mixed_prompts, output, 2)
                        else:
                            for length in lengths:
                                infer(
                                    diagnostic / "grouped_prompts" / str(length),
                                    output,
                                    2,
                                )
                        torch.cuda.synchronize()
                        timings[order] = time.perf_counter() - started
                        print(
                            f"[Grouping check] {order}: {timings[order]:.1f} seconds",
                            flush=True,
                        )
                    differences = []
                    for file in sample:
                        eid = json.loads(file.read_text())["experiment_id"]
                        left = valid_response(
                            diagnostic / "mixed_responses" / f"{eid}.json", eid, cap
                        )
                        right = valid_response(
                            diagnostic / "grouped_responses" / f"{eid}.json", eid, cap
                        )
                        if left["raw_text"] != right["raw_text"]:
                            differences.append(eid)
                    decision = {
                        "batch_size": 2,
                        "sample_lengths": lengths,
                        "sample_count": len(sample),
                        "seconds": timings,
                        "different_raw_text_ids": differences,
                        "use_grouped": grouping_decision(
                            timings["mixed"], timings["grouped"], differences
                        ),
                        "note": "Padding lengths can differ. Sample agreement is not full-dataset identity; unchanged checkpoint/data/cap/greedy settings. Fixed-order timing is a diagnostic, not a definitive benchmark.",
                    }
                    atomic_json(decision_file, decision)
                grouped = decision["use_grouped"]
                print("[Grouping check] Decision:", json.dumps(decision), flush=True)
            if grouped:
                # Keep the same model loaded across all lengths; skip copied or saved responses.
                for index, length in enumerate(spec["difficulty_values"]):
                    print(
                        f"[Full-CoT grouped] length {length}, batch size 2", flush=True
                    )
                    infer(
                        data / "per_length" / str(length),
                        responses,
                        2,
                        reset=(index == 0 and not a.check_length_grouping),
                    )
            else:
                infer(
                    data / "eval",
                    responses,
                    a.batch_size,
                    reset=not a.check_length_grouping,
                )
            for eid in expected_ids:
                valid_response(responses / f"{eid}.json", eid, cap)
            for length in spec["difficulty_values"]:
                metrics = summarize_variant_results(
                    data / "per_length" / str(length), responses
                )
                if (
                    metrics.get("missing_responses")
                    or metrics["num_examples"] != spec["eval_examples_per_difficulty"]
                ):
                    raise ValueError(f"Incomplete length {length}")
                atomic_json(
                    work / "per_length_eval" / f"length_{length}.json",
                    {
                        "task": a.task,
                        "model": a.model,
                        "seed": a.seed,
                        "condition": "full_cot",
                        "length": length,
                        "metrics": metrics,
                        "new_response_batch_size": a.batch_size,
                        "new_response_batching_order": (
                            "within_length" if grouped else "filename_order"
                        ),
                        "reused_canonical_ids_file": str(reuse_file),
                        "concealment_metric_applicable": False,
                    },
                )
            print(f"Complete isolated evaluation: {work}", flush=True)
    finally:
        clear_model_cache()


if __name__ == "__main__":
    main()
