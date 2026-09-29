import json
import random
import struct
import sys
import tempfile
import unittest
import os
import contextlib
import io
import subprocess
import tarfile
from unittest.mock import patch
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        )
    ),
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, response_valid, safetensors_valid


def sample_prompt(task: str, directory: Path) -> tuple[str, str]:
    saved = next(directory.glob("*.json"), None)
    if saved is not None:
        return saved.name, saved.read_text()

    examples = ROOT / "sample_records/scaling_regime/evaluation_examples.jsonl"
    if examples.is_file():
        for line in examples.read_text().splitlines():
            row = json.loads(line)
            if row["task"] == task and row["protocol"] == "vanilla":
                record = row["prompt_record"]
                return f'{record["experiment_id"]}.json', json.dumps(record)
    raise FileNotFoundError(f"No generated or released sample prompt for {task}")


class IntegrityTests(unittest.TestCase):
    def test_hold_is_idempotent_and_verifies_nonzero_exit(self):
        import repair_pre

        row = lambda status: {"ClusterId": 10, "ProcId": 0, "JobStatus": status}
        with (
            patch.object(repair_pre, "query", return_value=[row(5)]),
            patch("repair_pre.subprocess.run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            repair_pre.ensure_held(10)
            run.assert_not_called()
        with (
            patch.object(repair_pre, "query", side_effect=[[row(2)], [row(5)]]),
            patch(
                "repair_pre.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1),
            ),
        ):
            repair_pre.ensure_held(10)
        with (
            patch.object(repair_pre, "query", side_effect=[[row(2)], [row(2)]]),
            patch(
                "repair_pre.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1),
            ),
        ):
            with self.assertRaises(ValueError):
                repair_pre.ensure_held(10)

    def test_pre_repair_preserves_training_and_uses_absolute_python(self):
        import repair_pre, common, compact

        original = Path.cwd()
        manager = lambda cluster, status: {
            "ClusterId": cluster,
            "ProcId": 0,
            "JobStatus": status,
            "Cmd": "/usr/bin/condor_dagman",
        }
        train = {
            "ClusterId": 11,
            "ProcId": 0,
            "JobStatus": 2,
            "Args": "experiments/scaling_regime/protocols/filler/recovery_sic/run.sh train s5 qwen 0",
        }
        queries = [
            [manager(10, 5), train],
            [manager(20, 2)],
            [manager(10, 5), train],
            [manager(20, 2)],
            [manager(20, 5)],
        ]
        plan = [
            {"task": "s5", "model": "qwen", "seed": 0},
            {"task": "s5", "model": "qwen", "seed": 2},
        ]
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                compact.COMPACT.mkdir(parents=True)
                (common.WORK / "plan.json").write_text(json.dumps(plan))
                (compact.COMPACT / "transition.json").write_text(
                    json.dumps({"old_controller": 10})
                )
                for item in plan:
                    name = f's5_qwen_{item["seed"]}_eval'
                    (compact.COMPACT / f"{name}.sub").write_text(
                        f"output = {compact.COMPACT}/logs/{name}.out\n"
                    )
                with (
                    patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
                    patch.object(repair_pre, "query", side_effect=queries),
                    patch.object(
                        repair_pre,
                        "checkpoint_valid",
                        side_effect=[ValueError("training"), "weight"],
                    ),
                    patch("repair_pre.subprocess.run") as run,
                    patch.object(
                        sys, "argv", ["repair", "--failed-controller", "20", "--apply"]
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    repair_pre.main()
                self.assertEqual(
                    [c.args[0] for c in run.call_args_list], [["condor_hold", "20.0"]]
                )
                dag = (compact.COMPACT / "pre_fixed/evaluation.dag").read_text()
                self.assertEqual(dag.count("JOB "), 2)
                self.assertIn(str(Path(sys.executable).resolve()) + " -S", dag)
                self.assertIn("--controller 10 --controller 20", dag)
                self.assertNotIn("_train.sub", dag)
            finally:
                os.chdir(original)

    def test_inference_cache_reset_is_opt_in(self):
        from chain_of_lies.evaluation.experiment_evaluation import run_variant_inference
        import common

        config = json.loads(common.CONFIG.read_text())
        sample_name, sample_text = sample_prompt(
            "s5_state_tracking",
            common.ROOT
            / Path(config["tasks"]["s5"]["data_root"])
            / "shared_eval/length_1/s5_length_control",
        )
        with tempfile.TemporaryDirectory() as d:
            prompts = Path(d) / "prompts"
            responses = Path(d) / "responses"
            prompts.mkdir()
            responses.mkdir()
            (prompts / sample_name).write_text(sample_text)
            (responses / sample_name).write_text("{}")
            with patch(
                "chain_of_lies.evaluation.experiment_evaluation.clear_model_cache"
            ) as clear:
                run_variant_inference(
                    prompts,
                    responses,
                    model_id="same",
                    max_new_tokens=1536,
                    temperature=0.0,
                    keep_model_cache=True,
                )
                run_variant_inference(
                    prompts,
                    responses,
                    model_id="same",
                    max_new_tokens=1536,
                    temperature=0.0,
                    reset_model_cache=False,
                    keep_model_cache=True,
                )
                self.assertEqual(clear.call_count, 1)
                run_variant_inference(
                    prompts,
                    responses,
                    model_id="same",
                    max_new_tokens=1536,
                    temperature=0.0,
                )
                self.assertEqual(
                    clear.call_count, 3
                )  # Original default clears before and after.

    def test_ready_gate_waits_for_old_workers(self):
        import ready

        args = [
            "ready",
            "--controller",
            "10",
            "--task",
            "s5",
            "--model",
            "qwen",
            "--seed",
            "0",
        ]
        row = {
            "ClusterId": 11,
            "ProcId": 0,
            "JobStatus": 2,
            "Args": "experiments/scaling_regime/protocols/filler/recovery_sic/run.sh train s5 qwen 0",
        }
        with (
            patch.object(sys, "argv", args),
            patch.object(ready, "query", return_value=[row]),
            patch.object(ready, "checkpoint_valid") as validate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(ready.main(), 75)
            validate.assert_not_called()
        with (
            patch.object(sys, "argv", args),
            patch.object(ready, "query", return_value=[]),
            patch.object(ready, "checkpoint_valid", return_value="weight"),
        ):
            self.assertEqual(ready.main(), 0)

    def test_compact_worker_keeps_one_model_cache(self):
        import worker, common

        config = json.loads(common.CONFIG.read_text())
        original = Path.cwd()
        sample_name, text = sample_prompt(
            "s5_state_tracking",
            common.ROOT
            / Path(config["tasks"]["s5"]["data_root"])
            / "shared_eval/length_1/s5_length_control",
        )
        calls = []
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                shards = Path("shards")
                shards.mkdir()
                for part in range(2):
                    shard = shards / f"length_1_part_{part}"
                    (shard / "prompts").mkdir(parents=True)
                    (shard / "responses").mkdir()
                    file = shard / "prompts" / sample_name
                    file.write_text(text)
                    (shard / "binding.json").write_text(
                        json.dumps(
                            {
                                "checkpoint_sha256": None,
                                "prompts": {file.name: common.sha(file)},
                            }
                        )
                    )

                def fake_infer(prompts, responses, **kwargs):
                    calls.append(kwargs)
                    for file in prompts.glob("*.json"):
                        (responses / file.name).write_text(
                            json.dumps(
                                {
                                    "experiment_id": file.stem,
                                    "raw_text": "wrong",
                                    "generated_token_ids": [1],
                                }
                            )
                        )

                with (
                    patch.object(worker, "checkpoint_valid", return_value="weight"),
                    patch.object(
                        sys,
                        "argv",
                        [
                            "worker",
                            "eval",
                            "s5",
                            "qwen",
                            "2",
                            "--shards-root",
                            str(shards),
                        ],
                    ),
                    patch(
                        "chain_of_lies.evaluation.run_variant_inference",
                        side_effect=fake_infer,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    worker.main()
                self.assertEqual([c["reset_model_cache"] for c in calls], [True, False])
                self.assertTrue(all(c["keep_model_cache"] for c in calls))
                self.assertEqual(calls[0]["model_id"], calls[1]["model_id"])
                self.assertTrue(
                    all((s / "complete.json").exists() for s in shards.iterdir())
                )
            finally:
                os.chdir(original)

    def test_compact_transition_preserves_old_training_and_started_eval(self):
        import compact, common

        config = json.loads(common.CONFIG.read_text())
        original = Path.cwd()

        def row(cluster, action, task, model, seed, status, starts):
            return {
                "ClusterId": cluster,
                "ProcId": 0,
                "JobStatus": status,
                "NumJobStarts": starts,
                "Args": f"experiments/scaling_regime/protocols/filler/recovery_sic/run.sh {action} {task} {model} {seed}",
            }

        manager = {
            "ClusterId": 10,
            "ProcId": 0,
            "JobStatus": 2,
            "Cmd": "/usr/bin/condor_dagman",
        }
        jobs = [
            row(11, "train", "s5", "qwen", 0, 2, 1),
            row(12, "eval", "s5", "qwen", 2, 1, 0),
            row(13, "eval", "s5", "qwen", 2, 2, 1),
        ]
        queries = [
            [manager] + jobs,
            [dict(manager, JobStatus=5)] + jobs,
            [dict(manager, JobStatus=5)] + [jobs[0], jobs[2]],
        ]
        plan = [
            {"task": "s5", "model": "qwen", "seed": 0},
            {"task": "s5", "model": "qwen", "seed": 2},
        ]
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                common.WORK.mkdir(parents=True)
                (common.WORK / "plan.json").write_text(json.dumps(plan))
                with (
                    patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
                    patch.object(compact, "query", side_effect=queries),
                    patch.object(
                        compact,
                        "checkpoint_valid",
                        side_effect=[ValueError("in progress"), "weight"],
                    ),
                    patch("compact.subprocess.run") as run,
                    patch.object(
                        sys, "argv", ["compact", "--controller", "10", "--apply"]
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    compact.main()
                self.assertEqual(run.call_args_list[0].args[0], ["condor_hold", "10.0"])
                self.assertEqual(len(run.call_args_list), 2)
                remove = run.call_args_list[1].args[0]
                self.assertEqual(remove[:2], ["condor_rm", "-constraint"])
                self.assertIn("ClusterId == 12", remove[2])
                self.assertIn("NumJobStarts == 0", remove[2])
                record = json.loads((compact.COMPACT / "transition.json").read_text())
                self.assertEqual(record["additional_training"], [])
                self.assertEqual(record["preserved_old_jobs"], ["11.0", "13.0"])
                dag = (compact.COMPACT / "compact.dag").read_text()
                self.assertEqual(dag.count("JOB "), 2)
                self.assertIn("SCRIPT DEFER 75 60 PRE", dag)
                sub = (compact.COMPACT / "s5_qwen_2_eval.sub").read_text()
                self.assertIn("--shards-root", sub)
                self.assertIn("ON_EXIT_OR_EVICT", sub)
            finally:
                os.chdir(original)

    def test_ssh_stream_capture_and_safe_extract(self):
        from capture import capture_ssh, extract_snapshot

        original = Path.cwd()
        real_run = subprocess.run
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                artifact = Path(
                    "artifacts/filler_only_scaling_v1/s5/qwen/seed_0/filler_only"
                )
                artifact.mkdir(parents=True)
                (artifact / "weights.bin").write_bytes(b"test weights")
                response = Path(
                    "generated_data/filler_only_scaling_v1_eval_responses/s5/qwen/seed_0/filler_only"
                )
                response.mkdir(parents=True)
                (response / "a.json").write_text("{}")
                destination = Path("snapshot")
                destination.mkdir()

                def emulate_ssh(argv, **kwargs):
                    self.assertEqual(
                        argv[:4], ["condor_ssh_to_job", "-ssh", "ssh -T", "63136.0"]
                    )
                    return real_run(["/bin/sh", "-c", argv[-1]], **kwargs)

                with (
                    patch("capture.subprocess.run", side_effect=emulate_ssh),
                    (destination / "log").open("w") as log,
                ):
                    self.assertEqual(
                        capture_ssh("63136.0", "s5", "qwen", 0, destination, log), 0
                    )
                self.assertEqual(
                    (destination / artifact / "weights.bin").read_bytes(),
                    b"test weights",
                )
                self.assertTrue((destination / response / "a.json").exists())
                unsafe = Path("unsafe.tar.gz")
                with tarfile.open(unsafe, "w:gz") as bundle:
                    entry = tarfile.TarInfo("../escape")
                    bundle.addfile(entry)
                with self.assertRaises(ValueError):
                    extract_snapshot(unsafe, destination, [str(artifact)])
                archive = destination / "outputs.tar.gz"
                archive.write_bytes(archive.read_bytes()[:-5])
                with self.assertRaises(EOFError):
                    extract_snapshot(archive, destination, [str(artifact)])
            finally:
                os.chdir(original)

    def test_wrapped_parity_report(self):
        from audit import report_metrics

        expected = {"num_examples": 1000, "private_exact_rate": 0.5}
        wrapped = {"reports": [{"variant_name": "parity_control", "rl": expected}]}
        self.assertEqual(report_metrics(wrapped, "rl", "parity_control"), expected)
        self.assertEqual(
            report_metrics({"rl": expected}, "rl", "parity_control"), expected
        )
        with self.assertRaises(ValueError):
            report_metrics(wrapped, "rl", "wrong_variant")

    def test_snapshot_plan_and_publish(self):
        import common, plan, finalize
        from capture import JOBS
        from experiments.scaling_regime.protocols.filler.experiment import (
            _directories,
            _training,
        )

        config = json.loads(common.CONFIG.read_text())
        samples = {}
        for task in ["s5", "parity"]:
            spec = config["tasks"][task]
            directory = common.ROOT / (
                _directories(spec, 0)[1]
                if task == "parity"
                else Path(spec["data_root"]) / "shared_eval/length_1" / spec["variant"]
            )
            samples[task] = sample_prompt(
                "s5_state_tracking" if task == "s5" else "plain_parity",
                directory,
            )
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                for task in ["s5", "parity"]:
                    spec = config["tasks"][task]
                    spec["difficulty_values"] = spec["difficulty_values"][:1]
                    spec["train_examples_per_difficulty"] = 1
                    spec["eval_examples_per_difficulty"] = 1
                    spec["data_root"] = f"data/{task}"
                    spec.pop("source_manifest_pattern", None)
                    spec.pop("source_configs_by_seed", None)
                    spec["source_manifest"] = f"data/{task}/manifest.json"
                    spec["source_config"] = f"data/{task}/config.json"
                    Path(spec["data_root"]).mkdir(parents=True)
                    Path(spec["source_manifest"]).write_text("{}")
                    Path(spec["source_config"]).write_text("{}")
                    for seed in range(3):
                        for directory in _directories(spec, seed):
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / samples[task][0]).write_text(samples[task][1])
                    if task == "s5":
                        directory = (
                            Path(spec["data_root"])
                            / "shared_eval/length_1"
                            / spec["variant"]
                        )
                        directory.mkdir(parents=True)
                        (directory / samples[task][0]).write_text(samples[task][1])
                local = Path("config.json")
                local.write_text(json.dumps(config))
                snapshot = Path("snapshot")
                snapshot.mkdir()
                capture = []
                for job, (task, model, seed) in JOBS.items():
                    spec = config["tasks"][task]
                    settings = _training(spec, seed)
                    steps = settings["epochs"]
                    root = (
                        snapshot / job / common.canonical_condition(task, model, seed)
                    )
                    adapter = root / spec["variant"]
                    final = adapter / "ckpt_final"
                    final.mkdir(parents=True)
                    metadata = {
                        "base_model": config["models"][model],
                        "supervision_mode": "filler_only",
                        "steps": steps,
                        "train_examples_seen": steps,
                        "target_train_examples_seen": steps,
                        "selection_criterion": "fixed_final_epoch",
                        "filler_token_counts": spec["filler_token_counts"],
                        "filler_token_count_field": spec["difficulty_field"],
                    }
                    for key in [
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "lora_r",
                        "lora_alpha",
                        "lora_dropout",
                        "memory_efficient_ce",
                        "activation_cpu_offload",
                    ]:
                        metadata[key] = settings[key]
                    for directory in [adapter, final]:
                        (directory / "training_metadata.json").write_text(
                            json.dumps(metadata)
                        )
                    (adapter / "train_history.json").write_text(
                        json.dumps([{"step": i} for i in range(1, steps + 1)])
                    )
                    (final / "adapter_config.json").write_text(
                        json.dumps(
                            {
                                "r": settings["lora_r"],
                                "lora_alpha": settings["lora_alpha"],
                            }
                        )
                    )
                    header = json.dumps(
                        {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
                    ).encode()
                    (final / "adapter_model.safetensors").write_bytes(
                        struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
                    )
                    provenance = root / "provenance"
                    provenance.mkdir()
                    (provenance / "cell_manifest.json").write_text(
                        json.dumps(
                            {
                                "task": task,
                                "model": model,
                                "seed": seed,
                                "training": settings,
                                "filler_token_counts": spec["filler_token_counts"],
                                "source_manifest_sha256": common.sha(
                                    spec["source_manifest"]
                                ),
                                "source_experiment_config_sha256": common.sha(
                                    spec["source_config"]
                                ),
                            }
                        )
                    )
                    responses = (
                        snapshot / job / common.canonical_responses(task, model, seed)
                    )
                    responses /= (
                        f'length_1/ckpt_final/trained/{spec["variant"]}'
                        if task == "s5"
                        else f'ckpt_final/finetuned/{spec["variant"]}'
                    )
                    responses.mkdir(parents=True)
                    name = samples[task][0]
                    (responses / name).write_text(
                        json.dumps(
                            {
                                "experiment_id": Path(name).stem,
                                "raw_text": "<ANSWER>wrong</ANSWER>",
                                "generated_token_ids": [1],
                            }
                        )
                    )
                    capture.append({"job": job, "stable": True})
                (snapshot / "capture.json").write_text(json.dumps({"cells": capture}))
                stream = io.StringIO()
                with (
                    patch.object(plan, "CONFIG", local),
                    patch.object(
                        sys, "argv", ["plan", "--snapshot", str(snapshot), "--inspect"]
                    ),
                    contextlib.redirect_stdout(stream),
                ):
                    plan.main()
                inspection = json.loads(stream.getvalue())
                self.assertEqual(len(inspection["cells"]), 8)
                self.assertTrue(
                    all(r["validated_final_checkpoint"] for r in inspection["cells"])
                )
                self.assertFalse(common.WORK.exists())
                with (
                    patch.object(plan, "CONFIG", local),
                    patch.object(finalize, "CONFIG", local),
                    patch.object(sys, "argv", ["plan", "--snapshot", str(snapshot)]),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    plan.main()
                    finalize.main()
                self.assertTrue((common.WORK / "published.json").exists())
                self.assertNotIn("JOB ", (common.WORK / "recovery.dag").read_text())
                for task, model, seed in JOBS.values():
                    common.checkpoint_valid(
                        common.canonical_condition(task, model, seed),
                        task,
                        model,
                        seed,
                        config,
                    )
            finally:
                os.chdir(original)

    def test_response_checks(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "a.json"
            file.write_text(
                json.dumps(
                    {
                        "experiment_id": "a",
                        "raw_text": "bad answer is still a valid response",
                        "generated_token_ids": [1, 2],
                    }
                )
            )
            response_valid(file, "a", 2)
            with self.assertRaises(ValueError):
                response_valid(file, "b", 2)
            with self.assertRaises(ValueError):
                response_valid(file, "a", 1)
            file.write_text("{")
            with self.assertRaises(ValueError):
                response_valid(file, "a", 2)

    def test_safetensors_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "adapter.safetensors"
            header = json.dumps(
                {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            ).encode()
            payload = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
            file.write_bytes(payload)
            safetensors_valid(file)
            file.write_bytes(payload[:-1])
            with self.assertRaises(ValueError):
                safetensors_valid(file)

    def test_optimizer_dropout_resume(self):
        import torch
        from chain_of_lies.training.ce.recovery import (
            save_state,
            load_state,
            restore_rng,
        )

        def create():
            torch.manual_seed(17)
            random.seed(17)
            model = torch.nn.Sequential(
                torch.nn.Linear(3, 3), torch.nn.Dropout(0.3), torch.nn.Linear(3, 1)
            )
            return model, torch.optim.AdamW(model.parameters(), lr=0.01)

        def update(model, opt):
            opt.zero_grad()
            loss = model(torch.ones(2, 3)).square().sum()
            loss.backward()
            opt.step()

        with tempfile.TemporaryDirectory() as d:
            model, opt = create()
            for _ in range(3):
                update(model, opt)
            path = Path(d) / "state.pt"
            save_state(
                path,
                model.state_dict(),
                opt,
                "signature",
                3,
                6,
                [{"step": i} for i in range(1, 4)],
            )
            for _ in range(4):
                update(model, opt)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            model, opt = create()
            state = load_state(path, "signature")
            model.load_state_dict(state["weights"])
            opt.load_state_dict(state["optimizer"])
            restore_rng(state)
            for _ in range(4):
                update(model, opt)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, expected[k]), k)
            with self.assertRaises(ValueError):
                load_state(path, "wrong")


if __name__ == "__main__":
    unittest.main()
