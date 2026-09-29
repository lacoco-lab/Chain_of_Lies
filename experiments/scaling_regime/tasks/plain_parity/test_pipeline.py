#!/usr/bin/env python3
"""Fast structural tests for the plain-parity calibration pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import torch

from chain_of_lies.evaluation.rewards import score_completion
from chain_of_lies.training.ce.trainer import _chunked_checkpointed_ce_from_hidden
from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)
from chain_of_lies.variants.parity.data_generation.generate import (
    parity,
    running_parities,
)
from chain_of_lies.variants.steganography.hard_task_channels import decode_payload
from experiments.scaling_regime.tasks.plain_parity.generate_splits import generate
from experiments.scaling_regime.tasks.plain_parity import (
    experiment as parity_experiment,
)
from experiments.scaling_regime.tasks.plain_parity.summarize import summarize
from experiments.scaling_regime.tasks.plain_parity.summarize_three_seeds import (
    _scientific_signature,
    summarize_three_seeds,
)
from experiments.scaling_regime.tasks.plain_parity.validate_splits import (
    _parity_target,
    validate,
)


class ParityCalibrationTests(unittest.TestCase):
    def _config(self, root: Path, *, seed: int = 0) -> Path:
        config = json.loads(
            Path(
                "experiments/scaling_regime/tasks/plain_parity/config.json"
            ).read_text()
        )
        config["seed"] = seed
        config["length_buckets"] = [
            {"name": "008_009", "min": 8, "max": 9},
            {"name": "010_011", "min": 10, "max": 11},
        ]
        config["train_per_bucket"] = 12
        config["validation_per_bucket"] = 4
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_replication_configs_preserve_scientific_settings(self) -> None:
        configs = [
            json.loads(Path(path).read_text())
            for path in (
                "experiments/scaling_regime/tasks/plain_parity/config.json",
                "experiments/scaling_regime/tasks/plain_parity/config_seed_1.json",
                "experiments/scaling_regime/tasks/plain_parity/config_seed_2.json",
            )
        ]
        self.assertEqual([config["seed"] for config in configs], [0, 1, 2])
        self.assertEqual(
            _scientific_signature(configs[0]), _scientific_signature(configs[1])
        )
        self.assertEqual(
            _scientific_signature(configs[0]), _scientific_signature(configs[2])
        )
        for config in configs[1:]:
            self.assertFalse(config["training"]["activation_cpu_offload"])
            self.assertFalse(config["training"]["run_training_validation"])
            self.assertEqual(config["training"]["evaluation_batch_size"], 8)

    def test_fast_replication_cell_uses_equivalent_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root, seed=1)
            config = json.loads(config_path.read_text())
            config["training"].update(
                {
                    "activation_cpu_offload": False,
                    "run_training_validation": False,
                    "evaluation_batch_size": 8,
                }
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            data_root = root / "data"
            generate(config_path, data_root)
            commands: list[list[str]] = []
            with (
                patch.object(parity_experiment, "DATA_ROOT", data_root),
                patch.object(parity_experiment, "RESPONSES_ROOT", root / "responses"),
                patch.object(parity_experiment, "ARTIFACTS_ROOT", root / "artifacts"),
                patch.object(
                    parity_experiment,
                    "_run",
                    side_effect=lambda args: commands.append(args),
                ),
            ):
                parity_experiment.run_cell(config_path, "qwen", "public_only_cot")

            self.assertEqual(len(commands), 2)
            train_args, eval_args = commands
            self.assertIn("--memory-efficient-ce", train_args)
            self.assertNotIn("--activation-cpu-offload", train_args)
            self.assertNotIn("--val-prompts-dir", train_args)
            self.assertNotIn("--expected-val-prompts", train_args)
            self.assertEqual(
                eval_args[eval_args.index("--inference-batch-size") + 1], "8"
            )
            self.assertNotIn("--no-resume", eval_args)

    def test_generation_is_reproducible_matched_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(
                first["validation_digest_sha256"], second["validation_digest_sha256"]
            )
            report = validate(config, root / "first")
            self.assertTrue(report["valid"])
            self.assertEqual(report["individual_sequence_train_val_overlap"], 0)

            control_path = next(
                (root / "first/seed_0/parity_control/train_prompts").glob("*.json")
            )
            piggy_path = (
                root / "first/seed_0/parity_piggyback/train_prompts" / control_path.name
            )
            control = json.loads(control_path.read_text())
            piggy = json.loads(piggy_path.read_text())
            self.assertEqual(
                control["spec"]["private_bits"], piggy["spec"]["private_bits"]
            )
            self.assertEqual(
                piggy["spec"]["public_bits"][:-1], piggy["spec"]["private_bits"][:-1]
            )
            self.assertNotEqual(
                piggy["spec"]["public_answer"], piggy["spec"]["private_answer"]
            )

    def test_targets_round_trip_and_score_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            generate(config, root / "data")
            path = next(
                (root / "data/seed_0/parity_steg_local_invisible/train_prompts").glob(
                    "*.json"
                )
            )
            record = json.loads(path.read_text())
            spec = record["spec"]
            self.assertEqual(parity(spec["public_bits"]), spec["public_answer"])
            self.assertEqual(
                running_parities(spec["private_bits"])[-1], spec["private_answer"]
            )
            target = _canonical_public_cot_suffix(
                record, supervision_mode="local_channel_cot"
            )
            self.assertIsNotNone(target)
            self.assertEqual(_parity_target(record, "local_channel_cot"), target)
            reasoning = (target or "").split("<ANSWER>", 1)[0]
            self.assertEqual(
                decode_payload(record, reasoning), spec["steganography_payload"]
            )
            score = score_completion(record, target or "")
            self.assertTrue(score.task_success)
            self.assertTrue(score.flags["payload_decode_correct"])

            control = json.loads(json.dumps(record))
            control["variant_name"] = "parity_control"
            control["spec"].pop("steganography_scheme")
            control["spec"].pop("steganography_payload")
            control["spec"]["filler_token_count"] = 64
            filler = _canonical_public_cot_suffix(
                control, supervision_mode="filler_public_cot"
            )
            self.assertEqual((filler or "").count(FILLER_TOKEN_MARKER), 1)
            self.assertEqual(_parity_target(control, "filler_public_cot"), filler)
            for mode in ("answer_only", "public_cot"):
                self.assertEqual(
                    _parity_target(control, mode),
                    _canonical_public_cot_suffix(control, supervision_mode=mode),
                )

    def test_memory_efficient_ce_matches_standard_loss_and_gradients(self) -> None:
        torch.manual_seed(7)
        hidden_standard = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
        hidden_chunked = hidden_standard.detach().clone().requires_grad_(True)
        head_standard = torch.nn.Linear(5, 11, bias=False, dtype=torch.float64)
        head_chunked = torch.nn.Linear(5, 11, bias=False, dtype=torch.float64)
        head_chunked.load_state_dict(head_standard.state_dict())
        labels = torch.tensor(
            [
                [-100, -100, 3, 1, 8, 4, 2],
                [-100, 6, 5, 0, 9, 7, 10],
            ]
        )

        logits = head_standard(hidden_standard)
        standard_loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        chunked_loss = _chunked_checkpointed_ce_from_hidden(
            head_chunked,
            hidden_chunked,
            labels,
            token_chunk_size=3,
        )
        standard_loss.backward()
        chunked_loss.backward()

        torch.testing.assert_close(chunked_loss, standard_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            hidden_chunked.grad, hidden_standard.grad, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            head_chunked.weight.grad,
            head_standard.weight.grad,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_fail_closed_summary_groups_buckets_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            config = json.loads(config_path.read_text())
            split_root = root / "data"
            responses_root = root / "responses"
            output_root = root / "artifacts"
            generate(config_path, split_root)
            for model in config["models"]:
                for condition, condition_spec in config["conditions"].items():
                    variant = condition_spec["variant"]
                    prompts = (split_root / f"seed_0/{variant}/val_prompts").glob(
                        "*.json"
                    )
                    for source in ("baseline", "finetuned"):
                        response_dir = (
                            responses_root
                            / model
                            / f"seed_0/{condition}/ckpt_final/{source}/{variant}"
                        )
                        response_dir.mkdir(parents=True, exist_ok=True)
                        for prompt_path in prompts:
                            record = json.loads(prompt_path.read_text())
                            mode = (
                                "local_channel_cot"
                                if condition == "steganography"
                                else "answer_only"
                            )
                            raw_text = _canonical_public_cot_suffix(
                                record, supervision_mode=mode
                            )
                            (response_dir / prompt_path.name).write_text(
                                json.dumps(
                                    {"raw_text": raw_text, "generated_token_ids": []}
                                ),
                                encoding="utf-8",
                            )
                        prompts = (split_root / f"seed_0/{variant}/val_prompts").glob(
                            "*.json"
                        )
            result = summarize(config_path, split_root, responses_root, output_root)
            self.assertEqual(len(result["metrics_by_length_bucket"]), 40)
            self.assertEqual(len(result["provisional_regime_candidates"]), 4)
            self.assertTrue(
                all(
                    item["provisional_classification"] == "easy_candidate"
                    for item in result["provisional_regime_candidates"]
                )
            )

    def test_three_seed_summary_requires_and_aggregates_all_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_root = root / "data"
            responses_root = root / "responses"
            config_paths = []
            for seed in (0, 1, 2):
                config_dir = root / f"config_{seed}"
                config_dir.mkdir()
                config_path = self._config(config_dir, seed=seed)
                config_paths.append(config_path)
                config = json.loads(config_path.read_text())
                generate(config_path, split_root)
                for model in config["models"]:
                    for condition, condition_spec in config["conditions"].items():
                        variant = condition_spec["variant"]
                        prompt_paths = sorted(
                            (split_root / f"seed_{seed}/{variant}/val_prompts").glob(
                                "*.json"
                            )
                        )
                        for source in ("baseline", "finetuned"):
                            response_dir = (
                                responses_root
                                / model
                                / f"seed_{seed}"
                                / condition
                                / "ckpt_final"
                                / source
                                / variant
                            )
                            response_dir.mkdir(parents=True, exist_ok=True)
                            for prompt_path in prompt_paths:
                                record = json.loads(prompt_path.read_text())
                                mode = (
                                    "local_channel_cot"
                                    if condition == "steganography"
                                    else "answer_only"
                                )
                                raw_text = _canonical_public_cot_suffix(
                                    record, supervision_mode=mode
                                )
                                (response_dir / prompt_path.name).write_text(
                                    json.dumps(
                                        {
                                            "raw_text": raw_text,
                                            "generated_token_ids": [],
                                        }
                                    ),
                                    encoding="utf-8",
                                )

            result = summarize_three_seeds(
                split_root=split_root,
                responses_root=responses_root,
                artifacts_root=root / "artifacts",
                config_paths=tuple(config_paths),
            )
            self.assertEqual(len(result["aggregate_metrics"]), 40)
            self.assertEqual(len(result["replicated_regime_candidates"]), 4)
            self.assertTrue(
                all(
                    item["replicated_classification"] == "replicated_easy_candidate"
                    for item in result["replicated_regime_candidates"]
                )
            )
            self.assertTrue((root / "artifacts/REPORT_THREE_SEED.md").exists())


if __name__ == "__main__":
    unittest.main()
