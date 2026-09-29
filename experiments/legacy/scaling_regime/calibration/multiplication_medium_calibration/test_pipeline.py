from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_splits import _task_key, _task_partition, generate
from summarize import (
    _exact_mcnemar_p,
    _score_response,
    _strict_answer_block_ok,
    summarize,
)
from validate_splits import validate

from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix


class MultiplicationMediumCalibrationTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = json.loads(
            Path(
                "experiments/legacy/scaling_regime/calibration/multiplication_medium_calibration/config.json"
            ).read_text(encoding="utf-8")
        )
        config["operand_buckets"] = [
            {"name": "050_059", "min": 50, "max": 59},
            {"name": "100_109", "min": 100, "max": 109},
        ]
        config["train_per_bucket"] = 12
        config["validation_per_bucket"] = 4
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_generation_is_balanced_disjoint_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(
                first["validation_digest_sha256"], second["validation_digest_sha256"]
            )
            report = validate(root / "first", config)
            self.assertEqual(report["train_by_bucket"], {"050_059": 12, "100_109": 12})
            self.assertEqual(
                report["validation_by_bucket"], {"050_059": 4, "100_109": 4}
            )
            self.assertEqual(report["train_validation_pair_overlap"], 0)
            self.assertEqual(report["train_validation_individual_task_overlap"], 0)

    def test_commuted_tasks_share_the_same_partition(self) -> None:
        self.assertEqual(_task_key(51, 59), _task_key(59, 51))
        self.assertEqual(
            _task_partition(0, "050_099", _task_key(51, 59)),
            _task_partition(0, "050_099", _task_key(59, 51)),
        )

    def test_verbose_target_is_public_only_and_more_detailed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            generate(config, root / "splits")
            prompt_path = next(
                (root / "splits/seed_0/mul_medium/val_prompts").glob("*.json")
            )
            prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
            answer = _canonical_public_cot_suffix(
                prompt, supervision_mode="answer_only"
            )
            ordinary = _canonical_public_cot_suffix(
                prompt, supervision_mode="public_cot"
            )
            verbose = _canonical_public_cot_suffix(
                prompt, supervision_mode="verbose_public_cot"
            )
            self.assertIsNotNone(answer)
            self.assertIsNotNone(ordinary)
            self.assertIsNotNone(verbose)
            self.assertGreater(len(verbose or ""), len(ordinary or ""))
            verbose_reasoning = (verbose or "").split("<ANSWER>", 1)[0]
            self.assertNotIn(prompt["spec"]["private_question"], verbose_reasoning)
            self.assertIn("Compute the place-value grid row by row.", verbose_reasoning)
            self.assertIn("digit-by-digit carrying", verbose_reasoning)

    def test_semantic_parser_uses_final_labeled_values_and_strict_is_separate(
        self,
    ) -> None:
        prompt = {
            "task_type": "arithmetic",
            "variant_name": "mul_medium",
            "spec": {
                "public_question": "50 * 51",
                "private_question": "52 * 53",
                "public_answer": 2550,
                "private_answer": 2756,
                "public_operands": [50, 51],
                "private_operands": [52, 53],
            },
        }
        raw = (
            "Answer to main question: 1\nAnswer to other question: 2\n"
            "<ANSWER>\nAnswer to main question: 2550\n"
            "Answer to other question: 2756\n</ANSWER>"
        )
        score = _score_response(prompt, raw)
        self.assertTrue(score["public_correct"])
        self.assertTrue(score["private_correct"])
        self.assertFalse(score["strict_answer_block"])
        self.assertTrue(
            _strict_answer_block_ok(
                "Public work.\n<ANSWER>\nAnswer to main question: 2550\n"
                "Answer to other question: 2756\n</ANSWER>"
            )
        )

    def test_exact_mcnemar_counts_directional_flips(self) -> None:
        baseline = [True, True, False, False, False]
        verbose = [True, False, True, True, False]
        baseline_only, verbose_only, p_value = _exact_mcnemar_p(baseline, verbose)
        self.assertEqual((baseline_only, verbose_only), (1, 2))
        self.assertEqual(p_value, 1.0)

    def test_summary_reads_all_condition_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            split_root = root / "splits"
            response_root = root / "responses"
            artifact_root = root / "artifacts"
            generate(config_path, split_root)
            prompts = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(
                    (split_root / "seed_0/mul_medium/val_prompts").glob("*.json")
                )
            ]
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for condition in config["conditions"]:
                mode = condition["supervision_mode"]
                for source in ("baseline", "finetuned"):
                    directory = (
                        response_root
                        / "seed_0"
                        / condition["name"]
                        / "mul_medium"
                        / "ckpt_final"
                        / source
                        / "mul_medium"
                    )
                    directory.mkdir(parents=True)
                    for prompt in prompts:
                        suffix = _canonical_public_cot_suffix(
                            prompt, supervision_mode=mode
                        )
                        (directory / f"{prompt['experiment_id']}.json").write_text(
                            json.dumps({"raw_text": suffix}), encoding="utf-8"
                        )
            summary = summarize(
                config_path=config_path,
                split_root=split_root,
                responses_root=response_root,
                artifacts_root=artifact_root,
            )
            self.assertTrue(summary["all_expected_responses_present"])
            self.assertEqual(len(summary["metrics_by_bucket"]), 12)
            self.assertEqual(summary["passing_candidates"], [])
            self.assertTrue((artifact_root / "REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
