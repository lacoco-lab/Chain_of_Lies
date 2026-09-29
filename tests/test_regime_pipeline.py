from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from chain_of_lies.evaluation.rewards import default_reward_config, score_completion
from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix
from experiments.legacy.scaling_regime.early_regimes.pre_unified_easy_regime.generate_regime_splits import (
    generate_suite,
)


class RegimeGenerationTests(unittest.TestCase):
    def _small_config(self, directory: Path) -> Path:
        source = json.loads(
            Path(
                "experiments/legacy/scaling_regime/early_regimes/pre_unified_easy_regime/easy_regime.json"
            ).read_text(encoding="utf-8")
        )
        for variant in source["variants"]:
            variant["train_n"] = 20
            variant["val_n"] = 5
        target = directory / "easy.json"
        target.write_text(json.dumps(source), encoding="utf-8")
        return target

    def test_eval_only_reproduces_identical_validation_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._small_config(root)
            full_root = root / "full"
            eval_root = root / "eval"
            generate_suite(config_path=config, output_root=full_root, seed=7)
            generate_suite(
                config_path=config, output_root=eval_root, seed=7, eval_only=True
            )
            for variant in ("mul_easy", "s5_easy", "knowledge_easy_1fact"):
                full_manifest = json.loads(
                    (full_root / "seed_7" / variant / "split_manifest.json").read_text()
                )
                eval_manifest = json.loads(
                    (eval_root / "seed_7" / variant / "split_manifest.json").read_text()
                )
                self.assertEqual(
                    full_manifest["validation_digest_sha256"],
                    eval_manifest["validation_digest_sha256"],
                )
                self.assertEqual(
                    full_manifest["component_validation_digest_sha256"],
                    eval_manifest["component_validation_digest_sha256"],
                )
                full_prompts = sorted(
                    (full_root / "seed_7" / variant / "val_prompts").glob("*.json")
                )
                eval_prompts = sorted(
                    (eval_root / "seed_7" / variant / "val_prompts").glob("*.json")
                )
                self.assertEqual(
                    [path.read_bytes() for path in full_prompts],
                    [path.read_bytes() for path in eval_prompts],
                )
            component_prompts = list(
                (eval_root / "seed_7/knowledge_easy_1fact/component_val_prompts").glob(
                    "*.json"
                )
            )
            self.assertEqual(len(component_prompts), 20)

    def test_gold_ordinary_cot_scores_all_task_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._small_config(root)
            generate_suite(config_path=config, output_root=root / "splits", seed=3)
            for variant in ("mul_easy", "s5_easy", "knowledge_easy_1fact"):
                prompt_path = next(
                    (root / "splits" / "seed_3" / variant / "val_prompts").glob(
                        "*.json"
                    )
                )
                prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
                suffix = _canonical_public_cot_suffix(
                    prompt, supervision_mode="public_cot"
                )
                self.assertIsNotNone(suffix)
                score = score_completion(prompt, suffix or "", default_reward_config())
                self.assertTrue(score.task_success)
                self.assertTrue(score.format_ok)
                self.assertNotIn(
                    prompt["spec"].get("private_question", "__missing__"), suffix
                )

            component_path = next(
                (
                    root / "splits/seed_3/knowledge_easy_1fact/component_val_prompts"
                ).glob("*.json")
            )
            component = json.loads(component_path.read_text(encoding="utf-8"))
            suffix = _canonical_public_cot_suffix(
                component, supervision_mode="public_cot"
            )
            self.assertTrue(score_completion(component, suffix or "").task_success)

    def test_multiplication_has_no_offset_and_s5_has_one_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._small_config(root)
            generate_suite(config_path=config, output_root=root / "splits", seed=1)
            mul = json.loads(
                next(
                    (root / "splits/seed_1/mul_easy/val_prompts").glob("*.json")
                ).read_text()
            )
            self.assertNotIn("+", mul["spec"]["public_question"])
            self.assertEqual(mul["spec"]["operand_range"], [2, 50])
            s5 = json.loads(
                next(
                    (root / "splits/seed_1/s5_easy/val_prompts").glob("*.json")
                ).read_text()
            )
            self.assertEqual(len(s5["spec"]["public_instruction_sequence"]), 1)
            self.assertEqual(len(s5["spec"]["private_instruction_sequence"]), 1)


if __name__ == "__main__":
    unittest.main()
