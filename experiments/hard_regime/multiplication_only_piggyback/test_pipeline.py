from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.variants.arithmetic.data_generation.generate import (  # noqa: E402
    build_arithmetic_prompt_text,
)
from experiments.hard_regime.multiplication_only_piggyback.generate_splits import (  # noqa: E402
    _prompt_text,
    generate,
)
from experiments.hard_regime.multiplication_only_piggyback.summarize import summarize  # noqa: E402
from experiments.hard_regime.multiplication_only_piggyback.validate_splits import validate  # noqa: E402


def _small_config(root: Path) -> Path:
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    config["train_n"] = 30
    config["val_n"] = 10
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class MultiplicationOnlyPiggybackTest(unittest.TestCase):
    def test_prompt_changes_only_offset_phrases(self) -> None:
        public, private = "731 * 846", "731 * 851"
        historical = build_arithmetic_prompt_text(public, private)
        expected = historical.replace(" before adding the offset", "").replace(
            ", and then adding the offset", ""
        )
        self.assertEqual(_prompt_text(public, private), expected)
        self.assertNotIn("offset", expected.lower())

    def test_generation_is_multiplication_only_and_globally_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _small_config(root)
            data_root = root / "data"
            manifest = generate(config_path, data_root)
            report = validate(config_path, data_root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["cross_seed_component_overlap"], 0)
            self.assertEqual(manifest["cross_seed_unique_questions"], 360)
            for seed in (0, 1, 2):
                seed_report = report["seed_reports"][f"seed_{seed}"]
                self.assertEqual(seed_report["global_train_unique_questions"], 90)
                self.assertEqual(seed_report["global_val_unique_questions"], 30)
                for path in (data_root / f"seed_{seed}").glob("**/*_prompts/*.json"):
                    record = json.loads(path.read_text(encoding="utf-8"))
                    self.assertNotIn(" + ", record["spec"]["public_question"])
                    self.assertNotIn(" + ", record["spec"]["private_question"])
                    self.assertNotIn("offset", record["prompt_text"].lower())

    def test_validator_rejects_commutative_train_eval_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _small_config(root)
            data_root = root / "data"
            generate(config_path, data_root)
            variant_root = data_root / "seed_0" / "arith_piggyback"
            train = json.loads(next((variant_root / "train_prompts").glob("*.json")).read_text())
            val_path = next((variant_root / "val_prompts").glob("*.json"))
            val = json.loads(val_path.read_text())
            left, right = train["spec"]["public_question"].split(" * ")
            old_public = val["spec"]["public_question"]
            val["spec"]["public_question"] = f"{right} * {left}"
            val["spec"]["public_answer"] = int(left) * int(right)
            val["prompt_text"] = val["prompt_text"].replace(
                f"Main question: {old_public}",
                f"Main question: {right} * {left}",
            )
            val_path.write_text(json.dumps(val), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "overlap|reuses|Digest"):
                validate(config_path, data_root)

    def test_config_is_frozen_to_final_training_recipe(self) -> None:
        config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["operand_range"], [500, 1000])
        self.assertEqual(config["seeds"], [0, 1, 2])
        self.assertEqual(set(config["models"]), {"qwen", "llama"})
        self.assertEqual(config["train_n"], 10000)
        self.assertEqual(config["val_n"], 1000)
        self.assertEqual(config["checkpoint"], "ckpt_final")
        self.assertEqual(config["piggyback"]["epochs"], 1)
        self.assertEqual(config["piggyback"]["batch_size"], 2)
        self.assertEqual(config["piggyback"]["learning_rate"], 0.00002)
        self.assertEqual(config["piggyback"]["lora_r"], 8)
        self.assertEqual(config["piggyback"]["lora_alpha"], 16)
        self.assertEqual(config["piggyback"]["lora_dropout"], 0.05)
        self.assertEqual(config["piggyback"]["question_format"], "multiplication_only")
        self.assertNotIn("offset_range", config["piggyback"])
        self.assertNotIn("mismatch", json.dumps(config))

    def test_condor_combines_exactly_six_cells(self) -> None:
        submit = (EXPERIMENT_DIR / "condor" / "all_cells.sub").read_text(encoding="utf-8")
        self.assertIn("run.sh cell $(model) $(seed)", submit)
        self.assertIn("when_to_transfer_output = ON_SUCCESS", submit)
        rows: list[tuple[str, str]] = []
        inside = False
        for raw in submit.splitlines():
            line = raw.strip()
            if line == "queue model,seed from (":
                inside = True
            elif inside and line == ")":
                inside = False
            elif inside and line:
                rows.append(tuple(line.split()))  # type: ignore[arg-type]
        self.assertEqual(
            rows,
            [(model, seed) for model in ("qwen", "llama") for seed in ("0", "1", "2")],
        )

    def test_summary_requires_all_cells_and_compares_affine_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = EXPERIMENT_DIR / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for model in config["models"]:
                for seed in config["seeds"]:
                    provenance = root / model / f"seed_{seed}" / "provenance" / "validation_report.json"
                    provenance.parent.mkdir(parents=True, exist_ok=True)
                    provenance.write_text(json.dumps({"valid": True}), encoding="utf-8")
                    for mode in config["piggyback"]["modes"]:
                        for variant in config["piggyback"]["variants"]:
                            private = 0.50 + 0.01 * seed if mode == "public_cot" and variant == "arith_piggyback" else 0.10
                            metrics = {
                                "num_examples": 1000,
                                "missing_responses": [],
                                "task_success_rate": private,
                                "public_exact_rate": 0.90,
                                "private_exact_rate": private,
                                "concealment_rate": 1.0,
                                "format_rate": 1.0,
                                "avg_cot_words": 45.0,
                            }
                            path = root / model / f"seed_{seed}" / mode / "per_variant_eval" / "ckpt_final" / f"{variant}.json"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps({"reports": [{"variant_name": variant, "rl": metrics}]}), encoding="utf-8")
            result = summarize(config_path, root)
            self.assertEqual(len(result["per_seed_metrics"]), 24)
            self.assertEqual(len(result["aggregate_contrasts"]), 2)
            self.assertIn("private_minus_affine_reference", result["aggregate_contrasts"][0])


if __name__ == "__main__":
    unittest.main()
