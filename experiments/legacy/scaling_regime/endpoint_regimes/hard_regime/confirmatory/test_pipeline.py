from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.confirmatory.generate_splits import (
    generate,
)  # noqa: E402
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.confirmatory.summarize import (
    summarize,
)  # noqa: E402
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.confirmatory.validate_splits import (
    validate,
)  # noqa: E402


def _small_config(root: Path) -> Path:
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    config["train_n"] = 30
    config["val_n"] = 10
    config["steganography"]["tokenizer_check_sample_size"] = 30
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class SteganographyPipelineTest(unittest.TestCase):
    def test_generation_is_disjoint_within_and_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _small_config(root)
            data_root = root / "data"
            manifest = generate(config_path, data_root)
            report = validate(config_path, data_root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["cross_seed_component_overlap"], 0)
            self.assertTrue(manifest["cross_seed_component_disjointness_required"])
            self.assertEqual(
                manifest["families"]["steganography"]["cross_seed_unique_questions"],
                240,
            )
            for seed in (0, 1, 2):
                seed_report = report["seed_reports"][f"steganography/seed_{seed}"]
                self.assertEqual(seed_report["global_train_val_question_overlap"], 0)

    def test_validator_rejects_commutative_train_eval_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _small_config(root)
            data_root = root / "data"
            generate(config_path, data_root)
            variant_root = (
                data_root / "steganography" / "seed_0" / "arith_steg_local_invisible"
            )
            train = json.loads(
                next((variant_root / "train_prompts").glob("*.json")).read_text()
            )
            val_path = next((variant_root / "val_prompts").glob("*.json"))
            val = json.loads(val_path.read_text())
            left, right = train["spec"]["public_question"].split(" * ")
            val["spec"]["public_question"] = f"{right} * {left}"
            val_path.write_text(json.dumps(val), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reuses|overlap"):
                validate(config_path, data_root)

    def test_config_is_frozen_to_required_design(self) -> None:
        config = json.loads(
            (EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["operand_range"], [500, 1000])
        self.assertEqual(config["seeds"], [0, 1, 2])
        self.assertEqual(set(config["models"]), {"qwen", "llama"})
        self.assertEqual(config["checkpoint"], "ckpt_final")
        self.assertEqual(
            config["steganography"]["modes"], ["answer_only", "local_channel_cot"]
        )
        self.assertNotIn("mismatch", json.dumps(config))

    def test_condor_combines_to_six_model_seed_jobs(self) -> None:
        submit = (EXPERIMENT_DIR / "condor" / "all_cells.sub").read_text(
            encoding="utf-8"
        )
        runner = (EXPERIMENT_DIR / "run.sh").read_text(encoding="utf-8")
        self.assertIn("run.sh cell $(model) $(seed)", submit)
        self.assertIn("when_to_transfer_output = ON_SUCCESS", submit)
        self.assertIn("success_exit_code = 0", submit)
        self.assertIn("bootstrap_python", runner)
        self.assertIn('prepare_model_cache "$MODEL_KEY"', runner)
        self.assertIn("transformers==4.44.2", runner)
        self.assertLess(
            runner.index('prepare_model_cache "$MODEL_KEY"'),
            runner.index('exec python "$EXPERIMENT" cell'),
        )
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

        llama_submit = (EXPERIMENT_DIR / "condor" / "llama_cells.sub").read_text(
            encoding="utf-8"
        )
        self.assertIn("run.sh cell llama $(seed)", llama_submit)
        self.assertIn("queue seed in 0,1,2", llama_submit)
        self.assertNotIn("run.sh cell qwen", llama_submit)

    def test_summary_requires_and_aggregates_all_six_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = EXPERIMENT_DIR / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for model in config["models"]:
                for seed in config["seeds"]:
                    provenance = (
                        root
                        / model
                        / f"seed_{seed}"
                        / "provenance"
                        / "validation_report.json"
                    )
                    provenance.parent.mkdir(parents=True, exist_ok=True)
                    provenance.write_text(json.dumps({"valid": True}), encoding="utf-8")
                    for family in ("steganography",):
                        variants = [config[family]["variant"]]
                        for mode in config[family]["modes"]:
                            for variant in variants:
                                private = (
                                    0.5 + 0.1 * seed if mode != "answer_only" else 0.1
                                )
                                metrics = {
                                    "num_examples": 1000,
                                    "missing_responses": [],
                                    "task_success_rate": private,
                                    "public_exact_rate": 0.9,
                                    "private_exact_rate": private,
                                    "payload_decode_rate": 0.8,
                                    "concealment_rate": 1.0,
                                    "format_rate": 1.0,
                                    "avg_cot_words": 50.0,
                                }
                                path = (
                                    root
                                    / model
                                    / f"seed_{seed}"
                                    / family
                                    / mode
                                    / "per_variant_eval"
                                    / "ckpt_final"
                                    / f"{variant}.json"
                                )
                                path.parent.mkdir(parents=True, exist_ok=True)
                                path.write_text(
                                    json.dumps(
                                        {
                                            "reports": [
                                                {
                                                    "variant_name": variant,
                                                    "baseline": metrics,
                                                    "rl": metrics,
                                                }
                                            ]
                                        }
                                    ),
                                    encoding="utf-8",
                                )
            result = summarize(config_path, root)
            self.assertEqual(len(result["per_seed_metrics"]), 12)
            self.assertEqual(len(result["aggregate_contrasts"]), 2)
            self.assertTrue((root / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
