from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXPERIMENT_DIR.parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.hard_regime.multiplication_only_piggyback.diagnostics.qwen_seed2_optimization.experiment import (  # noqa: E402
    load_config,
)
from experiments.hard_regime.multiplication_only_piggyback.diagnostics.qwen_seed2_optimization.summarize import (  # noqa: E402
    summarize,
)


class OptimizationDiagnosticTest(unittest.TestCase):
    def test_config_is_pinned_and_narrow(self) -> None:
        diagnostic, parent = load_config()
        observed = hashlib.sha256((PARENT_DIR / "config.json").read_bytes()).hexdigest()
        self.assertEqual(diagnostic["parent_config_sha256"], observed)
        self.assertEqual(diagnostic["data_seed"], 2)
        self.assertEqual(diagnostic["additional_optimization_seeds"], [3, 4, 5])
        self.assertEqual(diagnostic["model_key"], "qwen")
        self.assertEqual(diagnostic["variant"], "arith_piggyback")
        self.assertEqual(diagnostic["mode"], "public_cot")
        self.assertEqual(parent["train_n"], 10000)
        self.assertEqual(parent["piggyback"]["epochs"], 1)

    def test_condor_queues_three_parallel_cells_and_one_adapter_each(self) -> None:
        submit = (EXPERIMENT_DIR / "condor" / "all_seeds.sub").read_text(encoding="utf-8")
        runner = (EXPERIMENT_DIR / "experiment.py").read_text(encoding="utf-8")
        self.assertIn("queue optimization_seed in 3,4,5", submit)
        self.assertIn("seed_2", submit)
        self.assertIn("baseline/arith_piggyback", submit)
        self.assertIn('"--variant", str(diagnostic["variant"])', runner)
        self.assertNotIn("arith_piggyback_control", runner)
        self.assertNotIn("answer_only", runner)

    def test_summary_combines_original_and_all_additional_seeds(self) -> None:
        config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic_root = root / "diagnostic"
            original_root = root / "original"

            def write_report(path: Path, private: float) -> None:
                metrics = {
                    "num_examples": 1000,
                    "missing_responses": [],
                    "task_success_rate": private,
                    "public_exact_rate": 0.95,
                    "private_exact_rate": private,
                    "concealment_rate": 1.0,
                    "format_rate": 1.0,
                    "avg_cot_words": 42.0,
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"reports": [{"rl": metrics}]}), encoding="utf-8")

            write_report(
                original_root / "qwen/seed_2/public_cot/per_variant_eval/ckpt_final/arith_piggyback.json",
                0.35,
            )
            for seed, private in zip((3, 4, 5), (0.40, 0.50, 0.60)):
                provenance = diagnostic_root / f"opt_seed_{seed}" / "provenance"
                provenance.mkdir(parents=True, exist_ok=True)
                (provenance / "validation_report.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
                (provenance / "diagnostic_cell.json").write_text(
                    json.dumps({"data_seed": 2, "optimization_seed": seed}), encoding="utf-8"
                )
                write_report(
                    diagnostic_root / f"opt_seed_{seed}/public_cot/per_variant_eval/ckpt_final/arith_piggyback.json",
                    private,
                )
            result = summarize(EXPERIMENT_DIR / "config.json", diagnostic_root, original_root)
            self.assertEqual(len(result["per_optimization_seed"]), 4)
            self.assertEqual(result["additional_private_exact_values"], [0.4, 0.5, 0.6])
            self.assertAlmostEqual(result["aggregates"][0]["private_exact_rate_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
