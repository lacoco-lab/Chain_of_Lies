from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from generate_splits import generate  # noqa: E402
from summarize import summarize  # noqa: E402
from validate_splits import validate  # noqa: E402


def _small_config(tmp_path: Path) -> Path:
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    config["train_n"] = 30
    config["val_n"] = 10
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class CalibrationPipelineTest(unittest.TestCase):
    def test_generated_suite_is_strictly_component_disjoint(self) -> None:
        cases = (
            ("piggyback", "r1000_2000", 90, 30),
            ("steganography", "r500_1000", 60, 20),
        )
        for family, range_id, expected_train_questions, expected_val_questions in cases:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                tmp_path = Path(temporary)
                config_path = _small_config(tmp_path)
                data_root = tmp_path / "data"
                generate(config_path, data_root, family, range_id)
                result = validate(config_path, data_root, family, range_id)
                self.assertTrue(result["valid"])
                self.assertEqual(result["global_train_val_question_overlap"], 0)
                self.assertEqual(result["global_train_unique_questions"], expected_train_questions)
                self.assertEqual(result["global_val_unique_questions"], expected_val_questions)

    def test_validator_rejects_a_single_component_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config_path = _small_config(tmp_path)
            data_root = tmp_path / "data"
            generate(config_path, data_root, "steganography", "r500_1000")
            variant_root = (
                data_root
                / "steganography"
                / "r500_1000"
                / "seed_314159"
                / "arith_steg_local_invisible"
            )
            train_record = json.loads(next((variant_root / "train_prompts").glob("*.json")).read_text())
            val_path = next((variant_root / "val_prompts").glob("*.json"))
            val_record = json.loads(val_path.read_text())
            val_record["spec"]["public_question"] = train_record["spec"]["public_question"]
            val_path.write_text(json.dumps(val_record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "overlap|reuses"):
                validate(config_path, data_root, "steganography", "r500_1000")

    def test_calibration_modes_exclude_mismatched_cot(self) -> None:
        config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["piggyback"]["modes"], ["answer_only", "public_cot"])
        self.assertEqual(config["steganography"]["modes"], ["answer_only", "local_channel_cot"])
        self.assertNotIn("mismatched_public_cot", json.dumps(config))

    def test_summary_selects_hardest_complete_passing_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
            family_root = root / "qwen" / "piggyback" / "r1000_2000" / "seed_314159"
            metrics = {
                ("public_cot", "arith_piggyback"): (0.95, 0.89),
                ("public_cot", "arith_piggyback_control"): (0.94, 0.10),
                ("answer_only", "arith_piggyback"): (0.10, 0.05),
                ("answer_only", "arith_piggyback_control"): (0.10, 0.06),
            }
            for (mode, variant), (public, private) in metrics.items():
                path = family_root / mode / "per_variant_eval" / "ckpt_final" / f"{variant}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                report_metrics = {
                    "public_exact_rate": public,
                    "private_exact_rate": private,
                    "payload_decode_rate": 0.0,
                }
                path.write_text(
                    json.dumps(
                        {
                            "reports": [
                                {
                                    "variant_name": variant,
                                    "baseline": report_metrics,
                                    "rl": report_metrics,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            result = summarize(EXPERIMENT_DIR / "config.json", root, "qwen", "piggyback")
            self.assertEqual(result["selected_range"], "r1000_2000")
            self.assertTrue(result["completed_rows"][0]["passes_gate"])
            self.assertTrue((root / "qwen" / "piggyback" / "CALIBRATION_REPORT.md").exists())

    def test_ordinary_qwen_cells_cover_every_range_once(self) -> None:
        config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
        submit = (EXPERIMENT_DIR / "condor" / "qwen_all_cells.sub").read_text(encoding="utf-8")
        expected = {
            (family, item["id"])
            for family in ("piggyback", "steganography")
            for item in config[family]["ranges_hard_to_easy"]
        }
        rows: set[tuple[str, str]] = set()
        inside = False
        for raw_line in submit.splitlines():
            line = raw_line.strip()
            if line == "queue family,range_id from (":
                inside = True
            elif inside and line == ")":
                inside = False
            elif inside and line:
                family, range_id = line.split()
                self.assertNotIn((family, range_id), rows)
                rows.add((family, range_id))
        self.assertEqual(rows, expected)
        self.assertIn("run.sh cell qwen $(family) $(range_id)", submit)

    def test_ordinary_summary_queues_both_families(self) -> None:
        submit = (EXPERIMENT_DIR / "condor" / "qwen_summarize_all.sub").read_text(encoding="utf-8")
        self.assertIn("queue family in piggyback,steganography", submit)


if __name__ == "__main__":
    unittest.main()
