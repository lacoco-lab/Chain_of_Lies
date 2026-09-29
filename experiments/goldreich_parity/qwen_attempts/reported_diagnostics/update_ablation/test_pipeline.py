"""Fast tests for the update-ablation pipeline; no model download is required."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_module = load_module("update_ablation_generate", "generate_data.py")
summary_module = load_module("update_ablation_summary", "summarize.py")


class PipelineTests(unittest.TestCase):
    def test_conditions_are_balanced_paired_and_correct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
            config.update(
                {
                    "local_replay_n": 128,
                    "local_validation_n": 128,
                    "supplied_update_train_n": 128,
                    "supplied_update_validation_n": 128,
                    "condition_train_n": 126,
                    "condition_validation_n": 126,
                }
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = generate_module.generate(config_path, root / "data")
            self.assertTrue(manifest["all_seed_partitions_disjoint"])
            self.assertTrue(manifest["paired_condition_seeds"])

            def read(name: str) -> list[dict]:
                return [
                    json.loads(line)
                    for line in (root / "data" / f"{name}.jsonl")
                    .read_text()
                    .splitlines()
                ]

            previous = read("derive_previous_train")
            current = read("derive_current_train")
            self.assertEqual(
                {row["spec"]["prg_seed"] for row in previous},
                {row["spec"]["prg_seed"] for row in current},
            )
            for rows, condition in (
                (previous, "derive_previous"),
                (current, "derive_current"),
            ):
                self.assertEqual(
                    sum(row["supervised_suffix"] == "1" for row in rows), 63
                )
                for row in rows:
                    spec = row["spec"]
                    left, right = spec["input_pair"]
                    answer = (
                        spec["previous_c"]
                        ^ spec["previous_mask"]
                        ^ left
                        ^ right
                        ^ spec["current_mask"]
                    )
                    self.assertEqual(answer, spec["gold_bit"])
                    self.assertEqual(len(spec["attention_groups"]), 1)
                    self.assertEqual(len(spec["attention_groups"][0]), 5)
                    self.assertEqual(spec["task"], condition)

    def test_summary_identifies_two_mask_bottleneck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                (HERE / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            base = {
                "final_evaluations": {
                    "supplied_update": {"greedy_accuracy": 1.0},
                    "local": {"greedy_accuracy": 0.99},
                }
            }
            (root / "base").mkdir()
            (root / "base" / "metrics.json").write_text(
                json.dumps(base), encoding="utf-8"
            )
            for condition in ("derive_previous", "derive_current"):
                other = (
                    "derive_current"
                    if condition == "derive_previous"
                    else "derive_previous"
                )
                metrics = {
                    "final_evaluations": {
                        condition: {"n": 1008, "greedy_accuracy": 0.99, "bit_ce": 0.01},
                        other: {"n": 1008, "greedy_accuracy": 0.5, "bit_ce": 0.69},
                        "local": {"greedy_accuracy": 0.98},
                        "supplied_update": {"greedy_accuracy": 1.0},
                    },
                    "final_attention": {
                        "selected_attention_mass": 0.98,
                        "selected_seed_ratio": 0.99,
                    },
                }
                (root / condition).mkdir()
                (root / condition / "metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            result = summary_module.summarize(config_path, root)
            self.assertTrue(all(result["checks"].values()))
            self.assertIn("two Goldreich masks", result["diagnosis"])


if __name__ == "__main__":
    unittest.main()
