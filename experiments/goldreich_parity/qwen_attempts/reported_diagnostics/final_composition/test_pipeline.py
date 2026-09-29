"""Fast final-pipeline tests requiring no model download."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_module("final_goldreich_generator", "generate_data.py")
trainer = load_module("final_goldreich_trainer", "training_utils.py")
train_module = load_module("final_goldreich_train", "train.py")
summary_module = load_module("final_goldreich_summary", "summarize.py")


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.assertions = (tokenize, add_generation_prompt)
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(character) + 1 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class PipelineTests(unittest.TestCase):
    def test_seed_zero_data_is_locked_to_successful_rank32_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generator.generate(
                HERE / "config.json", Path(temporary) / "data"
            )
            calibration = manifest["rank32_calibration_recipe"]
            self.assertTrue(calibration["enforced"])
            self.assertTrue(calibration["digests_verified"])
            self.assertEqual(
                calibration["digests"], generator.RANK32_CALIBRATION_DIGESTS
            )
            train_module.verify_rank32_calibration(Path(temporary) / "data")
            self.assertEqual(
                train_module.RANK32_SELECTION_WEIGHTS,
                {"local": 0.25, "supplied_update": 0.10},
            )

    def test_balanced_disjoint_curriculum_and_full_attention_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
            config.update(
                {
                    "enforce_rank32_calibration": False,
                    "local_replay_n": 128,
                    "local_validation_n": 128,
                    "supplied_update_train_n": 128,
                    "supplied_update_validation_n": 128,
                    "composition_train_n": 126,
                    "composition_validation_n": 126,
                }
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = generator.generate(config_path, root / "data")
            self.assertTrue(manifest["all_seed_partitions_disjoint"])
            self.assertTrue(manifest["paired_task_seeds"])
            rows = trainer.read_jsonl(root / "data" / "full_update_validation.jsonl")
            self.assertEqual(sum(row["supervised_suffix"] == "1" for row in rows), 63)
            prepared = trainer.prepare_row(CharacterTokenizer(), rows[0])
            self.assertEqual(len(prepared["attention_groups"]), 2)
            self.assertEqual(len(prepared["seed_positions"]), 16)
            delta = trainer.read_jsonl(root / "data" / "mask_delta_validation.jsonl")
            zero = trainer.read_jsonl(root / "data" / "full_zero_validation.jsonl")
            self.assertEqual(
                sorted((row["spec"]["prg_seed"], row["spec"]["step"]) for row in delta),
                sorted((row["spec"]["prg_seed"], row["spec"]["step"]) for row in zero),
            )
            for row in delta:
                spec = row["spec"]
                self.assertEqual(
                    spec["gold_bit"], spec["previous_mask"] ^ spec["current_mask"]
                )

    def test_summary_requires_every_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                (HERE / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            position_final = {
                task: {
                    "n": 100,
                    "greedy_accuracy": 0.99,
                    "binary_accuracy": 0.99,
                    "bit_ce": 0.01,
                }
                for task in (
                    "derive_previous",
                    "derive_current",
                    "local",
                    "supplied_update",
                )
            }
            full_final = dict(position_final)
            for task in ("mask_delta", "full_zero", "full_one", "full_two"):
                full_final[task] = {
                    "n": 100,
                    "greedy_accuracy": 0.98,
                    "binary_accuracy": 0.98,
                    "bit_ce": 0.02,
                }
            full_final["full_update"] = {
                "n": 100,
                "greedy_accuracy": 0.98,
                "binary_accuracy": 0.98,
                "bit_ce": 0.02,
            }
            for name, final in (("positions", position_final), ("full", full_final)):
                directory = root / name
                directory.mkdir()
                metrics = {
                    "final_evaluations": final,
                    "passes_gate": True,
                    "elapsed_seconds": 1.0,
                    "final_attention": {"selected_attention_mass": 0.99},
                }
                (directory / "metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            result = summary_module.summarize(config_path, root)
            self.assertTrue(result["passes_all_gates"])
            self.assertEqual(len(result["metrics"]), 6)
            self.assertEqual(len(result["curriculum_diagnostics"]), 3)


if __name__ == "__main__":
    unittest.main()
