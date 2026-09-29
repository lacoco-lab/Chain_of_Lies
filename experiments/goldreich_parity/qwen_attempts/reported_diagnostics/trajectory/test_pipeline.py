"""Fast trajectory-pipeline tests requiring no model download."""

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


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load("trajectory_generator", "generate_data.py")
validator = load("trajectory_validator", "validate_data.py")
utils = load("trajectory_utils", "training_utils.py")
summary = load("trajectory_summary", "summarize.py")


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(character) + 1 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def source_row(seed: str, split: str, index: int) -> dict:
    return {
        "experiment_id": f"source_{split}_{index}",
        "spec": {"prg_seed": seed},
    }


class TrajectoryTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        config.update(
            {
                "lengths": [2, 4],
                "train_n_per_length": 8,
                "validation_n_per_length": 4,
                "transition_train_n": 8,
                "transition_validation_n": 4,
            }
        )
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        source = root / "source"
        source.mkdir()
        for split, start, n in (("train", 0, 16), ("validation", 32, 8)):
            rows = [
                source_row(format(start + index, "016b"), split, index)
                for index in range(n)
            ]
            generator.write_jsonl(source / f"derive_previous_{split}.jsonl", rows)
        data = root / "data"
        generator.generate(config_path, source, data)
        return config_path, source, data

    def test_balanced_disjoint_correct_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, data = self.make_fixture(root)
            report = validator.validate(config, source, data)
            self.assertEqual(report["trajectory_n4_validation"]["n"], 4)
            self.assertEqual(report["transition_validation"]["n"], 4)
            rows = generator.read_jsonl(data / "trajectory_n4_validation.jsonl")
            self.assertEqual({row["spec"]["final_parity"] for row in rows}, {0, 1})
            self.assertTrue(
                all(len(row["spec"]["encrypted_states"]) == 2 for row in rows)
            )

    def test_variable_bit_weights_cover_trace_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, data = self.make_fixture(root)
            row = generator.read_jsonl(data / "trajectory_n4_validation.jsonl")[0]
            prepared = utils.prepare_row(CharacterTokenizer(), row, 0.1, 1.0)
            self.assertEqual(
                sum(weight == 1.0 for weight in prepared["loss_weights"]), 3
            )
            self.assertEqual(
                prepared["encrypted_states"], row["spec"]["encrypted_states"]
            )

    def test_position_general_transition_is_single_token_and_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, data = self.make_fixture(root)
            rows = generator.read_jsonl(data / "transition_train.jsonl")
            self.assertEqual({row["supervised_suffix"] for row in rows}, {"0", "1"})
            prepared = utils.prepare_row(CharacterTokenizer(), rows[0], 0.1, 1.0)
            self.assertEqual(prepared["target_token_count"], 1)
            self.assertEqual(
                sum(weight == 1.0 for weight in prepared["loss_weights"]), 1
            )

    def test_all_thirty_two_bit_transition_positions_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
            config.update(
                {
                    "train_n_per_length": 30,
                    "validation_n_per_length": 30,
                    "transition_train_n": 60,
                    "transition_validation_n": 30,
                }
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            source = root / "source"
            source.mkdir()
            for split, start in (("train", 0), ("validation", 128)):
                rows = [
                    source_row(format(start + index, "016b"), split, index)
                    for index in range(30)
                ]
                generator.write_jsonl(source / f"derive_previous_{split}.jsonl", rows)
            data = root / "data"
            generator.generate(config_path, source, data)
            report = validator.validate(config_path, source, data)
            self.assertEqual(report["transition_validation"]["positions"], 15)
            rows = generator.read_jsonl(data / "transition_validation.jsonl")
            self.assertEqual(
                {int(row["spec"]["step"]) for row in rows}, set(range(1, 16))
            )

    def test_summary_reads_all_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
            config["lengths"] = [2, 4]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            metric = {
                "n": 4,
                "trace_bit_accuracy": 1.0,
                "trace_exact": 1.0,
                "final_parity_accuracy": 1.0,
                "joint_exact": 1.0,
                "strict_format": 1.0,
                "output_exact": 1.0,
                "samples": [],
            }
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "passes_gate": True,
                        "final_by_length": {"2": metric, "4": metric},
                        "final_prerequisites": {"full_update": {"accuracy": 1.0}},
                        "final_transitions": {
                            "overall": {"accuracy": 1.0},
                            "by_position": {"transition_step_1": {"accuracy": 1.0}},
                        },
                        "elapsed_seconds": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            result = summary.summarize(config_path, root)
            self.assertTrue(result["passes_all_gates"])
            self.assertEqual(len(result["metrics"]), 2)


if __name__ == "__main__":
    unittest.main()
