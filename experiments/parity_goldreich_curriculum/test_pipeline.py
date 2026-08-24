from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import encrypted, fixed_graph, generate, masks, predicate, states
from summarize import summarize
from validate_data import validate


class GoldreichCurriculumTests(unittest.TestCase):
    def _small_config(self, root: Path) -> Path:
        config = json.loads(Path("experiments/parity_goldreich_curriculum/config.json").read_text())
        config.update(
            {
                "max_mask_bits": 4,
                "lengths": [2, 4],
                "base_train_n": 40,
                "base_validation_n": 20,
                "local_train_n": 64,
                "local_validation_n": 32,
                "curriculum_train_n": 40,
                "update_stages": [
                    {"name": "local_update", "epochs": 1, "evaluation_task": "local_update"},
                    {"name": "supplied_2", "epochs": 1, "evaluation_task": "supplied_2"},
                    {"name": "supplied_4", "epochs": 1, "evaluation_task": "supplied_4"},
                ],
                "goldreich_stages": [
                    {"name": "predicate_local", "epochs": 1, "evaluation_task": "predicate_local"},
                    {"name": "mask_2", "epochs": 1, "evaluation_task": "mask_2"},
                    {"name": "mask_4", "epochs": 1, "evaluation_task": "mask_4"},
                    {"name": "joint_2", "epochs": 1, "evaluation_task": "joint_2"},
                    {"name": "joint_4", "epochs": 1, "evaluation_task": "joint_4"},
                ],
            }
        )
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_formulas(self) -> None:
        graph = [[0, 1, 2, 3, 4], [0, 3, 4, 1, 2]]
        self.assertEqual(predicate([1, 0, 0, 1, 1]), 0)
        self.assertEqual(masks("10011010", graph, 2), [0, 1])
        self.assertEqual(states("1011"), [1, 1])
        self.assertEqual(encrypted("1011", [0, 1]), [1, 0])
        self.assertEqual(fixed_graph(16, 4, 3), fixed_graph(16, 4, 3))

    def test_generation_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._small_config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(first["digests"], second["digests"])
            report = validate(config, root / "first")
            self.assertTrue(report["targets_valid"])
            self.assertEqual(report["train_validation_seed_overlap"], 0)

    def test_perfect_synthetic_responses_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._small_config(root)
            data = root / "data"
            artifacts = root / "artifacts"
            responses = root / "responses"
            generate(config_path, data)
            config = json.loads(config_path.read_text())
            for track, key in (("update", "update_stages"), ("goldreich", "goldreich_stages")):
                entries = []
                for index, stage in enumerate(config[key], start=1):
                    label = f"{index:02d}_{stage['name']}"
                    task = stage["evaluation_task"]
                    destination = responses / track / "stages" / label / "epoch_1" / task
                    destination.mkdir(parents=True)
                    for prompt_path in (data / "validation" / task).glob("*.json"):
                        prompt = json.loads(prompt_path.read_text())
                        (destination / f"{prompt['experiment_id']}.json").write_text(
                            json.dumps({"raw_text": prompt["supervised_suffix"]}), encoding="utf-8"
                        )
                    entries.append(
                        {
                            "stage_index": index,
                            "stage": stage["name"],
                            "epoch": 1,
                            "stage_epochs": 1,
                            "evaluation_task": task,
                        }
                    )
                final_tasks = (
                    ["local_update", "supplied_2", "supplied_4"]
                    if track == "update"
                    else [
                        "local_update",
                        "predicate_local",
                        "supplied_4",
                        "mask_2",
                        "mask_4",
                        "joint_2",
                        "joint_4",
                    ]
                )
                for task in final_tasks:
                    destination = responses / track / "final" / task
                    destination.mkdir(parents=True)
                    for prompt_path in (data / "validation" / task).glob("*.json"):
                        prompt = json.loads(prompt_path.read_text())
                        (destination / f"{prompt['experiment_id']}.json").write_text(
                            json.dumps({"raw_text": prompt["supervised_suffix"]}), encoding="utf-8"
                        )
                track_root = artifacts / track
                track_root.mkdir(parents=True)
                (track_root / "track_manifest.json").write_text(
                    json.dumps({"stages": entries, "final_evaluation_tasks": final_tasks}),
                    encoding="utf-8",
                )
            summary = summarize(
                config_path=config_path,
                data_root=data,
                artifacts_root=artifacts,
                responses_root=responses,
            )
            self.assertTrue(summary["passes_full_mechanism"])
            self.assertIsNone(summary["first_failed_mechanism"])


if __name__ == "__main__":
    unittest.main()
