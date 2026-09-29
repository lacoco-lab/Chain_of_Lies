from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import (
    encrypted,
    fixed_graph,
    generate,
    mask_bit,
    predicate,
    selector,
    states,
)
from summarize import summarize
from validate_data import validate


class ExplicitHintPipelineTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = json.loads(
            Path(
                "experiments/goldreich_parity/qwen_attempts/exploratory/explicit_hints/config.json"
            ).read_text()
        )
        config.update(
            {
                "max_steps": 4,
                "core_lengths": [2],
                "extension_lengths": [4],
                "local_train_n": 64,
                "local_validation_n": 32,
                "trace_train_n": 40,
                "trace_validation_n": 20,
                "curriculum_train_n": 40,
                "core_stages": [
                    {
                        "name": "predicate_local",
                        "epochs": 1,
                        "evaluation_task": "predicate_local",
                    },
                    {
                        "name": "hinted_mask_local",
                        "epochs": 1,
                        "evaluation_task": "hinted_mask_local",
                    },
                    {
                        "name": "hinted_start_local",
                        "epochs": 1,
                        "evaluation_task": "hinted_start_local",
                    },
                    {
                        "name": "hinted_update_local",
                        "epochs": 1,
                        "evaluation_task": "hinted_update_local",
                    },
                    {
                        "name": "hinted_joint_2",
                        "epochs": 1,
                        "evaluation_task": "hinted_joint_2",
                    },
                ],
                "extension_stages": [
                    {
                        "name": "hinted_joint_4",
                        "epochs": 1,
                        "evaluation_task": "hinted_joint_4",
                    },
                ],
            }
        )
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_formulas_and_selector(self) -> None:
        graph = [[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]
        self.assertEqual(predicate([1, 0, 0, 1, 1]), 0)
        self.assertEqual(mask_bit("100110", graph[0]), 0)
        self.assertEqual(selector(graph[1]), "(use Seed: F E D C B)")
        self.assertEqual(states("1011"), [1, 1])
        self.assertEqual(encrypted("1011", [0, 1]), [1, 0])
        self.assertEqual(fixed_graph(16, 4, 7), fixed_graph(16, 4, 7))

    def test_generation_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(first["digests"], second["digests"])
            report = validate(config, root / "first")
            self.assertTrue(report["public_targets_hide_seed_values_and_masks"])
            self.assertTrue(report["complementary_twins_valid"])

    def test_perfect_responses_pass_all_stage_and_retention_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            data, artifacts, responses = (
                root / "data",
                root / "artifacts",
                root / "responses",
            )
            generate(config_path, data)
            config = json.loads(config_path.read_text())
            for track in ("core", "extension"):
                entries = []
                for index, stage in enumerate(config[f"{track}_stages"], start=1):
                    task, label = (
                        stage["evaluation_task"],
                        f"{index:02d}_{stage['name']}",
                    )
                    destination = (
                        responses / track / "stages" / label / "epoch_1" / task
                    )
                    destination.mkdir(parents=True)
                    self._perfect(data / "validation" / task, destination)
                    entries.append(
                        {
                            "stage_index": index,
                            "stage": stage["name"],
                            "epoch": 1,
                            "stage_epochs": 1,
                            "evaluation_task": task,
                        }
                    )
                lengths = [2] if track == "core" else [2, 4]
                final_tasks = [
                    "predicate_local",
                    "hinted_mask_local",
                    "hinted_start_local",
                    "hinted_update_local",
                ] + [f"hinted_joint_{length}" for length in lengths]
                for task in final_tasks:
                    destination = responses / track / "final" / task
                    destination.mkdir(parents=True)
                    self._perfect(data / "validation" / task, destination)
                track_root = artifacts / track
                track_root.mkdir(parents=True)
                (track_root / "track_manifest.json").write_text(
                    json.dumps(
                        {"stages": entries, "final_evaluation_tasks": final_tasks}
                    ),
                    encoding="utf-8",
                )
            summary = summarize(config_path, data, artifacts, responses)
            self.assertTrue(summary["passes_core_mechanism"])
            self.assertTrue(summary["passes_64_extension"])
            self.assertIsNone(summary["first_failed_stage"])

    @staticmethod
    def _perfect(prompts: Path, destination: Path) -> None:
        for path in prompts.glob("*.json"):
            prompt = json.loads(path.read_text())
            (destination / path.name).write_text(
                json.dumps({"raw_text": prompt["supervised_suffix"]}), encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
