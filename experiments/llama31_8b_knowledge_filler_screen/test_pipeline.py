from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import (
    build_messages,
    component_probes,
    load_task_items,
    parse_answer,
    score_answer,
)
from prepare_data import prepare
from summarize import _mcnemar_exact, summarize


class LlamaKnowledgeFillerTests(unittest.TestCase):
    config_path = Path("experiments/llama31_8b_knowledge_filler_screen/config.json")

    def _prepared_config(self, root: Path) -> Path:
        output_root = root / "prepared"
        prepare(config_path=self.config_path, output_root=output_root)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["tasks"]["one_fact"]["dataset"] = str(output_root / "one_fact_addition.json")
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_official_and_generated_dataset_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._prepared_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(len(load_task_items(config, "one_fact")), 800)
            self.assertEqual(len(load_task_items(config, "two_fact")), 1500)
            letters = load_task_items(config, "letter_position")
            self.assertEqual(len(letters), 647)
            self.assertEqual(sum(row["subtask"] == "element_letter" for row in letters), 285)
            self.assertEqual(sum(row["subtask"] == "capital_letter" for row in letters), 362)

    def test_paper_prompt_has_five_shots_and_exact_dot_count_per_region(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._prepared_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            item = load_task_items(config, "two_fact")[0]
            messages = build_messages(item, 25)
            self.assertEqual(len(messages), 12)
            user_messages = [row["content"] for row in messages if row["role"] == "user"]
            self.assertEqual(len(user_messages), 6)
            for content in user_messages:
                filler = content.split("Filler: ", 1)[1].split("\n\nAnswer:", 1)[0]
                self.assertEqual(filler.split(), ["."] * 25)
            baseline = build_messages(item, 0)
            self.assertTrue(all("Filler:" not in row["content"] for row in baseline))

    def test_component_probes_cover_each_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._prepared_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for task in ("one_fact", "two_fact", "letter_position"):
                items = load_task_items(config, task)
                probes = component_probes(items, task)
                self.assertGreater(len(probes), 0)
                self.assertEqual(len({row["probe_id"] for row in probes}), len(probes))

    def test_answer_parsing(self) -> None:
        self.assertEqual(parse_answer("142", "integer"), 142)
        self.assertEqual(parse_answer("Answer: 142", "integer"), 142)
        self.assertEqual(parse_answer("l", "letter"), "l")
        self.assertEqual(parse_answer("Answer: L", "letter"), "l")
        self.assertTrue(score_answer("The answer is Brasília.", "Brasília", "string")[1])

    def test_mcnemar_exact(self) -> None:
        self.assertEqual(_mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(_mcnemar_exact(6, 0), 0.03125)

    def test_synthetic_positive_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            config["filler_lengths"] = [0, 50, 100]
            config["bootstrap_samples"] = 1000
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            responses = root / "responses"
            for task in config["tasks"]:
                rows = []
                for k in config["filler_lengths"]:
                    for index in range(20):
                        correct = index < (4 if k == 0 else 18)
                        subtask = (
                            "element_letter"
                            if task == "letter_position" and index < 10
                            else "capital_letter"
                            if task == "letter_position"
                            else task
                        )
                        rows.append(
                            {
                                "item_id": f"{task}_{index}",
                                "task": task,
                                "subtask": subtask,
                                "filler_length": k,
                                "correct": correct,
                                "predicted": 1 if correct else None,
                                "prompt_tokens": 100 + k,
                                "filler_model_tokens_per_region": k,
                            }
                        )
                payload = {
                    "rows": rows,
                    "component_rows": [
                        {
                            "item_id": f"probe_{index}",
                            "subtask": "fact",
                            "correct": True,
                        }
                        for index in range(20)
                    ],
                }
                directory = responses / task
                directory.mkdir(parents=True)
                (directory / f"{task}.json").write_text(json.dumps(payload), encoding="utf-8")
            summary = summarize(
                config_path=config_path,
                responses_dir=responses,
                artifacts_dir=root / "artifacts",
            )
            self.assertTrue(summary["any_medium_filler_signal"])
            self.assertTrue(all(row["medium_filler_signal"] for row in summary["task_gates"].values()))


if __name__ == "__main__":
    unittest.main()
