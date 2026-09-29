import json
from pathlib import Path
import tempfile
import unittest

from experiments.scaling_regime.protocols.full_cot.evaluation_efficiency import (
    atomic_json,
    digest,
    select_prompts,
    valid_response,
    cache_options,
    grouping_decision,
)


class EfficiencyTests(unittest.TestCase):
    def test_grouping_requires_agreement_and_speedup(self):
        self.assertTrue(grouping_decision(100, 80, []))
        self.assertFalse(grouping_decision(100, 99, []))
        self.assertFalse(grouping_decision(100, 80, ["changed_output"]))
        self.assertFalse(grouping_decision(100, 0, []))

    def test_legacy_evaluation_helper_compatibility(self):
        def legacy(prompts, responses, batch_size=2):
            pass

        self.assertEqual(cache_options(legacy, "evaluate", True), {})
        with self.assertRaisesRegex(RuntimeError, "sync that file"):
            cache_options(legacy, "benchmark", True)

    def test_current_cache_controls_preserved(self):
        def current(prompts, responses, reset_model_cache=True, keep_model_cache=False):
            pass

        self.assertEqual(
            cache_options(current, "benchmark", False),
            {"reset_model_cache": False, "keep_model_cache": True},
        )

    def test_atomic_response_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "nested" / "example.json"
            value = {
                "experiment_id": "example",
                "raw_text": "answer",
                "generated_token_ids": [1, 2],
            }
            atomic_json(file, value)
            self.assertEqual(valid_response(file, "example", 2), value)
            before = digest(file)
            atomic_json(file, {**value, "raw_text": "changed"})
            self.assertNotEqual(digest(file), before)
            self.assertFalse(file.with_suffix(".json.tmp").exists())

    def test_reject_bad_responses(self):
        good = {
            "experiment_id": "example",
            "raw_text": "answer",
            "generated_token_ids": [1, 2],
        }
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "example.json"
            for change in (
                {"experiment_id": "wrong"},
                {"raw_text": ""},
                {"generated_token_ids": []},
                {"generated_token_ids": [True]},
                {"generated_token_ids": [-1]},
                {"generated_token_ids": [1, 2, 3]},
            ):
                atomic_json(file, {**good, **change})
                with self.assertRaises(ValueError):
                    valid_response(file, "example", 2)
            file.write_text("{")
            with self.assertRaises(json.JSONDecodeError):
                valid_response(file, "example", 2)

    def test_sampling_covers_every_length_and_longest(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for length in range(1, 20):
                bucket = data / "per_length" / str(length)
                bucket.mkdir(parents=True)
                for i in range(6):
                    (bucket / f"{i}.json").write_text("{}")
            selected = select_prompts(data, list(range(1, 20)), 2)
            self.assertEqual(len(selected), 38)
            self.assertEqual(
                {f.parent.name for f in selected}, {str(i) for i in range(1, 20)}
            )
            self.assertEqual(selected[-1].parent.name, "19")
            self.assertEqual(selected, select_prompts(data, list(range(1, 20)), 2))
            with self.assertRaises(ValueError):
                select_prompts(data, [1], 7)


if __name__ == "__main__":
    unittest.main()
