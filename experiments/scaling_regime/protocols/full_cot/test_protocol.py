import copy
import json
from pathlib import Path
import unittest

from experiments.scaling_regime.protocols.full_cot.protocol import derive
from experiments.scaling_regime.protocols.filler.experiment import (
    load_config,
    _directories,
)
from chain_of_lies.training.shared.trainer_utils import (
    _public_cot_prefix,
    _answer_block_suffix,
    _canonical_public_cot_suffix,
)


class ProtocolTests(unittest.TestCase):
    def test_all_tasks_seeds_and_difficulties(self):
        config = load_config(
            Path("experiments/scaling_regime/protocols/filler/config.json")
        )
        count = 0
        for task, spec in config["tasks"].items():
            for seed in config["seeds"]:
                train, evaluation = _directories(spec, seed)
                if not train.is_dir() or not evaluation.is_dir():
                    self.skipTest(
                        "Complete generated records are required for this integration test."
                    )
                for directory in (train, evaluation):
                    remaining = {str(x) for x in spec["difficulty_values"]}
                    for file in sorted(directory.glob("*.json")):
                        original = json.loads(file.read_text())
                        difficulty = str(original["spec"][spec["difficulty_field"]])
                        if difficulty not in remaining:
                            continue
                        remaining.remove(difficulty)
                        snapshot = copy.deepcopy(original)
                        result = derive(
                            original, _public_cot_prefix, _answer_block_suffix
                        )
                        self.assertEqual(original, snapshot)
                        self.assertEqual(result["spec"], original["spec"])
                        self.assertEqual(
                            result["experiment_id"], original["experiment_id"]
                        )
                        self.assertTrue(
                            result["supervised_suffix"].endswith(
                                _answer_block_suffix(original)
                            )
                        )
                        self.assertIn(
                            "Reasoning for other question:", result["supervised_suffix"]
                        )
                        self.assertEqual(
                            _canonical_public_cot_suffix(
                                result, supervision_mode="record_target"
                            ),
                            result["supervised_suffix"],
                        )
                        self.assertNotIn("Do not mention", result["prompt_text"])
                        self.assertNotIn("only the main", result["prompt_text"])
                        self.assertIn(
                            "Show ordinary, human-readable reasoning for BOTH",
                            result["system_prompt"],
                        )
                        count += 1
                        if not remaining:
                            break
                    self.assertFalse(remaining, (task, seed, directory, remaining))
        self.assertEqual(count, 192)


if __name__ == "__main__":
    unittest.main()
