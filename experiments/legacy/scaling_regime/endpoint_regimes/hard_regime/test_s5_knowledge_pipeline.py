from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chain_of_lies.evaluation.rewards import score_completion
from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix
from chain_of_lies.variants.steganography.hard_task_channels import decode_payload
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.knowledge.generate_splits import (
    generate as generate_knowledge,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.knowledge.validate_splits import (
    validate as validate_knowledge,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.s5.generate_splits import (
    generate as generate_s5,
)
from experiments.legacy.scaling_regime.endpoint_regimes.hard_regime.s5.validate_splits import (
    validate as validate_s5,
)


class HardTaskPipelineTest(unittest.TestCase):
    def _smoke_config(self, source: Path, destination: Path) -> Path:
        config = json.loads(source.read_text(encoding="utf-8"))
        config["train_n"] = 12
        config["val_n"] = 4
        destination.write_text(json.dumps(config), encoding="utf-8")
        return destination

    def _run_suite(self, task: str) -> None:
        experiment_dir = Path(__file__).parent / task
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = self._smoke_config(
                experiment_dir / "config.json", root / "config.json"
            )
            data_root = root / "data"
            if task == "s5":
                manifest = generate_s5(config_path, data_root)
                report = validate_s5(config_path, data_root)
                variant = "s5_steg_local_invisible"
            else:
                manifest = generate_knowledge(config_path, data_root)
                report = validate_knowledge(config_path, data_root)
                variant = "knowledge_5fact_steg_local_invisible"
            self.assertTrue(report["valid"])
            self.assertEqual(manifest["cross_seed_component_overlap"], 0)
            sample_path = next(
                (data_root / "seed_0" / variant / "train_prompts").glob("*.json")
            )
            record = json.loads(sample_path.read_text(encoding="utf-8"))
            suffix = _canonical_public_cot_suffix(
                record, supervision_mode="local_channel_cot"
            )
            self.assertIsNotNone(suffix)
            reasoning = suffix.split("<ANSWER>", 1)[0]
            self.assertEqual(
                decode_payload(record, reasoning),
                record["spec"]["steganography_payload"],
            )
            scored = score_completion(record, suffix or "")
            self.assertTrue(scored.task_success)
            self.assertTrue(scored.flags["payload_decode_correct"])

    def test_s5(self) -> None:
        self._run_suite("s5")

    def test_knowledge(self) -> None:
        self._run_suite("knowledge")


if __name__ == "__main__":
    unittest.main()
