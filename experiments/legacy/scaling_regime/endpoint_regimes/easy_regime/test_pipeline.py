from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.generate_splits import (
    TASKS,
    generate,
    variant,
)
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.validate_splits import (
    validate,
)


class EasyPipelineTest(unittest.TestCase):
    def test_matched_suite_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((Path(__file__).with_name("config.json")).read_text())
            config["train_n"] = 12
            config["val_n"] = 4
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            data = root / "data"
            generate(config_path, data)
            self.assertTrue(validate(config_path, data)["valid"])
            for task in TASKS:
                control = json.loads(
                    next(
                        (
                            data
                            / task
                            / "seed_0"
                            / variant(task, "control")
                            / "train_prompts"
                        ).glob("*.json")
                    ).read_text()
                )
                steg = json.loads(
                    next(
                        (
                            data
                            / task
                            / "seed_0"
                            / variant(task, "steg_local_invisible")
                            / "train_prompts"
                        ).glob("*.json")
                    ).read_text()
                )
                self.assertEqual(
                    control["spec"]["private_question"],
                    steg["spec"]["private_question"],
                )
                self.assertIsNotNone(
                    _canonical_public_cot_suffix(
                        steg, supervision_mode="local_channel_cot"
                    )
                )
                control["spec"]["filler_token_count"] = 64
                filler = _canonical_public_cot_suffix(
                    control, supervision_mode="filler_public_cot"
                )
                self.assertEqual((filler or "").count(FILLER_TOKEN_MARKER), 1)


if __name__ == "__main__":
    unittest.main()
