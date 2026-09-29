"""Fast configuration and reporting tests; no model download is required."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def load_summary():
    spec = importlib.util.spec_from_file_location(
        "composition_rank_summary", HERE / "summarize.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary_module = load_summary()


class PipelineTests(unittest.TestCase):
    def test_rank_control_is_well_formed(self) -> None:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["ranks"], [32, 64, 128])
        self.assertEqual(config["condition"], "derive_previous")
        self.assertEqual(
            config["target_modules"],
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def test_summary_selects_lowest_successful_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                (HERE / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            for rank, accuracy, passes in (
                (32, 0.50, False),
                (64, 0.97, True),
                (128, 0.99, True),
            ):
                directory = root / f"rank_{rank}"
                directory.mkdir()
                metrics = {
                    "trainable_parameters": rank * 100,
                    "best_epoch": 2,
                    "final_evaluations": {
                        "composition": {"greedy_accuracy": accuracy, "bit_ce": 0.1},
                        "local": {"greedy_accuracy": 0.98},
                        "supplied_update": {"greedy_accuracy": 1.0},
                        "other_position": {"greedy_accuracy": 0.5},
                    },
                    "final_attention": {
                        "selected_attention_mass": 0.99,
                        "selected_seed_ratio": 0.99,
                    },
                    "passes_gate": passes,
                    "elapsed_seconds": 1.0,
                }
                (directory / "metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            result = summary_module.summarize(config_path, root)
            self.assertEqual(result["lowest_passing_rank"], 64)
            self.assertIn("capacity", result["diagnosis"])


if __name__ == "__main__":
    unittest.main()
