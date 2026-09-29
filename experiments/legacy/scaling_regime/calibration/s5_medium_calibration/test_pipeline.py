from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_splits import generate
from validate_splits import validate

from chain_of_lies.training.ce.trainer import _encode_supervised_suffix
from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)


class _AtomicDotTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_kwargs):
        ids = [46 if char == "." else 1000 + ord(char) for char in text]
        return SimpleNamespace(input_ids=ids)


class MediumCalibrationTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = json.loads(
            Path(
                "experiments/legacy/scaling_regime/calibration/s5_medium_calibration/config.json"
            ).read_text()
        )
        config["lengths"] = [1, 2, 4]
        config["train_per_length"] = 12
        config["validation_per_length"] = 4
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_generation_is_balanced_independent_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(
                first["validation_digest_sha256"], second["validation_digest_sha256"]
            )
            report = validate(root / "first", config)
            self.assertEqual(report["train_by_length"], {1: 12, 2: 12, 4: 12})
            self.assertEqual(report["validation_by_length"], {1: 4, 2: 4, 4: 4})
            self.assertEqual(report["train_val_overlap"], 0)

    def test_filler_target_inserts_exact_atomic_token_ids(self) -> None:
        record = {
            "task_type": "s5",
            "spec": {
                "initial_state": "BCADE",
                "public_initial_state": "BCADE",
                "private_initial_state": "BCADE",
                "public_instruction_sequence": ["swap(1,2)"],
                "private_instruction_sequence": ["swap(3,4)"],
                "public_answer": "CBADE",
                "private_answer": "BCDAE",
                "filler_token_count": 64,
                "filler_token_text": ".",
            },
        }
        suffix = _canonical_public_cot_suffix(
            record, supervision_mode="filler_public_cot"
        )
        self.assertIsNotNone(suffix)
        self.assertEqual((suffix or "").count(FILLER_TOKEN_MARKER), 1)
        token_ids = _encode_supervised_suffix(
            _AtomicDotTokenizer(), suffix or "", record
        )
        longest_run = current_run = 0
        for token_id in token_ids:
            current_run = current_run + 1 if token_id == 46 else 0
            longest_run = max(longest_run, current_run)
        self.assertEqual(longest_run, 64)


if __name__ == "__main__":
    unittest.main()
