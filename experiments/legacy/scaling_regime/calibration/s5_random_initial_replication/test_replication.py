from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_splits import generate_variant
from validate_splits import validate_split_root

from chain_of_lies.variants.s5.data_generation.generate import (
    S5_VARIANT_CONTROL,
    S5_VARIANT_PIGGYBACK,
)


class RandomInitialS5ReplicationTests(unittest.TestCase):
    def test_small_splits_are_valid_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            for variant in (S5_VARIANT_PIGGYBACK, S5_VARIANT_CONTROL):
                first_manifest = generate_variant(
                    variant=variant,
                    seed=7,
                    train_n=30,
                    val_n=10,
                    length_range=(10, 19),
                    output_root=first,
                )
                second_manifest = generate_variant(
                    variant=variant,
                    seed=7,
                    train_n=30,
                    val_n=10,
                    length_range=(10, 19),
                    output_root=second,
                )
                self.assertEqual(
                    first_manifest["validation_digest_sha256"],
                    second_manifest["validation_digest_sha256"],
                )
                report = validate_split_root(first / "seed_7", variant, 30, 10)
                self.assertEqual(report["train_val_overlap"], 0)
                self.assertGreater(report["unique_initial_states"], 1)

    def test_only_final_swap_differs_in_piggyback_and_initial_state_is_shared(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_variant(
                variant=S5_VARIANT_PIGGYBACK,
                seed=3,
                train_n=10,
                val_n=5,
                length_range=(10, 19),
                output_root=root,
            )
            prompt_path = next(
                (root / "seed_3/s5_piggyback/val_prompts").glob("*.json")
            )
            record = json.loads(prompt_path.read_text(encoding="utf-8"))
            spec = record["spec"]
            self.assertEqual(spec["initial_state"], spec["public_initial_state"])
            self.assertEqual(spec["initial_state"], spec["private_initial_state"])
            self.assertEqual(spec["changed_positions"], [spec["sequence_length"] - 1])
            self.assertEqual(
                spec["public_instruction_sequence"][:-1],
                spec["private_instruction_sequence"][:-1],
            )
            self.assertNotEqual(
                spec["public_instruction_sequence"][-1],
                spec["private_instruction_sequence"][-1],
            )


if __name__ == "__main__":
    unittest.main()
