#!/usr/bin/env python3
"""Fast structural tests for the matched PARITY controls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import (
    CONSTANT_POSITION,
    HEADS,
    STATE_POSITION,  # noqa: E402
    ParityControlTransformer,
)
from generate_easy_data import partition_domain  # noqa: E402
from train import audit_splits, sanitize_evaluation_overlap  # noqa: E402


class ControlTests(unittest.TestCase):
    def test_easy_domains_are_exhaustive_disjoint_and_balanced(self) -> None:
        for length, sizes in {
            4: (8, 4, 4),
            8: (128, 64, 64),
            16: (32768, 16384, 16384),
        }.items():
            parts = partition_domain(length, 0)
            self.assertEqual(tuple(map(len, parts.values())), sizes)
            sets = {split: set(values) for split, values in parts.items()}
            self.assertFalse(sets["train"] & sets["validation"])
            self.assertFalse(sets["train"] & sets["test"])
            self.assertFalse(sets["validation"] & sets["test"])
            self.assertEqual(len(set().union(*sets.values())), 1 << length)
            for values in parts.values():
                counts = [
                    sum(value.bit_count() % 2 == label for value in values)
                    for label in (0, 1)
                ]
                self.assertEqual(counts[0], counts[1])

    def test_matched_source_shape_contains_public_zeros_not_seed(self) -> None:
        model = ParityControlTransformer(32, 12, 32, 128)
        inputs = torch.randint(0, 2, (3, 32))
        sources = model.build_sources(inputs, torch.tensor([0, 1, 0]))
        self.assertEqual(sources.shape, (3, 46))
        self.assertTrue(torch.equal(sources[:, 0], torch.zeros(3, dtype=inputs.dtype)))
        self.assertTrue(torch.equal(sources[:, 1], torch.tensor([0, 1, 0])))
        self.assertEqual(int(sources[:, 2:14].sum()), 0)
        self.assertTrue(torch.equal(sources[:, 14:], inputs))

    def test_normal_routing_uses_previous_state_and_current_input(self) -> None:
        model = ParityControlTransformer(8, 5, 16, 32)
        targets = model.normal_routing_targets(torch.tensor([0, 7]))
        self.assertEqual(targets.shape, (2, HEADS))
        self.assertEqual(targets[:, 0].tolist(), [STATE_POSITION, STATE_POSITION])
        self.assertEqual(
            targets[:, 1].tolist(), [model.input_start, model.input_start + 7]
        )
        self.assertEqual(targets[:, 2:].unique().tolist(), [CONSTANT_POSITION])

    def test_direct_routing_covers_every_input_when_representable(self) -> None:
        model = ParityControlTransformer(8, 5, 16, 32)
        targets = model.direct_routing_targets(torch.tensor([7]))[0]
        self.assertEqual(
            targets[:8].tolist(), [model.input_start + index for index in range(8)]
        )
        self.assertEqual(targets[8:].unique().tolist(), [CONSTANT_POSITION])
        larger = ParityControlTransformer(16, 7, 16, 32)
        with self.assertRaises(ValueError):
            larger.direct_routing_targets(torch.tensor([15]))

    def test_parameter_matching_is_within_removed_predicate_gate(self) -> None:
        expected_encrypted = {
            32: 31_620,
            64: 58_884,
            256: 434_052,
            512: 850_308,
            1024: 1_674_372,
            2048: 3_308_676,
        }
        settings = {
            32: (12, 32),
            64: (19, 32),
            256: (49, 64),
            512: (79, 64),
            1024: (128, 64),
            2048: (208, 64),
        }
        for length, (padding, key_dim) in settings.items():
            model = ParityControlTransformer(length, padding, key_dim, 128)
            difference = (
                expected_encrypted[length]
                - model.architecture_metadata()["trainable_parameters"]
            )
            # The controls omit the unused five-input Goldreich predicate
            # module and add two shared filler logits: a net 1,024 parameters.
            self.assertEqual(difference, 1024)

    def test_validation_test_collision_is_removed_and_rebalanced(self) -> None:
        def part(names: list[str], labels: list[int]) -> dict:
            bits = torch.tensor(
                [[int(bit) for bit in value] for value in names], dtype=torch.uint8
            )
            return {
                "ids": names,
                "input_strings": names,
                "inputs": bits,
                "normal_states": torch.cumsum(bits.long(), dim=1).remainder(2).byte(),
                "answers": torch.tensor(labels, dtype=torch.long),
            }

        parts = {
            "train": part(["000", "001"], [0, 1]),
            "validation": part(["010", "011", "100", "101"], [0, 1, 0, 1]),
            "test": part(["010", "110"], [0, 1]),
        }
        report = sanitize_evaluation_overlap(parts)
        self.assertEqual(report["validation_test_colliding_inputs"], 1)
        self.assertEqual(report["validation_rows_removed"], 2)
        self.assertEqual(len(parts["validation"]["ids"]), 2)
        audit = audit_splits(parts)
        self.assertEqual(
            audit["input_overlap_counts"],
            {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
        )
        self.assertEqual(audit["label_balance"]["validation"], {"zero": 1, "one": 1})


if __name__ == "__main__":
    unittest.main()
