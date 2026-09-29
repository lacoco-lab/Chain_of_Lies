"""Matched seed-free controls using the final standard Transformer block."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOY_ROOT = Path(__file__).resolve().parents[1] / "main_experiment"
if str(TOY_ROOT) not in sys.path:
    sys.path.insert(0, str(TOY_ROOT))

from standard_model import StandardGoldreichTransformer  # noqa: E402

CONDITIONS = ("no_cot", "filler_cot", "normal_cot")


class StandardParityControl(StandardGoldreichTransformer):
    """The encrypted experiment's block with public zeros replacing the seed.

    The inherited mask heads and value probe remain inert, exactly as in the
    final v2 encrypted trainer. Only the primary bit output is used.
    """

    def zero_seeds(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch, self.seed_bits), dtype=torch.long, device=device)

    def control_targets(
        self, steps: torch.Tensor, condition: str, direct_lengths: set[int]
    ) -> torch.Tensor | None:
        """Return public route labels when the control has a known local route."""
        if condition == "normal_cot":
            rows = []
            for raw_step in steps.detach().cpu().tolist():
                step = int(raw_step)
                previous = 0 if step == 0 else self.cot_start + step - 1
                current = self.input_start + step if step < self.length else 0
                rows.append([previous, current, *([0] * (self.heads - 2))])
            return torch.tensor(rows, dtype=torch.long, device=steps.device)

        if self.length not in direct_lengths:
            return None
        rows = []
        for raw_step in steps.detach().cpu().tolist():
            step = int(raw_step)
            if condition == "filler_cot" and step < self.length:
                rows.append([0] * self.heads)
            else:
                # Use the same one-head-per-input route as the easy controls.
                # When N exceeds the fixed 16-head budget, the staged public
                # mechanism covers the largest representable prefix. The final
                # integration stage still trains and evaluates on all N bits.
                represented = min(self.length, self.heads)
                positions = [self.input_start + index for index in range(represented)]
                rows.append(positions + [0] * (self.heads - represented))
        return torch.tensor(rows, dtype=torch.long, device=steps.device)

    def control_sequence(
        self, inputs: torch.Tensor, prefixes: torch.Tensor, steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seeds = self.zero_seeds(len(inputs), inputs.device)
        return self.build_training_sequence(seeds, inputs, prefixes, steps)

    def control_cache(self, inputs: torch.Tensor):
        return self.initialize_decode_cache(
            self.zero_seeds(len(inputs), inputs.device), inputs
        )
