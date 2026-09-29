"""Matched one-layer hard-attention transformer for seed-free PARITY controls."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

CONSTANT_POSITION = 0
STATE_POSITION = 1
PADDING_START = 2
HEADS = 12


class BooleanGate(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.float())


class ParityControlTransformer(nn.Module):
    """A matched UHAT model with no private seed or encryption mechanism.

    The public-zero padding keeps the source layout and attention parameter
    count matched to the encrypted model. It is constant for every example
    and therefore contains no private random information.
    """

    def __init__(
        self, length: int, padding_bits: int, key_dim: int, gate_hidden_dim: int
    ) -> None:
        super().__init__()
        if length < 1 or padding_bits < 0:
            raise ValueError("Length must be positive and padding non-negative")
        self.length = length
        self.padding_bits = padding_bits
        self.steps = length
        self.input_start = PADDING_START + padding_bits
        self.sources = self.input_start + length
        self.keys = nn.Parameter(
            torch.randn(HEADS, self.sources, key_dim) / math.sqrt(key_dim)
        )
        self.queries = nn.Parameter(
            torch.randn(self.steps, HEADS, key_dim) / math.sqrt(key_dim)
        )
        self.xor_gate = BooleanGate(2, gate_hidden_dim)
        # Present in every condition so model size is identical. Only the
        # filler condition trains it; it cannot store example-specific state.
        self.filler_logits = nn.Parameter(torch.zeros(2))

    def normal_routing_targets(self, steps: torch.Tensor) -> torch.Tensor:
        rows = []
        for raw_step in steps.detach().cpu().tolist():
            step = int(raw_step)
            rows.append(
                [
                    STATE_POSITION,
                    self.input_start + step,
                    *([CONSTANT_POSITION] * (HEADS - 2)),
                ]
            )
        return torch.tensor(rows, dtype=torch.long, device=steps.device)

    def direct_routing_targets(self, steps: torch.Tensor) -> torch.Tensor:
        """Retrieve every input bit when the fixed head budget permits it."""
        if self.length > HEADS:
            raise ValueError(
                f"Direct PARITY at N={self.length} needs {self.length} heads, "
                f"but this matched model has {HEADS}"
            )
        row = [self.input_start + index for index in range(self.length)]
        row.extend([CONSTANT_POSITION] * (HEADS - self.length))
        return torch.tensor([row] * len(steps), dtype=torch.long, device=steps.device)

    def attention_scores(self, steps: torch.Tensor) -> torch.Tensor:
        queries = self.queries[steps]
        return torch.einsum("bhd,hsd->bhs", queries, self.keys) / math.sqrt(
            self.keys.shape[-1]
        )

    def build_sources(
        self, input_bits: torch.Tensor, previous_state: torch.Tensor
    ) -> torch.Tensor:
        batch = input_bits.shape[0]
        dtype = input_bits.dtype
        zero = torch.zeros((batch, 1), dtype=dtype, device=input_bits.device)
        padding = torch.zeros(
            (batch, self.padding_bits), dtype=dtype, device=input_bits.device
        )
        return torch.cat(
            (
                zero,
                previous_state.to(dtype).reshape(-1, 1),
                padding,
                input_bits.to(dtype),
            ),
            dim=1,
        )

    def route(
        self, sources: torch.Tensor, steps: torch.Tensor, hard: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_steps, inverse = torch.unique(steps, sorted=True, return_inverse=True)
        scores = self.attention_scores(unique_steps)
        if hard:
            selected = scores.argmax(dim=-1)[inverse]
            return sources.gather(1, selected).float(), selected
        probabilities = scores.softmax(dim=-1)[inverse]
        values = torch.einsum("bhs,bs->bh", probabilities, sources.float())
        return values, probabilities

    def _reduce_xor_soft(self, values: torch.Tensor) -> torch.Tensor:
        current = values[:, 0]
        logits = None
        for index in range(1, HEADS):
            logits = self.xor_gate(torch.stack((current, values[:, index]), dim=-1))
            current = logits.softmax(dim=-1)[:, 1]
        assert logits is not None
        return logits

    def soft_step(
        self,
        input_bits: torch.Tensor,
        previous_state: torch.Tensor,
        steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sources = self.build_sources(input_bits, previous_state)
        values, attention = self.route(sources, steps, hard=False)
        return {"logits": self._reduce_xor_soft(values), "attention": attention}

    @torch.no_grad()
    def hard_step(
        self,
        input_bits: torch.Tensor,
        previous_state: torch.Tensor,
        steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sources = self.build_sources(input_bits, previous_state)
        values, selected = self.route(sources, steps, hard=True)
        current = values[:, 0].long()
        for index in range(1, HEADS):
            current = self.xor_gate(
                torch.stack((current, values[:, index].long()), dim=-1)
            ).argmax(dim=-1)
        return {"state": current, "selected_positions": selected}

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "layers": 1,
            "heads": HEADS,
            "sources": self.sources,
            "public_zero_padding_bits": self.padding_bits,
            "key_dim": self.keys.shape[-1],
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }
