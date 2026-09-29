"""A small learned unique-hard-attention transformer for encrypted PARITY."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

CONSTANT_POSITION = 0
STATE_POSITION = 1
SEED_START = 2
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


class GoldreichToyTransformer(nn.Module):
    """One learned hard-attention layer followed by learned Boolean FFNs.

    Twelve fixed-role heads retrieve the previous encrypted state, one input
    bit, five previous-mask seed bits, and five current-mask seed bits. The
    public graph determines supervision targets but is never used to select a
    value during inference: hard inference selects each head's argmax score.
    """

    def __init__(
        self,
        length: int,
        seed_bits: int,
        graph: list[list[int]],
        key_dim: int,
        gate_hidden_dim: int,
    ) -> None:
        super().__init__()
        if length < 1 or seed_bits < 5 or len(graph) < length:
            raise ValueError(
                "Positive length, five seed bits, and one graph edge per step required"
            )
        self.length = length
        self.seed_bits = seed_bits
        self.steps = length
        self.input_start = SEED_START + seed_bits
        self.sources = self.input_start + length
        self.graph = [list(edge) for edge in graph[: self.steps]]
        self.keys = nn.Parameter(
            torch.randn(HEADS, self.sources, key_dim) / math.sqrt(key_dim)
        )
        self.queries = nn.Parameter(
            torch.randn(self.steps, HEADS, key_dim) / math.sqrt(key_dim)
        )
        self.predicate_gate = BooleanGate(5, gate_hidden_dim)
        self.xor_gate = BooleanGate(2, gate_hidden_dim)

    def routing_targets(self, steps: torch.Tensor) -> torch.Tensor:
        rows = []
        for raw_step in steps.detach().cpu().tolist():
            step = int(raw_step)
            previous = (
                [CONSTANT_POSITION] * 5
                if step == 0
                else [SEED_START + value for value in self.graph[step - 1]]
            )
            current = [SEED_START + value for value in self.graph[step]]
            rows.append(
                [
                    STATE_POSITION,
                    self.input_start + step,
                    *previous,
                    *current,
                ]
            )
        return torch.tensor(rows, dtype=torch.long, device=steps.device)

    def attention_scores(self, steps: torch.Tensor) -> torch.Tensor:
        queries = self.queries[steps]
        return torch.einsum("bhd,hsd->bhs", queries, self.keys) / math.sqrt(
            self.keys.shape[-1]
        )

    def route(
        self, source_bits: torch.Tensor, steps: torch.Tensor, hard: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_steps, inverse = torch.unique(steps, sorted=True, return_inverse=True)
        unique_scores = self.attention_scores(unique_steps)
        if hard:
            selected = unique_scores.argmax(dim=-1)[inverse]
            routed = source_bits.gather(1, selected)
            return routed.float(), selected
        else:
            probabilities = unique_scores.softmax(dim=-1)[inverse]
            routed = torch.einsum("bhs,bs->bh", probabilities, source_bits.float())
        return routed.float(), probabilities

    def build_sources(
        self,
        seed_bits: torch.Tensor,
        input_bits: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        batch = seed_bits.shape[0]
        dtype = seed_bits.dtype
        zero = torch.zeros((batch, 1), dtype=dtype, device=seed_bits.device)
        return torch.cat(
            (
                zero,
                previous_state.to(dtype).reshape(-1, 1),
                seed_bits.to(dtype),
                input_bits.to(dtype),
            ),
            dim=1,
        )

    def _xor_soft(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.xor_gate(torch.stack((left, right), dim=-1))
        return logits.softmax(dim=-1)[:, 1], logits

    def soft_step(
        self,
        seed_bits: torch.Tensor,
        input_bits: torch.Tensor,
        previous_state: torch.Tensor,
        steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sources = self.build_sources(seed_bits, input_bits, previous_state)
        bits, attention = self.route(sources, steps, hard=False)
        previous_mask_logits = self.predicate_gate(bits[:, 2:7])
        current_mask_logits = self.predicate_gate(bits[:, 7:12])
        previous_mask = previous_mask_logits.softmax(dim=-1)[:, 1]
        current_mask = current_mask_logits.softmax(dim=-1)[:, 1]
        value = bits[:, 0]
        xor_logits = []
        for operand in (bits[:, 1], previous_mask, current_mask):
            value, logits = self._xor_soft(value, operand)
            xor_logits.append(logits)
        return {
            "state_logits": xor_logits[-1],
            "previous_mask_logits": previous_mask_logits,
            "current_mask_logits": current_mask_logits,
            "attention": attention,
        }

    @torch.no_grad()
    def hard_step(
        self,
        seed_bits: torch.Tensor,
        input_bits: torch.Tensor,
        previous_state: torch.Tensor,
        steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sources = self.build_sources(seed_bits, input_bits, previous_state)
        bits, selected = self.route(sources, steps, hard=True)
        previous_mask = self.predicate_gate(bits[:, 2:7]).argmax(dim=-1)
        current_mask = self.predicate_gate(bits[:, 7:12]).argmax(dim=-1)
        value = bits[:, 0].long()
        for operand in (bits[:, 1].long(), previous_mask, current_mask):
            value = self.xor_gate(torch.stack((value, operand), dim=-1)).argmax(dim=-1)
        return {
            "state": value,
            "current_mask": current_mask,
            "selected_positions": selected,
        }

    @torch.no_grad()
    def hard_answer(
        self, encrypted_state: torch.Tensor, final_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.xor_gate(
            torch.stack((encrypted_state.long(), final_mask.long()), dim=-1)
        ).argmax(dim=-1)

    def soft_answer(
        self, encrypted_state: torch.Tensor, final_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.xor_gate(torch.stack((encrypted_state, final_mask), dim=-1))

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "seed_bits": self.seed_bits,
            "steps": self.steps,
            "heads": HEADS,
            "sources": self.sources,
            "key_dim": self.keys.shape[-1],
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "graph": self.graph,
        }
