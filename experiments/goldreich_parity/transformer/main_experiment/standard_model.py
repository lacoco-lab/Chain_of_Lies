"""Standard one-layer causal Transformer for Goldreich-encrypted PARITY.

The implementation computes only the final query row of causal self-attention. For one layer,
this is exactly the same output as constructing the full causal attention matrix and discarding
all rows except the one used to predict the next token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ZERO_TOKEN = 0
ONE_TOKEN = 1
QUERY_TOKEN = 2
PAD_TOKEN = 3
VOCAB_SIZE = 4
COMPUTATION_HEADS = 12
CANONICAL_BIT_SCALE = 4.0


@dataclass
class DecodeCache:
    keys: torch.Tensor
    values: torch.Tensor
    static_length: int


class StandardGoldreichTransformer(nn.Module):
    """One standard softmax-attention block and one residual ReLU FFN."""

    def __init__(
        self,
        *,
        length: int,
        seed_bits: int,
        graph: list[list[int]],
        model_dim: int,
        heads: int,
        token_dim: int,
        ff_dim: int,
    ) -> None:
        super().__init__()
        if length < 1 or seed_bits < 5 or len(graph) < length:
            raise ValueError(
                "Length, seed width, and one public edge per step are required"
            )
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if heads < COMPUTATION_HEADS:
            raise ValueError(f"At least {COMPUTATION_HEADS} heads are required")
        if not 0 < token_dim < model_dim:
            raise ValueError("token_dim must lie strictly between zero and model_dim")

        self.length = length
        self.seed_bits = seed_bits
        self.graph = [list(edge) for edge in graph[:length]]
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.token_dim = token_dim
        self.position_dim = model_dim - token_dim
        self.ff_dim = ff_dim

        # Sequence: [public zero, private seed, private input, generated encrypted prefix, query].
        self.seed_start = 1
        self.input_start = self.seed_start + seed_bits
        self.cot_start = self.input_start + length
        self.max_positions = self.cot_start + length + 1

        self.token_embedding = nn.Embedding(VOCAB_SIZE, token_dim)
        self.position_embedding = nn.Embedding(self.max_positions, self.position_dim)
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        # The only nonlinear block after attention. No layer normalization is used.
        self.ff_in = nn.Linear(model_dim, ff_dim)
        self.ff_out = nn.Linear(ff_dim, model_dim)
        self.output_head = nn.Linear(model_dim, 2)
        self.previous_mask_head = nn.Linear(model_dim, 2)
        self.current_mask_head = nn.Linear(model_dim, 2)
        self.mask_delta_head = nn.Linear(model_dim, 2)
        # Training-only probes ensure each routed value retains its selected source bit.
        self.value_probe_weight = nn.Parameter(
            torch.empty(COMPUTATION_HEADS, self.head_dim, 2)
        )
        self.value_probe_bias = nn.Parameter(torch.zeros(COMPUTATION_HEADS, 2))
        nn.init.xavier_uniform_(self.value_probe_weight)

    def _represent(
        self, token_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (self.token_embedding(token_ids), self.position_embedding(position_ids)),
            dim=-1,
        )

    def _project_sequence(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply standard linear K/V projections using an exact factored calculation."""
        batch, sequence = token_ids.shape
        positions = torch.arange(sequence, device=token_ids.device)
        token_part = self.token_embedding(token_ids)
        position_part = self.position_embedding(positions)

        def project(layer: nn.Linear) -> torch.Tensor:
            # Linear(concat(token, position)) is exactly the sum of these two terms.
            token_value = F.linear(token_part, layer.weight[:, : self.token_dim], None)
            position_value = F.linear(
                position_part, layer.weight[:, self.token_dim :], layer.bias
            )
            value = token_value + position_value.unsqueeze(0)
            return value.reshape(batch, sequence, self.heads, self.head_dim)

        return project(self.k_proj), project(self.v_proj)

    def _project_query(
        self, query_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.full_like(query_positions, QUERY_TOKEN)
        representation = self._represent(tokens, query_positions)
        batch = len(query_positions)
        query = self.q_proj(representation).reshape(batch, self.heads, self.head_dim)
        key = self.k_proj(representation).reshape(batch, self.heads, self.head_dim)
        value = self.v_proj(representation).reshape(batch, self.heads, self.head_dim)
        return representation, query, key, value

    def _finish_block(
        self, query_representation: torch.Tensor, attended: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        attention_residual = query_representation + self.out_proj(
            attended.reshape(len(attended), self.model_dim)
        )
        hidden = attention_residual + self.ff_out(
            F.relu(self.ff_in(attention_residual))
        )
        return {
            "output_logits": self.output_head(hidden),
            "previous_mask_logits": self.previous_mask_head(hidden),
            "current_mask_logits": self.current_mask_head(hidden),
            "mask_delta_logits": self.mask_delta_head(hidden),
            "routed_value_logits": (
                torch.einsum(
                    "bhd,hdc->bhc",
                    attended[:, :COMPUTATION_HEADS],
                    self.value_probe_weight,
                )
                + self.value_probe_bias.unsqueeze(0)
            ),
            "attention_residual": attention_residual,
            "hidden": hidden,
        }

    def canonical_residual(
        self, bits: torch.Tensor, steps: torch.Tensor
    ) -> torch.Tensor:
        """Training target exposing the 12 routed bits in fixed residual coordinates."""
        query_positions = self.cot_start + steps
        target, _, _, _ = self._project_query(query_positions)
        target = target.detach().clone()
        target[:, :COMPUTATION_HEADS] = (
            bits.float().mul(2.0).sub(1.0).mul(CANONICAL_BIT_SCALE)
        )
        return target

    def finish_from_residual(
        self, attention_residual: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Training-only entry point for supervising the public FFN truth table."""
        hidden = attention_residual + self.ff_out(
            F.relu(self.ff_in(attention_residual))
        )
        return {
            "output_logits": self.output_head(hidden),
            "previous_mask_logits": self.previous_mask_head(hidden),
            "current_mask_logits": self.current_mask_head(hidden),
            "mask_delta_logits": self.mask_delta_head(hidden),
            "hidden": hidden,
        }

    def computation_targets(self, steps: torch.Tensor) -> torch.Tensor:
        rows = []
        for raw_step in steps.detach().cpu().tolist():
            step = int(raw_step)
            if not 0 <= step <= self.length:
                raise ValueError(f"Step {step} is outside [0, {self.length}]")
            previous_state = 0 if step == 0 else self.cot_start + step - 1
            current_input = self.input_start + step if step < self.length else 0
            previous_mask = (
                [0] * 5
                if step == 0
                else [self.seed_start + index for index in self.graph[step - 1]]
            )
            current_mask = (
                [self.seed_start + index for index in self.graph[step]]
                if step < self.length
                else [0] * 5
            )
            rows.append([previous_state, current_input, *previous_mask, *current_mask])
        return torch.tensor(rows, dtype=torch.long, device=steps.device)

    def routing_targets(self, steps: torch.Tensor) -> torch.Tensor:
        computation = self.computation_targets(steps)
        if self.heads == COMPUTATION_HEADS:
            return computation
        padding = torch.zeros(
            (len(steps), self.heads - COMPUTATION_HEADS),
            dtype=torch.long,
            device=steps.device,
        )
        return torch.cat((computation, padding), dim=1)

    def synthesized_mechanism(
        self, bits: torch.Tensor, steps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the real V/W_O/residual/FFN path on supervised local Boolean inputs.

        This is used only for public-mechanism pretraining. It does not create an inference path.
        """
        if bits.shape != (len(steps), COMPUTATION_HEADS):
            raise ValueError(f"Expected bits with shape [batch, {COMPUTATION_HEADS}]")
        positions = self.routing_targets(steps)
        all_bits = torch.zeros(
            (len(steps), self.heads), dtype=torch.long, device=bits.device
        )
        all_bits[:, :COMPUTATION_HEADS] = bits.long()
        representations = self._represent(all_bits, positions)
        projected = self.v_proj(representations).reshape(
            len(steps), self.heads, self.heads, self.head_dim
        )
        head_indices = torch.arange(self.heads, device=bits.device)
        attended = projected[:, head_indices, head_indices]
        query_positions = self.cot_start + steps
        query_representation, _, _, _ = self._project_query(query_positions)
        return self._finish_block(query_representation, attended)

    def build_training_sequence(
        self,
        seeds: torch.Tensor,
        inputs: torch.Tensor,
        state_prefixes: torch.Tensor,
        steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create padded causal prefixes ending in a query token at each requested step."""
        batch = len(steps)
        maximum_step = int(steps.max())
        sequence = self.cot_start + maximum_step + 1
        token_ids = torch.full(
            (batch, sequence), PAD_TOKEN, dtype=torch.long, device=seeds.device
        )
        token_ids[:, 0] = ZERO_TOKEN
        token_ids[:, self.seed_start : self.input_start] = seeds.long()
        token_ids[:, self.input_start : self.cot_start] = inputs.long()
        valid = torch.zeros((batch, sequence), dtype=torch.bool, device=seeds.device)
        for row, raw_step in enumerate(steps.detach().cpu().tolist()):
            step = int(raw_step)
            if step:
                token_ids[row, self.cot_start : self.cot_start + step] = state_prefixes[
                    row, :step
                ]
            query_position = self.cot_start + step
            token_ids[row, query_position] = QUERY_TOKEN
            valid[row, : query_position + 1] = True
        return token_ids, valid

    def forward(
        self, token_ids: torch.Tensor, valid: torch.Tensor, steps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return the final query row of ordinary causal softmax self-attention."""
        keys, values = self._project_sequence(token_ids)
        query_positions = self.cot_start + steps
        query_representation, queries, _, _ = self._project_query(query_positions)
        scores = torch.einsum("bhd,blhd->bhl", queries, keys) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        probabilities = scores.softmax(dim=-1)
        attended = torch.einsum("bhl,blhd->bhd", probabilities, values)
        result = self._finish_block(query_representation, attended)
        result.update(
            {
                "attention_logits": scores,
                "attention": probabilities,
                "attended": attended,
            }
        )
        return result

    @torch.no_grad()
    def initialize_decode_cache(
        self, seeds: torch.Tensor, inputs: torch.Tensor
    ) -> DecodeCache:
        batch = len(seeds)
        static_tokens = torch.empty(
            (batch, self.cot_start), dtype=torch.long, device=seeds.device
        )
        static_tokens[:, 0] = ZERO_TOKEN
        static_tokens[:, self.seed_start : self.input_start] = seeds.long()
        static_tokens[:, self.input_start : self.cot_start] = inputs.long()
        static_keys, static_values = self._project_sequence(static_tokens)
        shape = (batch, self.max_positions, self.heads, self.head_dim)
        keys = torch.empty(shape, dtype=static_keys.dtype, device=seeds.device)
        values = torch.empty(shape, dtype=static_values.dtype, device=seeds.device)
        keys[:, : self.cot_start] = static_keys
        values[:, : self.cot_start] = static_values
        return DecodeCache(keys=keys, values=values, static_length=self.cot_start)

    @torch.no_grad()
    def decode_step(self, cache: DecodeCache, step: int) -> dict[str, torch.Tensor]:
        """Greedily decode one token with standard softmax attention and cached K/V."""
        batch = cache.keys.shape[0]
        query_position = self.cot_start + step
        positions = torch.full(
            (batch,), query_position, dtype=torch.long, device=cache.keys.device
        )
        query_representation, queries, query_keys, query_values = self._project_query(
            positions
        )
        prefix_keys = cache.keys[:, :query_position]
        prefix_values = cache.values[:, :query_position]
        prefix_scores = torch.einsum("bhd,blhd->bhl", queries, prefix_keys) / math.sqrt(
            self.head_dim
        )
        self_scores = (queries * query_keys).sum(-1, keepdim=True) / math.sqrt(
            self.head_dim
        )
        scores = torch.cat((prefix_scores, self_scores), dim=-1)
        probabilities = scores.softmax(dim=-1)
        prefix_attention = torch.einsum(
            "bhl,blhd->bhd", probabilities[:, :, :-1], prefix_values
        )
        attended = prefix_attention + probabilities[:, :, -1:] * query_values
        result = self._finish_block(query_representation, attended)
        result.update(
            {
                "attention_logits": scores,
                "attention": probabilities,
                "attended": attended,
            }
        )
        return result

    @torch.no_grad()
    def append_prediction(
        self, cache: DecodeCache, step: int, token_ids: torch.Tensor
    ) -> None:
        position = self.cot_start + step
        token_matrix = token_ids.long().reshape(-1, 1)
        token_part = self.token_embedding(token_matrix)
        position_part = self.position_embedding.weight[position : position + 1]

        def project(layer: nn.Linear) -> torch.Tensor:
            token_value = F.linear(token_part, layer.weight[:, : self.token_dim], None)
            position_value = F.linear(
                position_part, layer.weight[:, self.token_dim :], layer.bias
            )
            return (token_value + position_value.unsqueeze(0)).reshape(
                len(token_ids), self.heads, self.head_dim
            )

        cache.keys[:, position] = project(self.k_proj)
        cache.values[:, position] = project(self.v_proj)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "one_layer_standard_causal_transformer_query_row",
            "length": self.length,
            "seed_bits": self.seed_bits,
            "heads": self.heads,
            "computationally_assigned_heads": COMPUTATION_HEADS,
            "zero_padding_heads": self.heads - COMPUTATION_HEADS,
            "model_dim": self.model_dim,
            "head_dim": self.head_dim,
            "token_embedding_dim": self.token_dim,
            "position_embedding_dim": self.position_dim,
            "position_encoding": "learned absolute, concatenated with token embeddings",
            "ff_dim": self.ff_dim,
            "attention": "standard scaled dot-product softmax at train and inference",
            "layer_norm": False,
            "nonlinear_blocks_after_attention": 1,
            "training_only_canonical_bit_scale": CANONICAL_BIT_SCALE,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "graph": self.graph,
        }
