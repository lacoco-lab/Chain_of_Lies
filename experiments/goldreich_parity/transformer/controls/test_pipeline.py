#!/usr/bin/env python3
"""Fast structural tests for the standard matched-control pipeline."""

from __future__ import annotations

import torch

from model import CONDITIONS, StandardParityControl


def main() -> None:
    torch.manual_seed(0)
    for length in (4, 32):
        model = StandardParityControl(
            length=length,
            seed_bits=12,
            graph=[[0, 1, 2, 3, 4] for _ in range(length)],
            model_dim=128,
            heads=16,
            token_dim=16,
            ff_dim=512,
        )
        inputs = torch.randint(2, (3, length))
        ordinary = inputs.cumsum(1).remainder(2)
        for condition in CONDITIONS:
            prefixes = (
                ordinary if condition == "normal_cot" else torch.zeros_like(ordinary)
            )
            steps = torch.tensor([0, length // 2, length])
            tokens, valid = model.control_sequence(inputs, prefixes, steps)
            outputs = model(tokens, valid, steps)
            assert outputs["output_logits"].shape == (3, 2)
            cache = model.control_cache(inputs)
            predicted = model.decode_step(cache, 0)["output_logits"].argmax(-1)
            model.append_prediction(cache, 0, predicted)
        assert model.control_targets(
            torch.tensor([0, length]), "normal_cot", {4, 8, 16}
        ).shape == (2, 16)
        direct = model.control_targets(torch.tensor([0]), "no_cot", {4, 8, 16})
        assert (direct is not None) == (length <= 16)
        if length == 32:
            staged_large = model.control_targets(
                torch.tensor([0, 32]), "no_cot", {32, 64}
            )
            assert staged_large is not None and staged_large.shape == (2, 16)
            expected = torch.arange(model.input_start, model.input_start + 16)
            assert torch.equal(staged_large[0].cpu(), expected)
            filler = model.control_targets(
                torch.tensor([0, 32]), "filler_cot", {32, 64}
            )
            assert filler is not None
            assert torch.equal(filler[0], torch.zeros(16, dtype=torch.long))
            assert torch.equal(filler[1].cpu(), expected)
    print("standard control pipeline tests passed")


if __name__ == "__main__":
    main()
