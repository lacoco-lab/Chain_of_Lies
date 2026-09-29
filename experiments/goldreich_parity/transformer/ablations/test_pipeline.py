#!/usr/bin/env python3
"""Fast loss/path test for both final-standard supervision ablations."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from run_ablation import MODES, StandardGoldreichTransformer, losses


def main() -> None:
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    model = StandardGoldreichTransformer(
        length=4,
        seed_bits=5,
        graph=[[0, 1, 2, 3, 4]] * 4,
        model_dim=128,
        heads=16,
        token_dim=16,
        ff_dim=512,
    )
    batch = 3
    seeds = torch.randint(2, (batch, 5))
    inputs = torch.randint(2, (batch, 4))
    prefixes = torch.randint(2, (batch, 4))
    steps = torch.tensor([0, 2, 4])
    tokens, valid = model.build_training_sequence(seeds, inputs, prefixes, steps)
    targets = torch.randint(2, (batch,))
    previous = torch.randint(2, (batch,))
    current = torch.randint(2, (batch,))
    for mode in MODES:
        model.zero_grad(set_to_none=True)
        outputs = model(tokens, valid, steps)
        loss, pieces = losses(
            model,
            outputs,
            tokens,
            steps,
            targets,
            previous,
            current,
            mode,
            config,
            include_output=True,
        )
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert "output_ce" in pieces
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
    print("standard supervision-ablation pipeline tests passed")


if __name__ == "__main__":
    main()
