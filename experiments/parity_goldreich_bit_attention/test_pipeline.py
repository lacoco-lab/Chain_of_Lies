"""Fast tests for the data and attention-loss mechanics (no model download)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_data = load_module("bit_attention_generate", "generate_data.py")
train_condition = load_module("bit_attention_train", "train_condition.py")


def test_generation_is_balanced_and_seed_disjoint(tmp_path: Path) -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config["train_n"], config["validation_n"] = 256, 128
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = generate_data.generate(config_path, tmp_path / "data")
    assert manifest["train_unique_seeds"] == 256
    assert manifest["validation_unique_seeds"] == 128
    assert manifest["seed_overlap"] == 0


def test_attention_loss_rewards_the_five_selected_positions() -> None:
    import torch

    selected = [1, 3, 5, 7, 9]
    logits = torch.full((2, 2, 1, 20), -4.0, requires_grad=True)
    with torch.no_grad():
        logits[:, :, 0, selected] = 4.0
    attentions = tuple(torch.softmax(logits, dim=-1) for _ in range(4))
    batch = {
        "query_positions": torch.tensor([0]),
        "rows": [{"seed_positions": list(range(16)), "selected_slots": selected}],
    }
    loss, metrics = train_condition.attention_auxiliary_loss(
        attentions, batch, layer_count=4, head_count=4, coverage_weight=0.25
    )
    assert torch.isfinite(loss)
    assert metrics["selected_attention_mass"] > 0.99
    assert metrics["selected_seed_ratio"] > 0.99
    loss.backward()
    assert logits.grad is not None
