"""Fast bridge tests requiring no model download."""

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


generate_data = load_module("bridge_generate", "generate_data.py")
train_bridge = load_module("bridge_train", "train_bridge.py")
summarize_module = load_module("bridge_summary", "summarize.py")


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(character) + 1 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def test_generation_is_balanced_and_partitioned(tmp_path: Path) -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "local_replay_n": 128,
            "local_validation_n": 128,
            "supplied_start_train_n": 128,
            "supplied_start_validation_n": 128,
            "derived_start_train_n": 128,
            "derived_start_validation_n": 128,
            "supplied_update_train_n": 128,
            "supplied_update_validation_n": 128,
            "derived_update_train_n": 126,
            "derived_update_validation_n": 126,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    manifest = generate_data.generate(path, tmp_path / "data")
    assert manifest["all_seed_partitions_disjoint"]
    for name, n in manifest["dataset_sizes"].items():
        rows = train_bridge.read_jsonl(tmp_path / "data" / f"{name}.jsonl")
        assert len(rows) == n
        assert sum(row["supervised_suffix"] == "0" for row in rows) == n // 2


def test_derived_update_has_two_attention_groups_and_one_target_token() -> None:
    graph = generate_data.fixed_graph(16, 64, 20260818)
    row = generate_data.derived_update_rows(
        126,
        "validation",
        [format(i, "016b") for i in range(126)],
        graph,
        __import__("random").Random(0),
    )[0]
    prepared = train_bridge.prepare_row(CharacterTokenizer(), row)
    assert len(prepared["attention_groups"]) == 2
    assert len(prepared["seed_positions"]) == 16
    assert prepared["target_id"] == ord(row["supervised_suffix"]) + 1


def test_attention_loss_can_target_each_mask_separately() -> None:
    import torch

    selected = [1, 3, 5, 7, 9]
    logits = torch.full((2, 2, 1, 20), -4.0, requires_grad=True)
    with torch.no_grad():
        logits[:, :, 0, selected] = 4.0
    attentions = tuple(torch.softmax(logits, dim=-1) for _ in range(4))
    batch = {
        "query_positions": torch.tensor([0]),
        "rows": [
            {
                "attention_groups": [selected],
                "seed_positions": list(range(16)),
            }
        ],
    }
    loss, metrics = train_bridge.attention_auxiliary_loss(attentions, batch, 4, 4, 0.25)
    assert metrics["selected_attention_mass"] > 0.99
    assert metrics["selected_seed_ratio"] > 0.99
    loss.backward()
    assert logits.grad is not None


def test_summary_passes_successful_bridge(tmp_path: Path) -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    accuracy = {
        "n": 100,
        "greedy_accuracy": 0.99,
        "binary_accuracy": 0.99,
        "bit_ce": 0.01,
    }
    attention = {"selected_attention_mass": 0.99, "selected_seed_ratio": 0.99}
    final = {
        task: {
            "accuracy": accuracy,
            "attention": (
                attention
                if task in {"local", "derived_start", "derived_update"}
                else None
            ),
        }
        for task in (
            "local",
            "supplied_start",
            "derived_start",
            "supplied_update",
            "derived_update",
        )
    }
    training = {
        "baseline_local": accuracy,
        "final_evaluations": final,
        "elapsed_seconds": 1.0,
    }
    training_root, output_root = tmp_path / "training", tmp_path / "output"
    training_root.mkdir()
    (training_root / "training_metrics.json").write_text(
        json.dumps(training), encoding="utf-8"
    )
    result = summarize_module.summarize(config_path, training_root, output_root)
    assert result["passes_all_bridges"]
