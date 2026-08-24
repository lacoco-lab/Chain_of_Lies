"""Fast mechanics tests that require no model download."""

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


generate_data = load_module("full_attention_generate", "generate_data.py")
train_full = load_module("full_attention_train", "train_full.py")
summarize_module = load_module("full_attention_summarize", "summarize.py")


class CharacterTokenizer:
    """A boundary-transparent tokenizer for unit-testing character annotations."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [ord(character) + 1 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_generation_has_disjoint_partitions_and_correct_twins(tmp_path: Path) -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config.update({
        "core_lengths": [2], "extension_lengths": [4],
        "trace_train_n": 16, "trace_validation_n": 8,
        "local_replay_pool_n": 128, "local_validation_n": 128,
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = generate_data.generate(config_path, tmp_path / "data")
    assert manifest["all_seed_partitions_disjoint"]
    rows = train_full.read_jsonl(tmp_path / "data" / "trace_validation_4.jsonl")
    twins = {}
    for row in rows:
        twins.setdefault(row["spec"]["twin_id"], []).append(row["spec"]["gold_sequence"])
    assert all(len(values) == 2 and values[0] == values[1] for values in twins.values())


def test_target_offsets_separate_bits_answer_and_format() -> None:
    graph = generate_data.fixed_graph(16, 2, 20260818)
    row = generate_data._trace_records(2, 4, "validation", ["0101010101010101"],
                                       graph, __import__("random").Random(0))[0]
    prepared = train_full.prepare_row(CharacterTokenizer(), row)
    assert len(prepared["bit_indices"]) == 2
    assert prepared["answer_index"] is not None
    assert prepared["answer_index"] not in prepared["bit_indices"]
    assert prepared["format_indices"]
    assert len(prepared["attention_edges"]) == len(prepared["bit_indices"])


def test_attention_loss_rewards_selected_seed_values() -> None:
    import random
    import torch

    selected = [1, 3, 5, 7, 9]
    logits = torch.full((2, 2, 1, 20), -4.0, requires_grad=True)
    with torch.no_grad():
        logits[:, :, 0, selected] = 4.0
    attentions = tuple(torch.softmax(logits, dim=-1) for _ in range(4))
    batch = {"rows": [{
        "bit_indices": [1], "attention_edges": [selected],
        "seed_positions": list(range(16)),
    }]}
    loss, metrics = train_full.attention_auxiliary_loss(
        attentions, batch, 4, 4, 4, 0.25, random.Random(0)
    )
    assert metrics["selected_attention_mass"] > 0.99
    assert metrics["selected_seed_ratio"] > 0.99
    loss.backward()
    assert logits.grad is not None


def test_summary_applies_all_final_gates(tmp_path: Path) -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config["core_lengths"], config["extension_lengths"] = [2], [4]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    auto = {
        "n": 8, "trace_bit_accuracy": 0.99, "trace_exact": 0.98,
        "selector_accuracy": 1.0, "answer_accuracy": 1.0, "full_exact": 0.98,
        "parse_rate": 1.0, "twin_trace_match": 1.0, "seed_leak_rate": 0.0,
    }
    training = {
        "initial_adapter": "adapter", "baseline_local": {"bit_accuracy": 0.99},
        "final_local_retention": {"bit_accuracy": 0.99}, "elapsed_seconds": 1.0,
        "final_evaluations": {
            length: {"teacher": {"bit_accuracy": 1.0}, "autoregressive": auto}
            for length in ("2", "4")
        },
    }
    training_root, output_root = tmp_path / "training", tmp_path / "output"
    training_root.mkdir()
    (training_root / "training_metrics.json").write_text(json.dumps(training), encoding="utf-8")
    result = summarize_module.summarize(config_path, training_root, output_root)
    assert result["passes_core"] and result["passes_64_extension"]


def test_model_forward_falls_back_when_sparse_logits_are_unsupported() -> None:
    import torch
    from types import SimpleNamespace

    class OldQwen:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if "logits_to_keep" in kwargs:
                raise TypeError("Qwen2ForCausalLM.forward() got an unexpected keyword argument 'logits_to_keep'")
            return SimpleNamespace(logits=torch.zeros(1, 5, 7), attentions=None)

    model = OldQwen()
    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "logits_to_keep": 2,
    }
    outputs, offset = train_full.model_forward(model, batch, output_attentions=False)
    assert outputs.logits.shape == (1, 5, 7)
    assert offset == 0 and model.calls == 2
