from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filler_gate.analysis import (
    compute_metrics,
    exact_mcnemar_p,
    paired_bootstrap_delta,
)
from filler_gate.data import (
    dataset_summary,
    group_demonstrations,
    load_benchmark,
    load_components,
    load_demonstrations,
    validate_cross_references,
    validate_gold_semantics,
)
from filler_gate.prompting import (
    FILLER_MARKER,
    make_main_spec,
    prepare_prompt,
    resolve_atomic_filler_token,
    validate_non_filler_invariance,
)
from filler_gate.scoring import score_generation
from filler_gate.reporting import build_report


class FakeTokenizer:
    name_or_path = "fake"
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages) == 1
        return f"<user>{messages[0]['content']}</user><assistant>"


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = load_benchmark(ROOT / "data" / "benchmark.jsonl")
        cls.components = load_components(ROOT / "data" / "component_facts.jsonl")
        cls.demonstrations = load_demonstrations(
            ROOT / "data" / "demonstrations.jsonl"
        )

    def test_dataset_counts_and_cross_references(self):
        validate_cross_references(self.items, self.components)
        validate_gold_semantics(self.items, self.components)
        summary = dataset_summary(
            self.items,
            self.components,
            self.demonstrations,
        )
        self.assertEqual(summary["benchmark_items"], 40)
        self.assertEqual(summary["component_facts"], 40)
        self.assertEqual(summary["demonstrations"], 20)
        self.assertEqual(set(summary["benchmark_by_task"].values()), {10})

    def test_five_disjoint_demonstrations_per_task(self):
        grouped = group_demonstrations(self.demonstrations, per_task=5)
        benchmark_questions = {item.question for item in self.items}
        for demos in grouped.values():
            self.assertEqual(len(demos), 5)
            self.assertFalse(
                benchmark_questions & {demonstration.question for demonstration in demos}
            )


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        items = load_benchmark(ROOT / "data" / "benchmark.jsonl")
        demos = load_demonstrations(ROOT / "data" / "demonstrations.jsonl")
        grouped = group_demonstrations(demos, per_task=5)
        self.item = items[0]
        self.demos = grouped[self.item.task_type]

    def test_exact_token_filler_insertion(self):
        token_id, token_text = resolve_atomic_filler_token(
            self.tokenizer,
            [" .", "."],
        )
        self.assertEqual(token_text, ".")
        spec = make_main_spec(self.item, self.demos, filler_length=8)
        prepared = prepare_prompt(
            self.tokenizer,
            spec,
            filler_token_id=token_id,
            filler_token_text=token_text,
        )
        self.assertEqual(prepared.filler_occurrences, 6)
        self.assertEqual(
            len(prepared.input_ids),
            len(
                self.tokenizer.encode(
                    prepared.serialized_prompt_with_markers.replace(
                        FILLER_MARKER,
                        "",
                    )
                )
            )
            + 48,
        )

    def test_non_filler_scaffold_is_invariant(self):
        token_id, token_text = resolve_atomic_filler_token(
            self.tokenizer,
            ["."],
        )
        specs = [
            make_main_spec(self.item, self.demos, filler_length=value)
            for value in (0, 8, 16)
        ]
        validate_non_filler_invariance(
            self.tokenizer,
            specs,
            filler_token_id=token_id,
            filler_token_text=token_text,
        )


class ScoringTests(unittest.TestCase):
    def test_integer_and_string_scoring(self):
        self.assertTrue(score_generation("The answer is 1,234.", "1234", "integer")["correct"])
        self.assertTrue(score_generation("**H**", "h", "string")["correct"])
        self.assertFalse(score_generation("", "h", "string")["format_correct"])

    def test_mcnemar_and_bootstrap(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertLess(exact_mcnemar_p(10, 0), 0.01)
        low, high = paired_bootstrap_delta(
            [False] * 10,
            [True] * 10,
            samples=200,
            confidence_level=0.95,
            seed=1,
        )
        self.assertEqual((low, high), (1.0, 1.0))


class AnalysisIntegrationTests(unittest.TestCase):
    def test_metrics_and_markdown_report(self):
        items = load_benchmark(ROOT / "data" / "benchmark.jsonl")
        components = load_components(ROOT / "data" / "component_facts.jsonl")
        predictions = []
        for component in components:
            predictions.append(
                {
                    "prediction_key": f"component::{component.component_id}::k0",
                    "record_kind": "component",
                    "item_id": component.component_id,
                    "task_type": component.component_type,
                    "filler_length": 0,
                    "correct": True,
                    "format_correct": True,
                }
            )
        for item in items:
            for filler_length in (0, 8):
                predictions.append(
                    {
                        "prediction_key": f"main::{item.item_id}::k{filler_length}",
                        "record_kind": "main",
                        "item_id": item.item_id,
                        "task_type": item.task_type,
                        "filler_length": filler_length,
                        "correct": filler_length == 8,
                        "format_correct": True,
                    }
                )
        metrics = compute_metrics(
            predictions,
            items,
            components,
            analysis_config={
                "bootstrap_samples": 200,
                "confidence_level": 0.95,
                "knowledge_component_accuracy_min": 0.8,
                "knowledge_per_group_accuracy_min": 0.7,
                "filler_primary_length": 8,
                "filler_alpha": 0.05,
            },
            seed=1,
        )
        self.assertEqual(
            metrics["knowledge_screening"]["configured_gate"]["status"],
            "pass",
        )
        self.assertEqual(
            metrics["filler_gate"]["configured_gate"]["status"],
            "pass",
        )
        report = build_report(
            metrics,
            run_manifest={
                "model_id": "fake",
                "seed": 1,
                "filler_token_id": 46,
                "filler_token_text": ".",
                "dataset_summary": {
                    "benchmark_items": 40,
                    "component_facts": 40,
                },
            },
            plot_available=False,
        )
        self.assertIn("Overall main-task baseline", report)
        self.assertIn("Wrong→right", report)


if __name__ == "__main__":
    unittest.main()
