"""Fast tests for the purpose-built Goldreich transformer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # The Condor submit host intentionally has no PyTorch.
    torch = None


HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_data import (
    encrypted_trajectory,
    fixed_graph,
    generate,
    predicate,  # noqa: E402
    seed_bits_for_length,
    split_seed_values,
)
from validate_data import validate  # noqa: E402

if torch is not None:
    from evaluate_softmax import (
        evaluate_softmax,
        validate_supplied_config,
    )  # noqa: E402
    from model import (
        CONSTANT_POSITION,
        SEED_START,
        STATE_POSITION,  # noqa: E402
        GoldreichToyTransformer,
    )
    from train import evaluate, supervision_settings  # noqa: E402
    from standard_model import StandardGoldreichTransformer  # noqa: E402
    from train_standard import curriculum_selection  # noqa: E402
    from train_standard_v2 import (
        evaluate_public_mechanism,  # noqa: E402
        public_targets,
        run_stage,
    )


class ToyTransformerTests(unittest.TestCase):
    def test_worked_formula(self) -> None:
        seed = list(map(int, "0101011011001011"))
        bits = list(map(int, "10110110"))
        graph = [
            [10, 1, 9, 7, 3],
            [6, 9, 8, 10, 4],
            [15, 13, 11, 12, 5],
            [7, 8, 4, 3, 9],
        ] * 2
        masks, states, answer = encrypted_trajectory(bits, seed, graph)
        self.assertEqual(masks, [0, 1, 0, 0, 0, 1, 0, 0])
        self.assertEqual(states, [1, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(answer, 1)
        self.assertEqual(predicate([1, 1, 1, 0, 0]), 1)

    def test_data_generation_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
            config.update(
                {
                    "lengths": [8],
                    "seed_bits_by_length": {"8": 5},
                    "key_dim_by_length": {"8": 8},
                    "train_n_per_length": 64,
                    "validation_n_per_length": 32,
                    "test_n_per_length": 32,
                }
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            data = root / "data"
            generate(config_path, data)
            report = validate(config_path, data)
            self.assertTrue(report["seed_partitions_disjoint"])
            self.assertEqual(report["n8/test"]["n"], 32)
            # Independently transferred generation jobs keep their manifest
            # inside the per-length directory rather than at the common root.
            (data / "manifest.json").replace(data / "n8" / "manifest.json")
            per_length_report = validate(config_path, data)
            self.assertTrue(per_length_report["seed_partitions_disjoint"])
            self.assertEqual(per_length_report["n8/test"]["n"], 32)

    def test_scaled_seed_lengths_and_large_seed_sampling(self) -> None:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [seed_bits_for_length(config, n) for n in (32, 64, 256, 512)],
            [12, 19, 49, 79],
        )

    def test_easy_encrypted_seed_lengths_respect_five_local_predicate(self) -> None:
        config = json.loads(
            (HERE / "encrypted_easy_data_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [seed_bits_for_length(config, n) for n in (4, 8, 16)],
            [5, 5, 7],
        )
        small = dict(config)
        small.update({"train_seed_n": 10, "validation_seed_n": 4, "test_seed_n": 4})
        splits = split_seed_values(79, small, 512)
        self.assertEqual(
            {key: len(value) for key, value in splits.items()},
            {"train": 10, "validation": 4, "test": 4},
        )
        self.assertEqual(len(set().union(*map(set, splits.values()))), 18)
        scale = json.loads((HERE / "scale_config.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [seed_bits_for_length(scale, n) for n in (1024, 2048)], [128, 208]
        )

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_routing_targets_match_public_graph(self) -> None:
        graph = fixed_graph(5, 8, 20260818)
        model = GoldreichToyTransformer(8, 5, graph, key_dim=8, gate_hidden_dim=16)
        targets = model.routing_targets(torch.tensor([0, 1]))
        self.assertEqual(targets.shape, (2, 12))
        self.assertEqual(int(targets[0, 0]), STATE_POSITION)
        self.assertEqual(int(targets[0, 1]), model.input_start)
        self.assertEqual(targets[0, 2:7].tolist(), [CONSTANT_POSITION] * 5)
        self.assertEqual(
            targets[0, 7:12].tolist(), [SEED_START + value for value in graph[0]]
        )
        self.assertEqual(
            targets[1, 2:7].tolist(), [SEED_START + value for value in graph[0]]
        )
        self.assertEqual(
            targets[1, 7:12].tolist(), [SEED_START + value for value in graph[1]]
        )

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_source_layout_contains_only_private_bits_and_state(self) -> None:
        graph = fixed_graph(5, 8, 20260818)
        model = GoldreichToyTransformer(8, 5, graph, key_dim=8, gate_hidden_dim=16)
        seed = torch.tensor([[int(value) for value in "01010"]])
        bits = torch.tensor([[int(value) for value in "10110110"]])
        sources = model.build_sources(seed, bits, torch.tensor([1]))
        self.assertEqual(sources.shape, (1, model.input_start + 8))
        self.assertEqual(sources[0, CONSTANT_POSITION], 0)
        self.assertEqual(sources[0, STATE_POSITION], 1)
        self.assertTrue(
            torch.equal(sources[0, SEED_START : model.input_start], seed[0])
        )
        self.assertTrue(torch.equal(sources[0, model.input_start :], bits[0]))

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_vectorized_softmax_evaluation_matches_reference(self) -> None:
        graph = fixed_graph(5, 8, 20260818)
        model = GoldreichToyTransformer(8, 5, graph, key_dim=8, gate_hidden_dim=16)
        seed_rows = [list(map(int, value)) for value in ("01010", "10101", "11100")]
        input_rows = [
            list(map(int, value))
            for value in (
                "10110110",
                "00101101",
                "11110000",
            )
        ]
        mask_rows, state_rows, answers = [], [], []
        for bits, seed in zip(input_rows, seed_rows):
            masks, states, answer = encrypted_trajectory(bits, seed, graph)
            mask_rows.append(masks)
            state_rows.append(states)
            answers.append(answer)
        data = {
            "ids": ["a", "b", "c"],
            "seeds": torch.tensor(seed_rows),
            "inputs": torch.tensor(input_rows),
            "masks": torch.tensor(mask_rows),
            "states": torch.tensor(state_rows),
            "answers": torch.tensor(answers),
        }
        reference = evaluate(model, data, torch.device("cpu"), batch_size=2, hard=False)
        vectorized = evaluate_softmax(
            model,
            data,
            torch.device("cpu"),
            batch_size=2,
        )
        for key in (
            "state_bit_accuracy",
            "trace_exact",
            "final_parity_accuracy",
            "joint_exact",
            "per_step_accuracy",
        ):
            self.assertEqual(reference[key], vectorized[key])

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_eval_accepts_new_training_only_config_defaults(self) -> None:
        checkpoint_config = json.loads(
            (HERE / "config.json").read_text(encoding="utf-8")
        )
        checkpoint_config.pop("answer_loss_weight")
        checkpoint_config.pop("entropy_loss_weight")
        supplied = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        validate_supplied_config(checkpoint_config, supplied, 32)
        supplied["graph_seed"] += 1
        with self.assertRaises(RuntimeError):
            validate_supplied_config(checkpoint_config, supplied, 32)

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_standard_cached_decoding_matches_full_query_row(self) -> None:
        graph = fixed_graph(5, 8, 20260818)
        model = StandardGoldreichTransformer(
            length=8,
            seed_bits=5,
            graph=graph,
            model_dim=128,
            heads=16,
            token_dim=16,
            ff_dim=512,
        ).eval()
        seeds = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]])
        inputs = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1]])
        prefixes = torch.tensor([[1, 1, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 1]])
        cache = model.initialize_decode_cache(seeds, inputs)
        for step in range(9):
            steps = torch.full((2,), step, dtype=torch.long)
            token_ids, valid = model.build_training_sequence(
                seeds, inputs, prefixes, steps
            )
            direct = model(token_ids, valid, steps)
            cached = model.decode_step(cache, step)
            for key in (
                "output_logits",
                "previous_mask_logits",
                "current_mask_logits",
                "attention",
            ):
                self.assertTrue(torch.allclose(direct[key], cached[key], atol=1e-6))
            if step < 8:
                model.append_prediction(cache, step, prefixes[:, step])

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_standard_model_has_single_ffn_and_no_special_boolean_gates(self) -> None:
        graph = fixed_graph(5, 8, 20260818)
        model = StandardGoldreichTransformer(
            length=8,
            seed_bits=5,
            graph=graph,
            model_dim=256,
            heads=16,
            token_dim=16,
            ff_dim=1024,
        )
        self.assertFalse(
            any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
        )
        self.assertFalse(hasattr(model, "predicate_gate"))
        self.assertFalse(hasattr(model, "xor_gate"))
        self.assertEqual(model.ff_dim, 4 * model.model_dim)
        self.assertEqual(model.position_dim + model.token_dim, model.model_dim)

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_standard_curriculum_selects_the_requested_real_examples(self) -> None:
        previous = torch.tensor([0, 0, 1, 1])
        current = torch.tensor([0, 1, 0, 1])
        self.assertEqual(
            curriculum_selection(previous, current, "both_masks_zero").tolist(),
            [True, False, False, False],
        )
        self.assertEqual(
            curriculum_selection(previous, current, "current_mask_zero").tolist(),
            [True, False, True, False],
        )

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_standard_v2_public_targets_and_evaluator_execute(self) -> None:
        bits = torch.tensor([[0] * 12, [1] * 12])
        previous, current, delta, output = public_targets(bits)
        self.assertEqual(previous.tolist(), [0, 0])
        self.assertEqual(current.tolist(), [0, 0])
        self.assertEqual(delta.tolist(), [0, 0])
        self.assertEqual(output.tolist(), [0, 0])
        graph = fixed_graph(5, 8, 20260818)
        model = StandardGoldreichTransformer(
            length=8,
            seed_bits=5,
            graph=graph,
            model_dim=128,
            heads=16,
            token_dim=16,
            ff_dim=512,
        )
        metrics = evaluate_public_mechanism(
            model, repeats=1, batch_size=512, device=torch.device("cpu")
        )
        self.assertEqual(metrics["examples"], 4096)

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_standard_v2_can_record_a_nonfatal_stage_gate(self) -> None:
        def failed_stage():
            raise RuntimeError("diagnostic threshold missed")

        result, warning = run_stage(
            "representation", failed_stage, continue_after_gate_failure=True
        )
        self.assertFalse(result["passed_stage_gate"])
        self.assertTrue(result["continued_from_best_stage_state"])
        self.assertEqual(warning, "diagnostic threshold missed")
        with self.assertRaises(RuntimeError):
            run_stage(
                "representation",
                failed_stage,
                continue_after_gate_failure=False,
            )

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_length_2048_layout_is_well_formed(self) -> None:
        graph = fixed_graph(208, 2048, 20260818)
        model = GoldreichToyTransformer(
            2048, 208, graph, key_dim=64, gate_hidden_dim=128
        )
        targets = model.routing_targets(torch.tensor([0, 2047]))
        self.assertEqual(targets.shape, (2, 12))
        self.assertGreaterEqual(int(targets.min()), 0)
        self.assertLess(int(targets.max()), model.sources)
        self.assertEqual(
            model.architecture_metadata()["trainable_parameters"], 3_308_676
        )

    @unittest.skipIf(torch is None, "PyTorch is available inside the Condor job")
    def test_reduced_supervision_settings_remove_position_labels(self) -> None:
        config = json.loads((HERE / "ablation_config.json").read_text(encoding="utf-8"))
        mask_state = supervision_settings(config, "mask_state")
        cot_only = supervision_settings(config, "cot_answer_only")
        self.assertFalse(mask_state["attention_supervision"])
        self.assertEqual(mask_state["attention_loss_weight"], 0.0)
        self.assertGreater(mask_state["mask_loss_weight"], 0.0)
        self.assertFalse(cot_only["attention_supervision"])
        self.assertEqual(cot_only["attention_loss_weight"], 0.0)
        self.assertEqual(cot_only["mask_loss_weight"], 0.0)
        self.assertGreater(cot_only["answer_loss_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
