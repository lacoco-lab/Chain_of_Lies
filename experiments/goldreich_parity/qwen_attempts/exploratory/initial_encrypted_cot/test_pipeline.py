from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_splits import (
    encrypted_trace,
    final_parity,
    fixed_hypergraph,
    generate,
    goldreich_masks,
    goldreich_predicate,
    pair_cumulative_trace,
    supervised_suffix,
)
from summarize import parse_completion, summarize
from validate_splits import validate


class ParityGoldreichTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = json.loads(
            Path(
                "experiments/goldreich_parity/qwen_attempts/exploratory/initial_encrypted_cot/config.json"
            ).read_text(encoding="utf-8")
        )
        config["train_n"] = 40
        config["validation_n"] = 20
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_goldreich_predicate_and_concrete_encryption(self) -> None:
        seed = "10011010"
        graph = [[0, 1, 2, 3, 4], [0, 3, 4, 1, 2]]
        self.assertEqual(goldreich_predicate([1, 0, 0, 1, 1]), 0)
        self.assertEqual(goldreich_masks(seed, graph), [0, 1])
        self.assertEqual(pair_cumulative_trace("1011"), [1, 1])
        self.assertEqual(encrypted_trace("1011", [0, 1]), [1, 0])
        self.assertEqual(final_parity("1011"), 1)

    def test_fixed_graph_is_reproducible_and_five_local(self) -> None:
        first = fixed_hypergraph(seed_bits=16, mask_bits=64, graph_seed=20260818)
        second = fixed_hypergraph(seed_bits=16, mask_bits=64, graph_seed=20260818)
        different = fixed_hypergraph(seed_bits=16, mask_bits=64, graph_seed=20260819)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(len(edge) == len(set(edge)) == 5 for edge in first))

    def test_config_uses_requested_stretch(self) -> None:
        config = json.loads(
            Path(
                "experiments/goldreich_parity/qwen_attempts/exploratory/initial_encrypted_cot/config.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["mask_bits"], int(config["seed_bits"] ** 1.5))
        self.assertEqual(config["input_bits"], 2 * config["mask_bits"])
        epochs = {
            item["name"]: item["training_epochs"] for item in config["conditions"]
        }
        self.assertEqual(epochs["plaintext_pair_cot"], 3)
        self.assertEqual(epochs["provided_mask_encrypted_cot"], 6)
        self.assertEqual(epochs["goldreich_encrypted_cot"], 6)

    def test_generation_is_reproducible_and_validation_seeds_are_unseen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(
                first["validation_items_sha256"], second["validation_items_sha256"]
            )
            self.assertEqual(
                first["fixed_hypergraph_sha256"], second["fixed_hypergraph_sha256"]
            )
            self.assertEqual(first["train_validation_seed_overlap"], 0)
            report = validate(config, root / "first")
            self.assertEqual(report["train_n"], 40)
            self.assertEqual(report["validation_n"], 20)
            self.assertEqual(report["train_unique_prg_seeds"], 20)
            self.assertEqual(report["validation_unique_prg_seeds"], 10)
            self.assertEqual(report["train_validation_seed_overlap"], 0)
            self.assertTrue(report["conditions_share_identical_items"])
            self.assertTrue(report["fixed_graph"])
            self.assertTrue(report["no_private_task"])

    def test_all_targets_parse_strictly(self) -> None:
        bits = "10" * 64
        masks = [index % 2 for index in range(64)]
        for protocol in (
            "pair_cumulative",
            "provided_mask_encrypted",
            "goldreich_encrypted",
        ):
            target = supervised_suffix(bits, protocol, masks=masks)
            parsed = parse_completion(target)
            self.assertTrue(parsed["strict_format"])
            self.assertEqual(parsed["answer"], final_parity(bits))
            self.assertEqual(len(parsed["trace"]), 64)

    def test_synthetic_perfect_outputs_pass_mechanism_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            split_root = root / "splits"
            responses_root = root / "responses"
            artifacts_root = root / "artifacts"
            generate(config_path, split_root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for condition in config["conditions"]:
                name = condition["name"]
                prompts = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(
                        (split_root / "seed_0" / name / "val_prompts").glob("*.json")
                    )
                ]
                for source in ("baseline", "finetuned"):
                    directory = responses_root / "seed_0" / name / source
                    directory.mkdir(parents=True)
                    for prompt in prompts:
                        (directory / f"{prompt['experiment_id']}.json").write_text(
                            json.dumps({"raw_text": prompt["supervised_suffix"]}),
                            encoding="utf-8",
                        )
            summary = summarize(
                config_path=config_path,
                split_root=split_root,
                responses_root=responses_root,
                artifacts_root=artifacts_root,
            )
            self.assertTrue(summary["passes_mechanism_gate"])
            self.assertEqual(summary["train_validation_seed_overlap"], 0)
            encrypted = next(
                row
                for row in summary["metrics"]
                if row["condition"] == "goldreich_encrypted_cot"
                and row["source"] == "finetuned"
            )
            self.assertEqual(encrypted["trace_exact"], 1.0)
            self.assertEqual(encrypted["decrypted_plain_state_bit_accuracy"], 1.0)
            self.assertEqual(encrypted["recovered_mask_bit_accuracy"], 1.0)
            self.assertEqual(encrypted["twin_trace_match_rate"], 1.0)
            self.assertEqual(
                encrypted["trace_only_individual_bit_recovery_upper_bound"], 0.5
            )


if __name__ == "__main__":
    unittest.main()
