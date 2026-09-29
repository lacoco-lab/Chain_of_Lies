from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_splits import (
    complement,
    final_parity,
    generate,
    pair_cumulative_trace,
    single_cumulative_trace,
    supervised_suffix,
)
from summarize import (
    parse_completion,
    recover_bits_from_single_states,
    recover_pairs_from_states,
    summarize,
)
from validate_splits import validate


class ParityProp4Tests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = json.loads(
            Path(
                "experiments/goldreich_parity/qwen_attempts/exploratory/prop4_sanity/config.json"
            ).read_text(encoding="utf-8")
        )
        config["train_n"] = 40
        config["validation_n"] = 20
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_concrete_four_bit_example(self) -> None:
        bits = "1011"
        self.assertEqual(pair_cumulative_trace(bits), [1, 1])
        self.assertEqual(single_cumulative_trace(bits), [1, 1, 0, 1])
        self.assertEqual(final_parity(bits), 1)
        self.assertEqual(recover_pairs_from_states([1, 1]), [1, 0])
        self.assertEqual(recover_bits_from_single_states([1, 1, 0, 1]), [1, 0, 1, 1])

    def test_complementary_twin_has_same_pair_trace_and_opposite_bits(self) -> None:
        bits = "10110010100101100101001011001010"
        twin = complement(bits)
        self.assertEqual(len(bits), 32)
        self.assertEqual(pair_cumulative_trace(bits), pair_cumulative_trace(twin))
        self.assertEqual(final_parity(bits), final_parity(twin))
        self.assertTrue(all(a != b for a, b in zip(bits, twin)))

    def test_generation_is_reproducible_balanced_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            first = generate(config, root / "first")
            second = generate(config, root / "second")
            self.assertEqual(
                first["validation_input_digest_sha256"],
                second["validation_input_digest_sha256"],
            )
            report = validate(config, root / "first")
            self.assertEqual(report["train_n"], 40)
            self.assertEqual(report["validation_n"], 20)
            self.assertEqual(report["train_validation_overlap"], 0)
            self.assertTrue(report["conditions_share_identical_inputs"])
            self.assertTrue(report["no_private_task"])

    def test_targets_parse_strictly(self) -> None:
        bits = "10110010100101100101001011001010"
        for protocol in ("answer_only", "pair_cumulative", "single_cumulative"):
            target = supervised_suffix(bits, protocol)
            parsed = parse_completion(target, protocol)
            self.assertTrue(parsed["strict_format"])
            self.assertEqual(parsed["answer"], final_parity(bits))
            if protocol == "answer_only":
                self.assertIsNone(parsed["trace"])
            else:
                self.assertIsNotNone(parsed["trace"])

    def test_synthetic_perfect_outputs_pass_sanity_gate(self) -> None:
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
            self.assertTrue(summary["passes_sanity_gate"])
            pair = next(
                row
                for row in summary["metrics"]
                if row["condition"] == "prop4_pair_cot" and row["source"] == "finetuned"
            )
            leaky = next(
                row
                for row in summary["metrics"]
                if row["condition"] == "leaky_single_cot"
                and row["source"] == "finetuned"
            )
            self.assertEqual(pair["twin_trace_match_rate"], 1.0)
            self.assertEqual(
                pair["trace_only_individual_bit_recovery_upper_bound"], 0.5
            )
            self.assertEqual(leaky["leaky_individual_bit_recovery"], 1.0)


if __name__ == "__main__":
    unittest.main()
