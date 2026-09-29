from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chain_of_lies.inference import api
from chain_of_lies.types import ExperimentPrompt


class _FakeBatchEncoding(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[0] * 5, [0] * 5])
        self.input_ids = SimpleNamespace(shape=(2, 5))

    def to(self, _device: str) -> "_FakeBatchEncoding":
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]

    def __call__(self, _texts, **_kwargs):
        return _FakeBatchEncoding()

    def decode(self, token_ids, **_kwargs):
        return ",".join(str(token_id) for token_id in token_ids)


class _FakeModel:
    device = "cpu"

    def generate(self, **_kwargs):
        # Five padded input positions followed by two generated positions.
        self.generated = _FakeGenerated(
            [
                [0, 0, 11, 12, 13, 101, 102],
                [21, 22, 23, 24, 25, 201, 202],
            ]
        )
        return self.generated


class _FakeGenerated:
    def __init__(self, values, counter=None):
        self.values = values
        self.counter = counter if counter is not None else {"cpu_transfers": 0}

    def __getitem__(self, index):
        rows, columns = index
        return _FakeGenerated([row[columns] for row in self.values[rows]], self.counter)

    def detach(self):
        return self

    def cpu(self):
        self.counter["cpu_transfers"] += 1
        return self

    def tolist(self):
        return self.values


class BatchedInferenceTests(unittest.TestCase):
    def test_decodes_only_tokens_after_common_padded_input_width(self):
        prompts = [
            ExperimentPrompt(prompt_text="short", experiment_id="short"),
            ExperimentPrompt(prompt_text="longer", experiment_id="longer"),
        ]
        model = _FakeModel()
        with patch.object(
            api,
            "_get_model_and_tokenizer",
            return_value=(model, _FakeTokenizer()),
        ):
            responses = api.run_inference_batch(prompts, model_id="fake")

        self.assertEqual(
            [response.raw_text for response in responses], ["101,102", "201,202"]
        )
        self.assertEqual(
            [response.generated_token_ids for response in responses],
            [[101, 102], [201, 202]],
        )
        self.assertEqual(model.generated.counter["cpu_transfers"], 1)

    def test_single_prompt_transfers_once_and_preserves_tokens(self):
        model = _FakeModel()
        with patch.object(
            api, "_get_model_and_tokenizer", return_value=(model, _FakeTokenizer())
        ):
            response = api.run_inference(
                ExperimentPrompt(prompt_text="test", experiment_id="test"),
                model_id="fake",
            )
        self.assertEqual(response.generated_token_ids, [101, 102])
        self.assertEqual(response.raw_text, "101,102")
        self.assertEqual(model.generated.counter["cpu_transfers"], 1)


if __name__ == "__main__":
    unittest.main()
