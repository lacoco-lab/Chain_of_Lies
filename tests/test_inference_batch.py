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
        return [
            [0, 0, 11, 12, 13, 101, 102],
            [21, 22, 23, 24, 25, 201, 202],
        ]


class BatchedInferenceTests(unittest.TestCase):
    def test_decodes_only_tokens_after_common_padded_input_width(self):
        prompts = [
            ExperimentPrompt(prompt_text="short", experiment_id="short"),
            ExperimentPrompt(prompt_text="longer", experiment_id="longer"),
        ]
        with patch.object(
            api,
            "_get_model_and_tokenizer",
            return_value=(_FakeModel(), _FakeTokenizer()),
        ):
            responses = api.run_inference_batch(prompts, model_id="fake")

        self.assertEqual([response.raw_text for response in responses], ["101,102", "201,202"])
        self.assertEqual(
            [response.generated_token_ids for response in responses],
            [[101, 102], [201, 202]],
        )


if __name__ == "__main__":
    unittest.main()
