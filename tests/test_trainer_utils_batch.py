from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from chain_of_lies.training.shared.trainer_utils import _generate_batch


class _Encoding(dict):
    def __init__(self) -> None:
        input_ids = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 24, 25]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, _device: torch.device) -> "_Encoding":
        return self


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]

    def __call__(self, _texts, **_kwargs):
        return _Encoding()

    def decode(self, token_ids, **_kwargs):
        return ",".join(str(int(token_id)) for token_id in token_ids)


class _Model:
    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def generate(self, **_kwargs):
        return torch.tensor(
            [
                [0, 0, 11, 12, 13, 101, 102],
                [21, 22, 23, 24, 25, 201, 202],
            ]
        )


class TrainerBatchDecodingTests(unittest.TestCase):
    def test_left_padded_batch_decodes_after_common_input_width(self) -> None:
        completions = _generate_batch(
            _Model(),
            _Tokenizer(),
            ["short", "longer"],
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            device=torch.device("cpu"),
            do_sample=False,
        )
        self.assertEqual(completions, ["101,102", "201,202"])


if __name__ == "__main__":
    unittest.main()
