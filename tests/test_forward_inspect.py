from types import SimpleNamespace

import torch

from llm_runtime.forward_inspect import inspect_forward_pass


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        self.calls.append((prompt, return_tensors))
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"piece-{token_id}" for token_id in token_ids]

    def decode(self, token_ids: list[int]) -> str:
        return f"token-{token_ids[0]}"


class FakeModel:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.call_count = 0
        self.received_inputs: dict[str, torch.Tensor] = {}
        self.logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 200.0, 0.0, 0.0, 0.0],
                    [0.5, 4.0, -1.0, 2.0, 3.0, 1.0],
                ]
            ]
        )

    def parameters(self):
        return iter([self.parameter])

    def __call__(self, **model_inputs: torch.Tensor) -> SimpleNamespace:
        self.call_count += 1
        self.received_inputs = model_inputs
        return SimpleNamespace(logits=self.logits)


def test_inspect_forward_pass_reports_final_position_candidates() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    inspection = inspect_forward_pass(model, tokenizer, "test prompt")

    assert tokenizer.calls == [("test prompt", "pt")]
    assert model.call_count == 1
    assert set(model.received_inputs) == {"input_ids", "attention_mask"}
    assert inspection["token_ids"] == [10, 11, 12]
    assert inspection["token_pieces"] == ["piece-10", "piece-11", "piece-12"]
    assert inspection["input_ids"] == {
        "shape": (1, 3),
        "dtype": "torch.int64",
        "device": "cpu",
        "batch_size": 1,
        "sequence_length": 3,
    }
    assert inspection["logits"] == {
        "shape": (1, 3, 6),
        "dtype": "torch.float32",
        "device": "cpu",
    }
    assert inspection["last_position_logits"]["shape"] == (1, 6)
    assert [candidate["token_id"] for candidate in inspection["top_candidates"]] == [
        1,
        4,
        3,
        5,
        0,
    ]
    assert [candidate["raw_logit"] for candidate in inspection["top_candidates"]] == [
        4.0,
        3.0,
        2.0,
        1.0,
        0.5,
    ]
    assert inspection["top_candidates"][0] == {
        "rank": 1,
        "token_id": 1,
        "token_text": "token-1",
        "raw_logit": 4.0,
    }
    assert inspection["greedy_candidate"] == {
        "token_id": 1,
        "token_text": "token-1",
        "raw_logit": 4.0,
    }
