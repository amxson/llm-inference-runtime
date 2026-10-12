from types import SimpleNamespace

import pytest
import torch

from llm_runtime.greedy_decode import greedy_decode


class FakeTokenizer:
    def __init__(self, eos_token_id=None, with_mask=True, token_ids=None, with_token_types=False):
        self.eos_token_id = eos_token_id
        self.with_mask = with_mask
        self.token_ids = [1, 2] if token_ids is None else token_ids
        self.with_token_types = with_token_types
        self.calls = []

    def __call__(self, prompt, return_tensors):
        self.calls.append((prompt, return_tensors))
        inputs = {"input_ids": torch.tensor([self.token_ids], dtype=torch.long)}
        if self.with_mask:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if self.with_token_types:
            inputs["token_type_ids"] = torch.tensor([[0, 1]])
        return inputs

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(str(token) for token in token_ids if token != self.eos_token_id)


class FakeModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def parameters(self):
        return iter([self.parameter])

    def __call__(self, use_cache, **inputs):
        assert use_cache is False
        assert torch.is_inference_mode_enabled()
        self.calls.append({name: tensor.clone() for name, tensor in inputs.items()})
        length = inputs["input_ids"].shape[1]
        logits = torch.zeros(1, length, 8)
        logits[:, :-1, 0] = 100  # Earlier positions must not drive selection.
        logits[:, -1, length + 1] = 5
        return SimpleNamespace(logits=logits)


def test_multistep_argmax_appends_tokens_and_stops_at_limit():
    model = FakeModel()
    tokenizer = FakeTokenizer()

    result = greedy_decode(model, tokenizer, "prompt", 3)

    assert tokenizer.calls == [("prompt", "pt")]
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]],
    ]
    assert result["prompt_token_ids"] == [1, 2]
    assert result["generated_token_ids"] == [3, 4, 5]
    assert result["full_token_ids"] == [1, 2, 3, 4, 5]
    assert result["generated_text"] == "3 4 5"
    assert result["full_text"] == "1 2 3 4 5"
    assert result["generated_token_count"] == 3
    assert result["max_new_tokens"] == 3
    assert result["stop_reason"] == "max_new_tokens"
    assert result["stopped_on_eos"] is False
    assert [step["input_sequence_length"] for step in result["steps"]] == [2, 3, 4]
    for call in model.calls:
        assert call["attention_mask"].shape == call["input_ids"].shape
        assert call["attention_mask"].tolist() == [[1] * call["input_ids"].shape[1]]
        assert call["input_ids"].device == model.parameter.device


def test_eos_stops_early_and_is_retained_in_ids_but_skipped_in_text():
    model = FakeModel()
    result = greedy_decode(model, FakeTokenizer(eos_token_id=4), "prompt", 5)

    assert len(model.calls) == 2
    assert result["generated_token_ids"] == [3, 4]
    assert result["full_token_ids"] == [1, 2, 3, 4]
    assert result["generated_text"] == "3"
    assert result["full_text"] == "1 2 3"
    assert result["generated_token_count"] == 2
    assert result["stopped_on_eos"] is True
    assert result["stop_reason"] == "eos"


def test_zero_new_tokens_performs_no_forward_calls():
    model = FakeModel()
    result = greedy_decode(model, FakeTokenizer(), "prompt", 0)

    assert model.calls == []
    assert result["generated_token_ids"] == []
    assert result["generated_text"] == ""
    assert result["generated_token_count"] == 0
    assert result["full_token_ids"] == [1, 2]
    assert result["full_text"] == "1 2"
    assert result["steps"] == []
    assert result["stopped_on_eos"] is False
    assert result["stop_reason"] == "max_new_tokens"


def test_decoding_without_attention_mask():
    model = FakeModel()
    result = greedy_decode(model, FakeTokenizer(with_mask=False), "prompt", 2)

    assert len(model.calls) == 2
    assert all(set(call) == {"input_ids"} for call in model.calls)
    assert result["generated_token_ids"] == [3, 4]


def test_negative_token_limit_is_rejected_before_tokenization():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
        greedy_decode(model, tokenizer, "prompt", -1)
    assert tokenizer.calls == []
    assert model.calls == []


def test_token_types_continue_the_last_prompt_type():
    model = FakeModel()
    greedy_decode(model, FakeTokenizer(with_token_types=True), "prompt", 3)

    assert [call["token_type_ids"].tolist() for call in model.calls] == [
        [[0, 1]], [[0, 1, 1]], [[0, 1, 1, 1]],
    ]


def test_empty_token_sequence_is_rejected_without_forward_call():
    model = FakeModel()
    with pytest.raises(ValueError, match="at least one token"):
        greedy_decode(model, FakeTokenizer(token_ids=[]), "", 1)
    assert model.calls == []
