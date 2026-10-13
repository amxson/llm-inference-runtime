from types import SimpleNamespace

import pytest
import torch

from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.greedy_decode import greedy_decode


class FakeTokenizer:
    def __init__(self, eos_token_id=None, with_mask=True):
        self.eos_token_id = eos_token_id
        self.with_mask = with_mask
        self.calls = []

    def __call__(self, prompt, return_tensors):
        self.calls.append((prompt, return_tensors))
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "token_type_ids": torch.tensor([[0, 1, 1]]),
        }
        if self.with_mask:
            inputs["attention_mask"] = torch.tensor([[1, 1, 1]])
        return inputs

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(str(token) for token in token_ids if token != self.eos_token_id)


class FakeModel:
    """Distinct cache objects let tests detect stale-cache reuse."""
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.returned_caches = []

    def parameters(self):
        return iter([self.parameter])

    def __call__(self, use_cache, past_key_values=None, **inputs):
        assert torch.is_inference_mode_enabled()
        self.calls.append({
            "use_cache": use_cache,
            "cache": past_key_values,
            **{name: tensor.clone() for name, tensor in inputs.items()},
        })
        context = inputs["input_ids"][0].tolist()
        if past_key_values is not None:
            context = past_key_values.token_ids + context
        length = inputs["input_ids"].shape[1]
        logits = torch.zeros(1, length, 16)
        logits[:, :-1, 0] = 100
        logits[:, -1, len(context) + 1] = 5
        cache = SimpleNamespace(token_ids=context) if use_cache else None
        self.returned_caches.append(cache)
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.mark.parametrize("with_mask", [True, False])
def test_prefill_cache_handoff_masks_and_greedy_parity(with_mask):
    model = FakeModel()
    tokenizer = FakeTokenizer(with_mask=with_mask)
    result = cached_greedy_decode(model, tokenizer, "prompt", 3)
    uncached = greedy_decode(FakeModel(), FakeTokenizer(with_mask=with_mask), "prompt", 3)

    assert tokenizer.calls == [("prompt", "pt")]
    assert len(model.calls) == 3
    assert all(call["use_cache"] for call in model.calls)
    assert model.calls[0]["cache"] is None
    assert model.calls[1]["cache"] is model.returned_caches[0]
    assert model.calls[2]["cache"] is model.returned_caches[1]
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[1, 2, 3]], [[4]], [[5]],
    ]
    assert [step["phase"] for step in result["steps"]] == ["prefill", "decode", "decode"]
    assert [step["model_input_length"] for step in result["steps"]] == [3, 1, 1]
    assert [step["logical_context_length"] for step in result["steps"]] == [3, 4, 5]
    assert result["generated_token_ids"] == uncached["generated_token_ids"] == [4, 5, 6]
    for name in ["full_token_ids", "generated_text", "full_text", "stop_reason"]:
        assert result[name] == uncached[name]
    for index, call in enumerate(model.calls):
        assert call["input_ids"].device == model.parameter.device
        assert call["token_type_ids"].tolist() == ([[0, 1, 1]] if index == 0 else [[1]])
        if with_mask:
            assert call["attention_mask"].tolist() == [[1] * (3 + index)]
        else:
            assert "attention_mask" not in call


def test_eos_stops_without_extra_call_and_skips_special_text():
    model = FakeModel()
    result = cached_greedy_decode(model, FakeTokenizer(eos_token_id=5), "prompt", 5)
    assert len(model.calls) == 2
    assert result["generated_token_ids"] == [4, 5]
    assert result["generated_text"] == "4"
    assert result["full_text"] == "1 2 3 4"
    assert result["stopped_on_eos"] is True
    assert result["stop_reason"] == "eos"


def test_zero_tokens_and_new_request_start_without_cache():
    model = FakeModel()
    zero = cached_greedy_decode(model, FakeTokenizer(), "prompt", 0)
    assert model.calls == []
    assert zero["generated_token_ids"] == zero["steps"] == []
    assert zero["full_token_ids"] == [1, 2, 3]
    assert zero["generated_token_count"] == 0
    assert zero["stopped_on_eos"] is False
    for _ in range(2):
        cached_greedy_decode(model, FakeTokenizer(), "prompt", 1)
    assert all(call["cache"] is None for call in model.calls)


def test_negative_limit_rejected_before_tokenization():
    model, tokenizer = FakeModel(), FakeTokenizer()
    with pytest.raises(ValueError, match="max_new_tokens"):
        cached_greedy_decode(model, tokenizer, "prompt", -1)
    assert model.calls == tokenizer.calls == []


def test_missing_cache_fails_instead_of_silently_recomputing():
    class NoCacheModel(FakeModel):
        def __call__(self, **inputs):
            outputs = super().__call__(**inputs)
            outputs.past_key_values = None
            return outputs

    with pytest.raises(RuntimeError, match="did not return a KV cache"):
        cached_greedy_decode(NoCacheModel(), FakeTokenizer(), "prompt", 2)
