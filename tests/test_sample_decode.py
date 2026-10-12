from types import SimpleNamespace

import pytest
import torch

from llm_runtime import sample_decode as decoder


class FakeTokenizer:
    def __init__(self, eos_token_id=None, with_mask=True):
        self.eos_token_id = eos_token_id
        self.with_mask = with_mask
        self.calls = []

    def __call__(self, prompt, return_tensors):
        self.calls.append((prompt, return_tensors))
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "token_type_ids": torch.tensor([[0, 1]]),
        }
        if self.with_mask:
            inputs["attention_mask"] = torch.tensor([[1, 1]])
        return inputs

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens
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
        logits[:, :-1, 0] = 100
        logits[:, -1, length + 1] = 5
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("with_mask", [True, False])
def test_sampling_appends_tokens_and_grows_full_inputs_until_limit(with_mask):
    model = FakeModel()
    tokenizer = FakeTokenizer(with_mask=with_mask)
    result = decoder.sample_decode(model, tokenizer, "prompt", 3, top_k=1, seed=42)

    assert tokenizer.calls == [("prompt", "pt")]
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]],
    ]
    assert result["generated_token_ids"] == [3, 4, 5]
    assert result["full_token_ids"] == [1, 2, 3, 4, 5]
    assert result["generated_text"] == "3 4 5"
    assert result["full_text"] == "1 2 3 4 5"
    assert result["generated_token_count"] == 3
    assert result["stop_reason"] == "max_new_tokens"
    assert result["stopped_on_eos"] is False
    assert [step["input_sequence_length"] for step in result["steps"]] == [2, 3, 4]
    for call in model.calls:
        length = call["input_ids"].shape[1]
        assert call["input_ids"].device == model.parameter.device
        assert call["token_type_ids"].tolist() == [[0] + [1] * (length - 1)]
        if with_mask:
            assert call["attention_mask"].tolist() == [[1] * length]
        else:
            assert "attention_mask" not in call


def test_sampling_stops_at_eos_without_an_extra_call():
    model = FakeModel()
    result = decoder.sample_decode(model, FakeTokenizer(eos_token_id=4), "prompt", 5, top_k=1)
    assert len(model.calls) == 2
    assert result["generated_token_ids"] == [3, 4]
    assert result["full_token_ids"] == [1, 2, 3, 4]
    assert result["generated_text"] == "3"
    assert result["full_text"] == "1 2 3"
    assert result["generated_token_count"] == 2
    assert result["stop_reason"] == "eos"
    assert result["stopped_on_eos"] is True


def test_zero_token_limit_performs_no_forward_calls():
    model = FakeModel()
    result = decoder.sample_decode(model, FakeTokenizer(), "prompt", 0, seed=42)
    assert model.calls == []
    assert result["generated_token_ids"] == []
    assert result["generated_text"] == ""
    assert result["full_text"] == "1 2"
    assert result["generated_token_count"] == 0
    assert result["stop_reason"] == "max_new_tokens"
    assert result["stopped_on_eos"] is False
    assert result["steps"] == []


def test_sampling_configuration_and_final_logits_reach_selection(monkeypatch):
    received = []

    def fake_select(logits, temperature, top_k, top_p, generator):
        received.append((temperature, top_k, top_p, generator))
        assert logits.shape == (8,)
        assert logits[0].item() == 0  # Not the earlier-position value 100.
        return 3

    monkeypatch.setattr(decoder, "sample_next_token", fake_select)
    result = decoder.sample_decode(FakeModel(), FakeTokenizer(), "prompt", 2, 0.8, 4, 0.95, 42)
    assert len(received) == 2
    assert received[0][:3] == (0.8, 4, 0.95)
    assert received[1][:3] == (0.8, 4, 0.95)
    assert received[0][3] is received[1][3]
    assert received[0][3].initial_seed() == 42
    assert received[0][3].device == torch.device("cpu")
    assert result["sampling_config"] == {"temperature": 0.8, "top_k": 4, "top_p": 0.95, "seed": 42}
    assert result["generated_token_ids"] == [3, 3]


def test_seeded_decoder_repeats_without_changing_global_rng_state():
    state = torch.random.get_rng_state().clone()
    first = decoder.sample_decode(FakeModel(), FakeTokenizer(), "prompt", 3, seed=42)
    second = decoder.sample_decode(FakeModel(), FakeTokenizer(), "prompt", 3, seed=42)
    assert first["generated_token_ids"] == second["generated_token_ids"]
    assert torch.equal(torch.random.get_rng_state(), state)


def test_invalid_settings_fail_before_tokenization_even_with_zero_limit():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="temperature"):
        decoder.sample_decode(model, tokenizer, "prompt", 0, temperature=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        decoder.sample_decode(model, tokenizer, "prompt", -1)
    assert model.calls == []
    assert tokenizer.calls == []
