import torch
import pytest

from llm_runtime.batching import batched_greedy_decode
from llm_runtime.greedy_decode import greedy_decode
from llm_runtime.request import RequestState
from runtime_fakes import FakeModel, FakeTokenizer


def test_real_batch_rows_padding_positions_and_individual_parity():
    model, tokenizer = FakeModel(), FakeTokenizer()
    states = [RequestState("a", 3), RequestState("b", 2), RequestState("c", 1)]
    result = batched_greedy_decode(model, tokenizer, states)
    assert [call["input_ids"].shape[0] for call in model.calls] == [3, 2, 1]
    assert len(model.calls) == result["model_forward_call_count"] == 3
    first = model.calls[0]
    assert first["input_ids"].tolist() == [[9, 1, 2], [9, 9, 4], [7, 1, 2]]
    assert first["attention_mask"].tolist() == [[0, 1, 1], [0, 0, 1], [1, 1, 1]]
    assert first["position_ids"].tolist() == [[0, 0, 1], [0, 0, 0], [0, 1, 2]]
    assert tokenizer.pad_token_id is None and tokenizer.padding_side == "right"
    assert all(call["use_cache"] is False and call["cache"] is None for call in model.calls)
    for state in states:
        reference = greedy_decode(FakeModel(), tokenizer, state.prompt, state.max_new_tokens)
        assert state.generated_token_ids == reference["generated_token_ids"]
        assert state.kv_cache is None


def test_eos_and_zero_limits_finish_independently_and_do_not_reenter_batch():
    model = FakeModel()
    states = [RequestState("eos", 5), RequestState("a", 0), RequestState("b", 3)]
    result = batched_greedy_decode(model, FakeTokenizer(), states)
    assert [state.generated_token_count for state in states] == [1, 0, 3]
    assert [call["input_ids"].shape[0] for call in model.calls] == [2, 1, 1]
    assert states[0].stop_reason == "eos"
    assert result["use_cache"] is False
    empty = batched_greedy_decode(model, FakeTokenizer(), [])
    assert empty["results"] == [] and empty["model_forward_call_count"] == 0
    with pytest.raises(ValueError):
        batched_greedy_decode(model, FakeTokenizer(), [states[0]])
