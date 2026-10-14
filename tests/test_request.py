import json
from types import SimpleNamespace

import pytest

from llm_runtime.request import RequestState, RequestStatus


def test_lifecycle_token_recording_and_runtime_cleanup():
    state = RequestState("prompt", 2, request_id="a")
    assert state.status is RequestStatus.WAITING
    state.start_prefill([1, 2], {"attention_mask": object()})
    state.kv_cache = object()
    assert state.status is RequestStatus.PREFILL
    state.record_token(3)
    assert state.status is RequestStatus.DECODING
    assert state.generated_token_count == 1 and state.logical_context_length == 3
    state.record_token(4)
    assert state.is_finished and state.stop_reason == "max_new_tokens"
    assert state.model_execution_count == 2
    assert state.runtime_inputs == {} and state.kv_cache is None
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    result = state.to_result(tokenizer)
    assert result["full_token_ids"] == [1, 2, 3, 4]
    assert "kv_cache" not in result and "runtime_inputs" not in result
    json.dumps(result)


def test_eos_and_zero_token_completion():
    eos = RequestState("prompt", 5)
    eos.start_prefill([1])
    eos.record_token(9, eos_token_id=9)
    assert eos.is_finished and eos.stop_reason == "eos"
    assert eos.generated_token_ids == [9]
    zero = RequestState("prompt", 0)
    zero.start_prefill([1])
    assert zero.is_finished and zero.model_execution_count == 0
    assert zero.generated_token_ids == []


def test_ids_are_unique_and_lists_do_not_leak_between_requests():
    first, second = RequestState("a", 1), RequestState("b", 1)
    assert first.request_id != second.request_id
    first.start_prefill([1])
    first.record_token(2)
    assert second.prompt_token_ids == second.generated_token_ids == []


def test_invalid_transitions_and_validation():
    with pytest.raises(ValueError):
        RequestState("prompt", -1)
    with pytest.raises(ValueError):
        RequestState(" ", 1)
    state = RequestState("prompt", 1)
    with pytest.raises(ValueError):
        state.record_token(1)
    with pytest.raises(ValueError):
        state.finish("eos")
    state.start_prefill([1])
    with pytest.raises(ValueError):
        state.start_prefill([1])
    state.record_token(2)
    with pytest.raises(ValueError):
        state.record_token(3)
    with pytest.raises(ValueError):
        state.finish("cancelled")
