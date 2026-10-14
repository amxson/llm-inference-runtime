import pytest

from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.request import RequestState
from llm_runtime.scheduler import DecodeScheduler
from runtime_fakes import FakeModel, FakeTokenizer


def test_fifo_round_robin_prefill_and_request_local_cache_parity():
    model, tokenizer = FakeModel(), FakeTokenizer()
    scheduler = DecodeScheduler(model, tokenizer, max_active=3)
    states = [RequestState(prompt, limit, request_id=prompt) for prompt, limit in [("a", 1), ("b", 3), ("c", 2)]]
    for state in states:
        scheduler.add_request(state)
    results = scheduler.run_until_complete()
    assert scheduler.admission_order == ["a", "b", "c"]
    assert [event["request_id"] for event in scheduler.trace] == ["a", "b", "c", "b", "c", "b"]
    assert [call["input_ids"].shape[0] for call in model.calls] == [1] * 6
    assert [call["input_ids"].shape[1] for call in model.calls] == [2, 1, 3, 1, 1, 1]
    assert model.calls[3]["cache"] is model.caches[1]
    assert model.calls[4]["cache"] is model.caches[2]
    assert model.calls[5]["cache"] is model.caches[3]
    for state, result in zip(states, results):
        expected = cached_greedy_decode(FakeModel(), tokenizer, state.prompt, state.max_new_tokens)
        assert result["generated_token_ids"] == expected["generated_token_ids"]
        assert state.is_finished and state.kv_cache is None


def test_eos_zero_token_and_capacity_do_not_stop_other_requests():
    model = FakeModel()
    scheduler = DecodeScheduler(model, FakeTokenizer(), max_active=1)
    for prompt, limit in [("eos", 5), ("a", 0), ("b", 2)]:
        scheduler.add_request(RequestState(prompt, limit, request_id=prompt))
    results = scheduler.run_until_complete()
    assert [result["stop_reason"] for result in results] == ["eos", "max_new_tokens", "max_new_tokens"]
    assert [result["model_execution_count"] for result in results] == [1, 0, 2]
    assert scheduler.admission_order == ["eos", "a", "b"]
    assert scheduler.step() is None
    with pytest.raises(ValueError):
        scheduler.add_request(RequestState("b", 1, request_id="b"))
