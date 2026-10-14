import pytest

from llm_runtime.continuous_batcher import ContinuousBatcher
from llm_runtime.greedy_decode import greedy_decode
from llm_runtime.request import RequestState
from runtime_fakes import FakeModel, FakeTokenizer


def test_active_composition_changes_and_every_row_keeps_individual_result():
    model, tokenizer = FakeModel(), FakeTokenizer()
    engine = ContinuousBatcher(model, tokenizer, max_active=2)
    states = [RequestState(prompt, limit, request_id=name) for prompt, limit, name in [("a", 1, "A"), ("b", 3, "B"), ("c", 2, "C")]]
    for state in states:
        engine.add_request(state)
    first = engine.step()
    assert first["active_request_ids"] == ["A", "B"] and first["completed"] == ["A"]
    assert [state.request_id for state in engine.waiting] == ["C"]
    second = engine.step()
    assert second["active_request_ids"] == ["B", "C"] and second["admitted"] == ["C"]
    assert [event["phase"] for event in second["events"]] == ["decode", "prefill"]
    results = engine.run_until_complete()
    assert len(results) == 3 and not engine.has_work
    assert [call["input_ids"].shape[0] for call in model.calls] == [2, 2, 2]
    assert engine.model_forward_count == 3
    for state in states:
        expected = greedy_decode(FakeModel(), tokenizer, state.prompt, state.max_new_tokens)
        assert state.generated_token_ids == expected["generated_token_ids"]
        assert state.kv_cache is None
    assert engine.step() is None


def test_late_admission_zero_tokens_cancellation_and_history_release():
    engine = ContinuousBatcher(FakeModel(), FakeTokenizer(), max_active=1)
    first = RequestState("b", 3, request_id="first")
    zero = RequestState("a", 0, request_id="zero")
    engine.add_request(first)
    engine.step()
    engine.add_request(zero)
    assert len(engine.waiting) == 1
    with pytest.raises(ValueError):
        engine.forget_request("first")
    engine.cancel_request("first")
    engine.step()
    assert zero.is_finished and zero.model_execution_count == 0
    engine.forget_request("first")
    engine.forget_request("zero")
    assert engine.requests == {} and not engine.has_work
    assert first.stop_reason == "cancelled"


def test_capacity_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        ContinuousBatcher(FakeModel(), FakeTokenizer(), 0)
    engine = ContinuousBatcher(FakeModel(), FakeTokenizer(), 1)
    engine.add_request(RequestState("a", 1, request_id="same"))
    with pytest.raises(ValueError):
        engine.add_request(RequestState("b", 1, request_id="same"))
