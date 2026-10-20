import pytest
import torch

from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.request import RequestState
from test_cache_batch import tiny_gpt2


class TinyTokenizer:
    eos_token_id = None
    pad_token_id = 0

    def __call__(self, prompt, return_tensors):
        ids = [int(n) for n in prompt.split()]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


SPECS = [("1 2 3", 2, "A"), ("4 5 6 7 8 9 10 11", 5, "B"),
         ("12 13 14 15 16", 5, "C"), ("17 18", 4, "D")]


def test_changing_ragged_cohorts_real_gpt2_exact_independent_cached_parity():
    model, tokenizer = tiny_gpt2(), TinyTokenizer()
    calls = []
    hook = model.register_forward_pre_hook(lambda _, args, kwargs: calls.append(
        {"shape": list(kwargs["input_ids"].shape), "cached": kwargs["past_key_values"] is not None
         if "past_key_values" in kwargs else False}), with_kwargs=True)
    engine = CachedContinuousBatcher(model, tokenizer, 2)
    for prompt, budget, name in SPECS:
        engine.add_request(RequestState(prompt, budget, name))
    engine.run_until_complete()
    hook.remove()
    traces = list(engine.trace)
    assert traces[0]["prefill_input_shape"] == [2, 8]
    assert traces[0]["logical_cache_lengths_after"] == {"A": 3, "B": 8}
    assert traces[1]["logical_cache_lengths_before"] == {"A": 3, "B": 8}
    assert traces[1]["logical_cache_lengths_after"] == {"A": 4, "B": 9}
    assert traces[1]["completed"] == ["A"]
    assert traces[2]["admitted"] == ["C"]
    assert traces[2]["prefill_request_ids"] == ["C"]
    assert traces[2]["decode_request_ids"] == ["B"]
    assert traces[2]["model_forward_count_this_tick"] == 2
    assert traces[3]["decode_request_ids"] == ["B", "C"]
    assert traces[5]["admitted"] == ["D"]
    assert traces[6]["decode_request_ids"] == ["C", "D"]
    assert len(calls) == engine.model_forward_count
    assert all(call["shape"][1] == 1 for call in calls if call["cached"])
    assert any(call["shape"] == [2, 1] for call in calls if call["cached"])
    for frame in traces:
        assert len(frame["events"]) == len(set(frame["model_request_ids"]))
        for event in frame["events"]:
            if event["phase"] == "decode":
                assert event["logical_cache_length_after"] == event["logical_cache_length_before"] + 1
    for state in engine.requests.values():
        expected = cached_greedy_decode(model, tokenizer, state.prompt, state.max_new_tokens)
        assert state.generated_token_ids == expected["generated_token_ids"]
        assert state.kv_cache is None and not state.runtime_inputs
    assert engine.step() is None


def test_equal_length_rows_zero_cancel_and_ownership():
    engine = CachedContinuousBatcher(tiny_gpt2(), TinyTokenizer(), 2)
    states = [RequestState("1 2 3", budget, name) for name, budget in [("A", 4), ("B", 4), ("C", 0)]]
    for state in states:
        engine.add_request(state)
    engine.step()
    assert [s.kv_cache.get_seq_length() for s in states[:2]] == [3, 3]
    from llm_runtime.cache_batch import cache_tensors
    a, b = [cache_tensors(s.kv_cache)[0][0] for s in states[:2]]
    assert a.data_ptr() != b.data_ptr()
    engine.cancel_request("A")
    assert states[0].kv_cache is None and states[0].stop_reason == "cancelled"
    trace = engine.step()
    assert trace["completed"] == ["C"] and trace["prefill_input_shape"] is None
    assert states[2].model_execution_count == 0 and states[2].kv_cache is None
    engine.cancel_request("B")
    for state in states:
        engine.forget_request(state.request_id)
    assert not engine.has_work and not engine.requests


def test_eos_row_leaves_neighbor_unchanged():
    model, tokenizer = tiny_gpt2(), TinyTokenizer()
    selected = cached_greedy_decode(model, tokenizer, "1 2 3", 1)["generated_token_ids"][0]
    tokenizer.eos_token_id = selected
    engine = CachedContinuousBatcher(model, tokenizer, 2)
    for prompt, budget, name in SPECS[:2]:
        engine.add_request(RequestState(prompt, budget, name))
    engine.run_until_complete()
    assert engine.requests["A"].stop_reason == "eos"
    assert engine.requests["A"].generated_token_count == 1
    for state in engine.requests.values():
        assert state.generated_token_ids == cached_greedy_decode(
            model, tokenizer, state.prompt, state.max_new_tokens)["generated_token_ids"]
        assert state.kv_cache is None


def test_context_boundary_error_and_lifecycle_validation():
    model, tokenizer = tiny_gpt2(), TinyTokenizer()
    engine = CachedContinuousBatcher(model, tokenizer, 1)
    exact = RequestState(" ".join(["1"] * 63), 1, "exact")
    engine.add_request(exact)
    with pytest.raises(ValueError):
        engine.add_request(exact)
    with pytest.raises(ValueError):
        engine.forget_request("exact")
    engine.run_until_complete()
    assert exact.stop_reason == "max_new_tokens"
    bad = RequestState(" ".join(["1"] * 63), 2, "bad")
    engine.add_request(bad)
    with pytest.raises(ValueError, match="context limit"):
        engine.step()
    assert bad.stop_reason == "error" and bad.kv_cache is None
    engine.cancel_request("bad")
    def broken(*args, **kwargs):
        raise RuntimeError("forward failure")
    model.forward = broken
    failed = RequestState("1 2", 3, "failed")
    engine.add_request(failed)
    with pytest.raises(RuntimeError, match="forward failure"):
        engine.step()
    assert failed.stop_reason == "error" and failed.kv_cache is None
    with pytest.raises(ValueError):
        CachedContinuousBatcher(model, tokenizer, 0)
