import json
from types import SimpleNamespace

import pytest
import torch

from llm_runtime import benchmark_cache as benchmark
from llm_runtime.cached_decode import cached_greedy_decode


class FakeTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
        }

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(str(token) for token in ids if token != self.eos_token_id)


class FakeModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.caches = []

    def parameters(self):
        return iter([self.parameter])

    def __call__(self, use_cache, past_key_values=None, **inputs):
        assert torch.is_inference_mode_enabled()
        self.calls.append({
            "cache": past_key_values, "use_cache": use_cache,
            **{name: tensor.clone() for name, tensor in inputs.items()},
        })
        context = inputs["input_ids"][0].tolist()
        if past_key_values is not None:
            context = past_key_values.token_ids + context
        logits = torch.zeros(1, inputs["input_ids"].shape[1], 32)
        logits[:, :-1, 0] = 100
        logits[:, -1, len(context) + 1] = 5
        cache = SimpleNamespace(token_ids=context) if use_cache else None
        self.caches.append(cache)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_timed_cached_path_preserves_m6_semantics_and_m5_metric_math(monkeypatch):
    timestamps = iter([10.0, 10.25, 10.5, 11.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(timestamps))
    model = FakeModel()
    result = benchmark.timed_cached_greedy_decode(model, FakeTokenizer(), "prompt", 3)
    reference = cached_greedy_decode(FakeModel(), FakeTokenizer(), "prompt", 3)

    assert result["generated_token_ids"] == reference["generated_token_ids"] == [6, 7, 8]
    assert result["generated_text"] == reference["generated_text"]
    assert result["forward_input_lengths"] == [5, 1, 1]
    assert result["logical_context_lengths"] == [5, 6, 7]
    assert result["total_input_token_positions_processed"] == 7
    assert result["model_forward_call_count"] == 3
    assert result["ttft_ms"] == 250.0
    assert result["total_latency_ms"] == 1000.0
    assert result["inter_token_latencies_ms"] == [250.0, 500.0]
    assert result["mean_itl_ms"] == 375.0
    assert result["generated_tokens_per_second"] == 3.0
    assert model.calls[0]["cache"] is None
    for index, call in enumerate(model.calls):
        assert call["use_cache"] is True
        assert call["attention_mask"].shape[1] == 5 + index
        assert call["token_type_ids"].shape == call["input_ids"].shape
        if index:
            assert call["cache"] is model.caches[index - 1]


def synthetic_run(total, positions, ids=None):
    return {
        "generated_token_ids": [6, 7] if ids is None else ids,
        "forward_input_lengths": [5, positions - 5],
        "total_input_token_positions_processed": positions,
        **benchmark.calculate_timing_metrics(0.0, [0.5, total]),
    }


def test_comparison_ratios_use_named_numerators_and_denominators():
    comparison = benchmark.compare_runs([synthetic_run(2.0, 11)], [synthetic_run(1.0, 6)])
    assert comparison["exact_token_id_parity"] is True
    assert comparison["mean_uncached_input_positions"] == 11
    assert comparison["mean_cached_input_positions"] == 6
    assert comparison["input_position_ratio_cached_over_uncached"] == pytest.approx(6 / 11)
    assert comparison["latency_speedup_uncached_over_cached"] == 2.0
    assert comparison["throughput_speedup_cached_over_uncached"] == 2.0
    assert comparison["itl_ratio_uncached_over_cached"] == 3.0
    assert comparison["ttft_ratio_uncached_over_cached"] == 1.0
    slower = benchmark.compare_runs([synthetic_run(1.0, 11)], [synthetic_run(2.0, 6)])
    assert slower["latency_speedup_uncached_over_cached"] == 0.5


@pytest.mark.parametrize("uncached, cached", [
    ([synthetic_run(2, 11)], [synthetic_run(1, 6, [6, 8])]),
    ([synthetic_run(2, 11), synthetic_run(2, 11, [6, 8])], [synthetic_run(1, 6)] * 2),
    ([synthetic_run(2, 11)] * 2, [synthetic_run(1, 6), synthetic_run(1, 6, [6, 8])]),
])
def test_comparison_refuses_parity_mismatches_in_any_measured_run(uncached, cached):
    with pytest.raises(ValueError, match="Token parity failed"):
        benchmark.compare_runs(uncached, cached)


def test_warmups_excluded_order_alternates_and_json_serializes(monkeypatch):
    model, tokenizer = FakeModel(), FakeTokenizer()
    call_order = []
    counts = {"uncached": 0, "cached": 0}

    def execute(mode, passed_model, passed_tokenizer, prompt, max_new_tokens):
        assert passed_model is model and passed_tokenizer is tokenizer
        assert prompt == "prompt" and max_new_tokens == 2
        call_order.append(mode)
        counts[mode] += 1
        total = 100.0 if counts[mode] == 1 else float(counts[mode])
        return synthetic_run(total, 11 if mode == "uncached" else 6)

    monkeypatch.setattr(benchmark, "timed_greedy_decode", lambda *args: execute("uncached", *args))
    monkeypatch.setattr(benchmark, "timed_cached_greedy_decode", lambda *args: execute("cached", *args))
    result = benchmark.run_comparison(model, tokenizer, "prompt", 2, 1, 2, "fake")
    assert call_order == ["uncached", "cached", "uncached", "cached", "cached", "uncached"]
    assert result["uncached"]["summary"]["mean_total_latency_ms"] == 2500.0
    assert result["cached"]["summary"]["mean_total_latency_ms"] == 2500.0
    assert len(result["uncached"]["runs"]) == len(result["cached"]["runs"]) == 2
    assert all(run["cuda_max_memory_allocated_bytes"] is None for run in result["cached"]["runs"])
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert set(restored) == {"benchmark", "environment", "config", "uncached", "cached", "comparison"}


def test_paired_fake_paths_show_95_vs_14_input_positions():
    result = benchmark.run_comparison(FakeModel(), FakeTokenizer(), "prompt", 10, 0, 1, "fake")
    assert result["comparison"]["exact_token_id_parity"] is True
    assert result["uncached"]["runs"][0]["forward_input_lengths"] == list(range(5, 15))
    assert result["cached"]["runs"][0]["forward_input_lengths"] == [5] + [1] * 9
    assert result["cached"]["runs"][0]["logical_context_lengths"] == list(range(5, 15))
    assert result["comparison"]["mean_uncached_input_positions"] == 95
    assert result["comparison"]["mean_cached_input_positions"] == 14


def test_zero_tokens_and_eos_have_well_defined_edge_metrics():
    model = FakeModel()
    zero = benchmark.run_comparison(model, FakeTokenizer(), "prompt", 0, 1, 1, "fake")
    assert model.calls == []
    assert zero["comparison"]["exact_token_id_parity"] is True
    assert zero["comparison"]["latency_speedup_uncached_over_cached"] is None
    assert zero["comparison"]["throughput_speedup_cached_over_uncached"] is None
    assert zero["comparison"]["input_position_ratio_cached_over_uncached"] is None
    assert zero["comparison"]["itl_ratio_uncached_over_cached"] is None
    json.dumps(zero, allow_nan=False)

    eos = benchmark.timed_cached_greedy_decode(model, FakeTokenizer(eos_token_id=6), "prompt", 10)
    assert eos["generated_token_ids"] == [6]
    assert eos["model_forward_call_count"] == 1
    assert eos["stop_reason"] == "eos" and eos["stopped_on_eos"]
    assert eos["mean_itl_ms"] is None


def test_cli_does_not_write_comparison_file_on_parity_failure(monkeypatch, tmp_path):
    output = tmp_path / "invalid_comparison.json"
    monkeypatch.setattr("sys.argv", ["benchmark_cache", "--output", str(output)])
    monkeypatch.setattr(benchmark, "load_model_and_tokenizer", lambda *args: (FakeModel(), FakeTokenizer()))

    def fail(*args):
        raise ValueError("Token parity failed; the benchmark comparison is invalid.")

    monkeypatch.setattr(benchmark, "compare_runs", fail)
    with pytest.raises(ValueError, match="Token parity failed"):
        benchmark.main()
    assert not output.exists()
