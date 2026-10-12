import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from llm_runtime import benchmark_naive as benchmark
from llm_runtime.greedy_decode import greedy_decode


class FakeTokenizer:
    def __init__(self, eos_token_id=None, with_mask=True):
        self.eos_token_id = eos_token_id
        self.with_mask = with_mask
        self.calls = []

    def __call__(self, prompt, return_tensors):
        self.calls.append((prompt, return_tensors))
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
        }
        if self.with_mask:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
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
        self.calls.append({name: value.clone() for name, value in inputs.items()})
        length = inputs["input_ids"].shape[1]
        logits = torch.zeros(1, length, 16)
        logits[:, :-1, 0] = 100
        logits[:, -1, length + 1] = 5
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("with_mask", [True, False])
def test_timed_path_preserves_greedy_semantics_and_counts_work(with_mask):
    model = FakeModel()
    tokenizer = FakeTokenizer(with_mask=with_mask)
    result = benchmark.timed_greedy_decode(model, tokenizer, "prompt", 3)
    reference = greedy_decode(FakeModel(), FakeTokenizer(with_mask=with_mask), "prompt", 3)

    assert tokenizer.calls == [("prompt", "pt")]
    assert result["generated_token_ids"] == reference["generated_token_ids"] == [6, 7, 8]
    assert result["generated_text"] == reference["generated_text"]
    assert result["full_text"] == reference["full_text"]
    assert result["prompt_token_count"] == 5
    assert result["generated_token_count"] == result["model_forward_call_count"] == 3
    assert result["forward_input_lengths"] == [5, 6, 7]
    assert result["total_input_token_positions_processed"] == 18
    assert result["stop_reason"] == "max_new_tokens"
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 6, 7]],
    ]
    for call in model.calls:
        length = call["input_ids"].shape[1]
        assert call["input_ids"].device == model.parameter.device
        assert call["token_type_ids"].tolist() == [[0, 0, 0] + [1] * (length - 3)]
        if with_mask:
            assert call["attention_mask"].tolist() == [[1] * length]
        else:
            assert "attention_mask" not in call


def test_metric_calculation_from_known_timestamps():
    metrics = benchmark.calculate_timing_metrics(10.0, [10.25, 10.5, 11.0])
    assert metrics == {
        "total_latency_ms": 1000.0,
        "ttft_ms": 250.0,
        "inter_token_latencies_ms": [250.0, 500.0],
        "mean_itl_ms": 375.0,
        "generated_tokens_per_second": 3.0,
        "token_availability_ms": [250.0, 500.0, 1000.0],
    }


def test_timing_starts_before_tokenization_and_finishes_before_text_decode(monkeypatch):
    events = []
    timestamps = iter([10.0, 10.25, 10.5, 11.0])

    def clock():
        events.append("timestamp")
        return next(timestamps)

    class RecordingTokenizer(FakeTokenizer):
        def __call__(self, prompt, return_tensors):
            events.append("tokenization")
            return super().__call__(prompt, return_tensors)

        def decode(self, token_ids, skip_special_tokens):
            events.append("text_decode")
            return super().decode(token_ids, skip_special_tokens)

    monkeypatch.setattr(benchmark.time, "perf_counter", clock)
    result = benchmark.timed_greedy_decode(FakeModel(), RecordingTokenizer(), "prompt", 3)
    assert events == ["timestamp", "tokenization", "timestamp", "timestamp", "timestamp", "text_decode", "text_decode"]
    assert result["total_latency_ms"] == 1000.0
    assert result["ttft_ms"] == 250.0


def test_cuda_sync_is_explicit_and_cpu_is_a_noop(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    benchmark.synchronize_device(torch.device("cpu"))
    synchronize.assert_not_called()
    benchmark.synchronize_device(torch.device("cuda:1"))
    synchronize.assert_called_once_with(torch.device("cuda:1"))


def test_warmup_excluded_from_statistics_and_result_serializes(monkeypatch):
    durations = iter([100.0, 0.125, 0.25])
    calls = []

    def fake_timed_decode(model, tokenizer, prompt, max_new_tokens):
        calls.append((prompt, max_new_tokens))
        return {
            "generated_token_ids": [6],
            **benchmark.calculate_timing_metrics(0.0, [next(durations)]),
        }

    monkeypatch.setattr(benchmark, "timed_greedy_decode", fake_timed_decode)
    result = benchmark.run_benchmark(FakeModel(), FakeTokenizer(), "prompt", 1, 1, 2, "fake")
    assert calls == [("prompt", 1)] * 3
    assert len(result["runs"]) == 2
    assert result["summary"]["mean_total_latency_ms"] == 187.5
    assert result["summary"]["min_total_latency_ms"] == 125.0
    assert result["summary"]["max_total_latency_ms"] == 250.0
    assert result["summary"]["mean_ttft_ms"] == 187.5
    assert result["summary"]["mean_itl_ms"] is None
    assert result["summary"]["mean_generated_tokens_per_second"] == 6.0
    assert result["summary"]["generated_token_ids_identical"] is True
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert set(restored) == {"benchmark", "environment", "config", "runs", "summary"}
    assert restored["config"]["use_cache"] is False
    assert restored["environment"]["cuda_max_memory_allocated_bytes"] is None


def test_zero_tokens_and_single_eos_handle_unavailable_intervals():
    model = FakeModel()
    zero = benchmark.timed_greedy_decode(model, FakeTokenizer(), "prompt", 0)
    assert model.calls == []
    assert zero["total_latency_ms"] == zero["generated_tokens_per_second"] == 0.0
    assert zero["ttft_ms"] is zero["mean_itl_ms"] is None
    assert zero["inter_token_latencies_ms"] == zero["generated_token_ids"] == []
    assert zero["total_input_token_positions_processed"] == 0
    assert benchmark.summarize_runs([zero])["mean_ttft_ms"] is None

    eos = benchmark.timed_greedy_decode(model, FakeTokenizer(eos_token_id=6), "prompt", 3)
    assert len(model.calls) == 1
    assert eos["generated_token_ids"] == [6]
    assert eos["generated_text"] == ""
    assert eos["stop_reason"] == "eos"
    assert eos["stopped_on_eos"] is True
    assert eos["inter_token_latencies_ms"] == []
    assert eos["mean_itl_ms"] is None


def test_summary_pools_itls_and_reports_token_discrepancy():
    first = {"generated_token_ids": [1, 2], **benchmark.calculate_timing_metrics(0, [1, 2])}
    second = {"generated_token_ids": [3, 4, 5], **benchmark.calculate_timing_metrics(0, [1, 3, 6])}
    summary = benchmark.summarize_runs([first, second])
    assert summary["mean_itl_ms"] == 2000.0
    assert summary["generated_token_ids_identical"] is False


@pytest.mark.parametrize("tokens, warmups, runs", [(-1, 0, 1), (1, -1, 1), (1, 0, 0)])
def test_invalid_benchmark_counts_fail_before_execution(tokens, warmups, runs):
    model = FakeModel()
    with pytest.raises(ValueError, match="must be"):
        benchmark.run_benchmark(model, FakeTokenizer(), max_new_tokens=tokens, warmup_runs=warmups, runs=runs)
    assert model.calls == []
