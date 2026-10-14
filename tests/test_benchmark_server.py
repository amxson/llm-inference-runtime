import asyncio
import json

import httpx
import pytest

from llm_runtime.benchmark_server import aggregate_requests, measure_request, percentile, run_benchmark


def record(success, latency, ttft, tokens):
    return {"success": success, "total_latency_ms": latency, "ttft_ms": ttft,
            "generated_token_count": tokens, "generated_token_ids": list(range(tokens))}


def test_percentiles_use_linear_interpolation_and_handle_empty_single_values():
    assert percentile([40, 10, 30, 20], 50) == 25
    assert percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)
    assert percentile([10, 20, 30, 40], 99) == pytest.approx(39.7)
    assert percentile([], 95) is None
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_aggregation_failure_handling_throughput_and_serialization():
    records = [record(True, 100, 20, 4), record(True, 200, 40, 4), record(False, 300, None, 1)]
    summary = aggregate_requests(records, 2)
    assert summary["total_requests"] == 3 and summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1 and summary["failed_partial_tokens"] == 1
    assert summary["aggregate_generated_tokens"] == 8
    assert summary["requests_per_second"] == 1 and summary["generated_tokens_per_second"] == 4
    assert summary["mean_latency_ms"] == 150 and summary["mean_ttft_ms"] == 30
    assert summary["p50_latency_ms"] == 150
    json.dumps(summary, allow_nan=False)
    failed = aggregate_requests([record(False, 10, None, 0)], 1)
    assert failed["mean_latency_ms"] is None and failed["generated_tokens_per_second"] == 0
    assert failed["p95_ttft_ms"] is None


def test_sse_measurement_and_failed_http_status_are_preserved():
    done = {"stop_reason": "max_new_tokens", "generated_token_ids": [3], "generated_token_count": 1}
    frames = f'event: token\ndata: {{"token_id":3}}\n\nevent: done\ndata: {json.dumps(done)}\n\n'
    async def scenario():
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(200, text=frames))) as client:
            result = await measure_request(client, {"prompt": "a"}, 0)
            assert result["success"] and result["generated_token_ids"] == [3]
            assert 0 <= result["ttft_ms"] <= result["total_latency_ms"]
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
            failed = await measure_request(client, {"prompt": "a"}, 1)
            assert not failed["success"] and failed["http_status"] == 503
            assert failed["error_type"] == "HTTPStatusError"
    asyncio.run(scenario())


def test_concurrency_limit_is_real_and_result_schema_serializes(monkeypatch):
    from llm_runtime import benchmark_server as module
    active, maximum = 0, 0
    async def fake_measure(client, body, index):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        await asyncio.sleep(0)
        active -= 1
        return record(True, 100, 20, 1)
    monkeypatch.setattr(module, "measure_request", fake_measure)
    original_client = httpx.AsyncClient
    health = {"model_forward_calls": 0, "model": "fake", "device": "cpu"}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=health))
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    result = asyncio.run(run_benchmark("http://test", "a", 1, [2], 4, 0))
    assert maximum == 2 and result["levels"][0]["summary"]["successful_requests"] == 4
    assert result["config"]["transport"] == "network_http_sse"
    json.dumps(result, allow_nan=False)
