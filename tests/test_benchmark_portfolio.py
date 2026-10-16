"""Download-free checks for finalization measurement integrity."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location("benchmark_portfolio", Path(__file__).parents[1] / "scripts/benchmark_portfolio.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_distribution_has_explicit_interpolated_quantiles_and_empty_policy():
    result = benchmark.distribution([1, 2, 3, 4, 5])
    assert result["count"] == 5
    assert result["median"] == 3
    assert result["p90"] == pytest.approx(4.6)
    assert result["p95"] == pytest.approx(4.8)
    assert result["max"] == 5
    assert benchmark.distribution([])["median"] is None


def test_receive_sse_rejects_completion_without_matching_tokens():
    def handler(request):
        return httpx.Response(200, headers={"content-type":"text/event-stream"},
            text='event: done\ndata: {"request_id":"r","generated_token_ids":[1]}\n\n')
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="mismatch"):
            benchmark.receive_sse(client, "http://test", {"max_new_tokens":1})


def test_committed_measurement_is_internally_consistent():
    artifact = json.loads((Path(__file__).parents[1] / "benchmarks/results/final_cpu.json").read_text(encoding="utf8"))
    paired = artifact["paired"]
    ids = paired["uncached"]["runs"][0]["generated_token_ids"]
    assert len(ids) == 128
    for mode in ("uncached", "cached"):
        assert len(paired[mode]["runs"]) == 5
        for run in paired[mode]["runs"]:
            assert run["generated_token_ids"] == ids
            assert len(run["inter_token_latencies_ms"]) == 127
            assert len(run["token_availability_ms"]) == 128
            assert all(gap >= 0 for gap in run["inter_token_latencies_ms"])
    values = []
    for run in artifact["network_runs"]:
        events = run["events"]
        tokens = [e for e in events if e["event"] == "token"]
        done = [e for e in events if e["event"] == "done"]
        assert len(tokens) == 128 and len(done) == 1
        assert not any(e["event"] == "error" for e in events)
        assert [e["data"]["token_index"] for e in tokens] == list(range(1,129))
        assert [e["data"]["token_id"] for e in tokens] == done[0]["data"]["generated_token_ids"] == ids
        assert len(run["worker_steps"]) == 128
        values.extend(s["worker_step_ms"] for s in run["worker_steps"])
    assert artifact["worker_step_ms"] == benchmark.distribution(values)
