import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from llm_runtime.portfolio import BENCHMARK_FILES
from llm_runtime.server import GenerateBody, ServingRuntime, create_app
from runtime_fakes import FakeModel, FakeTokenizer


def test_read_only_snapshot_and_benchmark_allowlist():
    with TestClient(create_app(FakeModel(), FakeTokenizer())) as client:
        before = client.get("/runtime/snapshot").json()
        assert before["tick"] == 0 and before["serving_cache"] is False
        assert before["logical_allocator"] is None
        client.post("/generate", json={"prompt": "a", "max_new_tokens": 2})
        after = client.get("/runtime/snapshot").json()
        assert after["model_forward_count"] == 2
        assert [trace["events"][0]["token_id"] for trace in after["traces"]] == [3, 4]
        assert after["traces"][0]["events"][0]["phase"] == "prefill"
        assert after["traces"][1]["events"][0]["phase"] == "decode"
        assert after["traces"][0]["worker_step_ms"] >= 0
        assert after["traces"][1]["observed_at_ms"] >= after["traces"][0]["observed_at_ms"]
        serialized = json.dumps(after)
        assert "runtime_inputs" not in serialized and "kv_cache" not in serialized
        assert {item["name"] for item in client.get("/benchmarks").json()["artifacts"]} == set(BENCHMARK_FILES)
        assert client.get("/benchmarks/m7_cached_vs_uncached_cpu.json").json()["comparison"]["exact_token_id_parity"]
        assert client.get("/benchmarks/pyproject.toml").status_code == 404
        assert client.get("/benchmarks/not-installed.json").status_code == 404


def test_explicit_cors_configuration(monkeypatch):
    monkeypatch.setenv("LLM_RUNTIME_CORS_ORIGINS", "http://localhost:5173")
    with TestClient(create_app(FakeModel(), FakeTokenizer())) as client:
        headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
        assert client.options("/generate/stream", headers=headers).headers["access-control-allow-origin"] == headers["Origin"]
        headers["Origin"] = "https://untrusted.example"
        assert "access-control-allow-origin" not in client.options("/generate/stream", headers=headers).headers
    with pytest.raises(ValueError, match="wildcard"):
        create_app(cors_origins=["*"])


def test_snapshot_during_forward_is_a_stable_published_boundary():
    entered, release = threading.Event(), threading.Event()
    class Gated(FakeModel):
        def __call__(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().__call__(**kwargs)
    async def scenario():
        runtime = ServingRuntime(Gated(), FakeTokenizer())
        await runtime.start()
        state, channel = runtime.submit(GenerateBody(prompt="a", max_new_tokens=1))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            snapshot = runtime.snapshot()
            assert snapshot["in_flight"] and snapshot["tick"] == 0
            assert snapshot["traces"] == []
            snapshot["active_request_ids"].clear()
            assert runtime.snapshot()["active_request_ids"] == []  # Still waiting at published boundary.
            assert runtime.snapshot()["waiting_request_ids"] == [state.request_id]
            release.set()
            assert (await asyncio.wait_for(channel.get(), 5))[0] == "token"
        finally:
            release.set()
            runtime.release(state.request_id)
            await runtime.close()
    asyncio.run(scenario())


def test_missing_benchmark_installation_returns_empty_index(tmp_path):
    with TestClient(create_app(FakeModel(), FakeTokenizer(), benchmark_directory=tmp_path)) as client:
        assert client.get("/benchmarks").json()["artifacts"] == []
        assert client.get("/benchmarks/m7_cached_vs_uncached_cpu.json").status_code == 404


def test_telemetry_history_is_bounded_and_completed_requests_are_released():
    with TestClient(create_app(FakeModel(), FakeTokenizer())) as client:
        for _ in range(45):
            assert client.post("/generate", json={"prompt": "a", "max_new_tokens": 3}).status_code == 200
        snapshot = client.get("/runtime/snapshot").json()
        assert snapshot["tick"] == 135 and len(snapshot["traces"]) == 128
        assert snapshot["requests"] == [] and snapshot["queue_depth"] == 0
        assert snapshot["traces"][0]["tick"] == 8
