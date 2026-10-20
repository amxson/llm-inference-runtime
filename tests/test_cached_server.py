import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from fastapi.testclient import TestClient

from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.server import GenerateBody, ServingRuntime, create_app
from test_cache_batch import tiny_gpt2
from test_cached_continuous_batcher import TinyTokenizer, SPECS
from test_server import parse_events


def test_overlapping_http_sse_clients_use_shared_cached_decode_and_dynamic_admission():
    model, tokenizer = tiny_gpt2(), TinyTokenizer()
    expected = [cached_greedy_decode(model, tokenizer, p, n)["generated_token_ids"] for p, n, _ in SPECS]
    entered, advance, all_submitted = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def before_forward(_, args, kwargs):
        calls.append(list(kwargs["input_ids"].shape))
        if len(calls) == 1:
            entered.set()
            assert advance.wait(15), "First prefill was not released"

    hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)
    app = create_app(model, tokenizer, max_active=2, engine="continuous_cached_greedy")
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=4) as pool:
        runtime = app.state.runtime
        original_submit = runtime.submit
        submitted = []

        def submit(body):
            result = original_submit(body)
            submitted.append(result[0].request_id)
            if len(submitted) == 4:
                all_submitted.set()
            return result

        runtime.submit = submit
        def send(spec):
            prompt, budget, _ = spec
            return client.post("/generate/stream", json={"prompt": prompt, "max_new_tokens": budget})
        futures = [pool.submit(send, SPECS[0])]
        try:
            assert entered.wait(15)
            futures.extend(pool.submit(send, spec) for spec in SPECS[1:])
            assert all_submitted.wait(15)
            # Polls must remain safe during a cached model forward too.
            for _ in range(10):
                health = client.get("/health").json()
                snapshot = client.get("/runtime/snapshot").json()
                assert health["engine"] == "continuous_cached_greedy" and health["serving_cache"]
                assert snapshot["in_flight"] and snapshot["model_forward_count"] == 0
            advance.set()
            for future, ids in zip(futures, expected):
                response = future.result(30)
                assert response.status_code == 200
                events = parse_events(response.text)
                assert [kind for kind, _ in events] == ["token"] * len(ids) + ["done"]
                tokens = [data for kind, data in events if kind == "token"]
                assert [e["token_id"] for e in tokens] == ids
                assert [e["token_index"] for e in tokens] == list(range(1, len(ids) + 1))
                assert [e["generated_text"] for e in tokens] == [tokenizer.decode(ids[:i]) for i in range(1, len(ids) + 1)]
                assert events[-1][1]["generated_token_ids"] == ids
            snapshot = client.get("/runtime/snapshot").json()
            assert isinstance(runtime.engine, CachedContinuousBatcher)
            traces = snapshot["traces"]
            assert any(t["admitted"] and t["decode_request_ids"] for t in traces)
            assert any(t["decode_input_shape"] == [2, 1] for t in traces)
            assert all(t["decode_input_shape"] is None or t["decode_input_shape"][1] == 1 for t in traces)
            assert snapshot["requests"] == [] and not runtime.engine.active
            assert snapshot["max_observed_batch_size"] == 2
        finally:
            advance.set()
            hook.remove()


def test_cached_cancellation_releases_local_cache_and_worker_survives():
    async def scenario():
        model = tiny_gpt2()
        gate, entered = threading.Event(), threading.Event()
        count = 0
        def before_forward(_, args, kwargs):
            nonlocal count
            count += 1
            if count == 2:
                entered.set()
                assert gate.wait(10)
        hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        runtime = ServingRuntime(model, TinyTokenizer(), engine="continuous_cached_greedy")
        await runtime.start()
        state, channel = runtime.submit(GenerateBody(prompt="1 2 3", max_new_tokens=5))
        try:
            assert (await asyncio.wait_for(channel.get(), 10))[0] == "token"
            assert await asyncio.to_thread(entered.wait, 10)
            assert state.kv_cache is not None
            runtime.release(state.request_id)  # Same cleanup path as SSE disconnect.
            gate.set()
        finally:
            gate.set()
            await runtime.close()
            hook.remove()
        assert state.stop_reason == "cancelled" and state.kv_cache is None
        assert not runtime.engine.requests
    asyncio.run(scenario())


def test_engine_configuration_cached_default_zero_error_and_context(monkeypatch):
    tokenizer, model = TinyTokenizer(), tiny_gpt2()
    monkeypatch.delenv("LLM_RUNTIME_ENGINE", raising=False)
    with TestClient(create_app(model, tokenizer)) as client:
        assert client.get("/health").json()["serving_cache"] is True
        zero = parse_events(client.post("/generate/stream", json={"prompt": "1 2", "max_new_tokens": 0}).text)
        assert [kind for kind, _ in zero] == ["done"]
        assert client.post("/generate", json={"prompt": " ".join(["1"] * 63), "max_new_tokens": 2}).status_code == 422
        original = model.forward
        def broken(*args, **kwargs):
            raise RuntimeError("private model error")
        model.forward = broken
        response = client.post("/generate/stream", json={"prompt": "1 2", "max_new_tokens": 2})
        assert [kind for kind, _ in parse_events(response.text)] == ["error", "done"]
        assert "private model error" not in response.text
        model.forward = original
        assert client.post("/generate", json={"prompt": "1 2", "max_new_tokens": 2}).status_code == 200
    monkeypatch.setenv("LLM_RUNTIME_ENGINE", "continuous_uncached_greedy")
    with TestClient(create_app(model, tokenizer)) as client:
        assert client.get("/health").json()["serving_cache"] is False
    with TestClient(create_app(model, tokenizer, engine="continuous_cached_greedy")) as client:
        assert client.get("/health").json()["serving_cache"] is True
    with pytest.raises(ValueError, match="Unknown engine"):
        create_app(engine="invented")
