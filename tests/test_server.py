import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import queue
import threading

from fastapi.testclient import TestClient

from llm_runtime.server import GenerateBody, ServingRuntime, create_app
from runtime_fakes import FakeModel, FakeTokenizer


def parse_events(text):
    events = []
    for frame in text.strip().split("\n\n"):
        lines = frame.splitlines()
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


def test_health_validation_generation_sse_and_lifespan_model_reuse():
    model, tokenizer = FakeModel(), FakeTokenizer()
    loads = []
    def loader(name, device):
        loads.append((name, device))
        return model, tokenizer
    app = create_app(loader=loader)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        first = client.post("/generate", json={"prompt": "a", "max_new_tokens": 3})
        assert first.status_code == 200
        assert first.json()["generated_token_ids"] == [3, 4, 5]
        stream = client.post("/generate/stream", json={"prompt": "b", "max_new_tokens": 2})
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = parse_events(stream.text)
        assert [name for name, _ in events] == ["token", "token", "done"]
        assert [data["token_id"] for name, data in events if name == "token"] == [5, 6]
        assert events[-1][1]["generated_token_count"] == 2
        zero = client.post("/generate/stream", json={"prompt": "a", "max_new_tokens": 0})
        assert [name for name, _ in parse_events(zero.text)] == ["done"]
        for body in [{"prompt": ""}, {"prompt": " "}, {"prompt": "a", "max_new_tokens": -1}, {"prompt": "a", "temperature": 1.0}]:
            assert client.post("/generate", json=body).status_code == 422
        assert loads == [("distilgpt2", "cpu")]
    assert app.state.runtime.worker.done()
    assert not app.state.runtime.engine.requests


def test_token_event_is_available_before_generation_completes_and_cancel_is_serialized():
    gate = threading.Event()
    class GatedModel(FakeModel):
        def __call__(self, **kwargs):
            if len(self.calls) == 1:
                assert gate.wait(timeout=5)
            return super().__call__(**kwargs)

    async def scenario():
        model = GatedModel()
        runtime = ServingRuntime(model, FakeTokenizer())
        await runtime.start()
        state, channel = runtime.submit(GenerateBody(prompt="a", max_new_tokens=5))
        try:
            event, data = await asyncio.wait_for(channel.get(), timeout=5)
            assert event == "token" and data["token_id"] == 3
            assert len(model.calls) == 1 and not state.is_finished
            runtime.release(state.request_id)
            gate.set()
        finally:
            gate.set()
            await runtime.close()
        assert not runtime.engine.requests
        assert state.stop_reason == "cancelled"
    asyncio.run(scenario())


def test_model_error_emits_error_and_done_and_worker_keeps_serving(caplog):
    class BrokenModel(FakeModel):
        fail = True
        def __call__(self, **kwargs):
            if self.fail:
                raise RuntimeError("private diagnostic should not reach clients")
            return super().__call__(**kwargs)
    model = BrokenModel()
    app = create_app(model, FakeTokenizer())
    with caplog.at_level(logging.ERROR, logger="llm_runtime.server"), TestClient(app) as client:
        response = client.post("/generate/stream", json={"prompt": "a"})
        events = parse_events(response.text)
        assert [name for name, _ in events] == ["error", "done"]
        assert "private diagnostic" not in response.text
        assert events[0][1] == {"message": "Inference failed."}
        assert client.get("/health").status_code == 200
        failure = client.post("/generate", json={"prompt": "a"})
        assert failure.status_code == 500
        assert "private diagnostic" not in failure.text
        assert not app.state.runtime.worker.done()
        model.fail = False
        assert client.post("/generate", json={"prompt": "a", "max_new_tokens": 2}).status_code == 200
    records = [r for r in caplog.records if r.name == "llm_runtime.server"]
    assert len(records) == 2
    assert all(r.levelno == logging.ERROR and r.exc_info for r in records)
    assert all(isinstance(r.exc_info[1], RuntimeError) for r in records)
    assert "private diagnostic" in caplog.text


def test_health_polls_published_state_during_actual_inference():
    """Hold three real worker forwards; a live-trace read fails deterministically."""
    entered = queue.Queue()
    advance = threading.Semaphore(0)
    active = threading.Event()

    class GatedModel(FakeModel):
        def __call__(self, **kwargs):
            active.set()
            entered.put(True)
            try:
                assert advance.acquire(timeout=10), "Test did not release forward"
                return super().__call__(**kwargs)
            finally:
                active.clear()

    class OwnerTrace(deque):
        def __iter__(self):
            assert not active.is_set(), "Reader traversed live inference trace"
            return super().__iter__()

        def __getitem__(self, key):
            assert not active.is_set(), "Reader indexed live inference trace"
            return super().__getitem__(key)

    app = create_app(GatedModel(), FakeTokenizer())
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        app.state.runtime.engine.trace = OwnerTrace(maxlen=1024)
        request = pool.submit(client.post, "/generate", json={"prompt": "a", "max_new_tokens": 3})
        try:
            for completed in range(3):
                entered.get(timeout=10)
                for _ in range(20):
                    response = client.get("/health")
                    assert response.status_code == 200
                    health = response.json()
                    assert health["status"] == "ok"
                    assert health["model"] == "distilgpt2" and health["device"] == "cpu"
                    assert health["engine"] == "continuous_uncached_greedy"
                    assert health["max_active"] == 4
                    assert health["model_forward_calls"] == completed
                    assert health["last_batch_size"] == int(completed > 0)
                    assert health["max_observed_batch_size"] == int(completed > 0)
                advance.release()
            assert request.result(timeout=10).status_code == 200
            assert client.get("/health").json()["model_forward_calls"] == 3
        finally:
            for _ in range(3):
                advance.release()
