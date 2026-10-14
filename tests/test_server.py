import asyncio
import json
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


def test_model_error_emits_error_and_done_and_worker_keeps_serving():
    class BrokenModel(FakeModel):
        def __call__(self, **kwargs):
            raise RuntimeError("private diagnostic should not reach clients")
    with TestClient(create_app(BrokenModel(), FakeTokenizer())) as client:
        response = client.post("/generate/stream", json={"prompt": "a"})
        events = parse_events(response.text)
        assert [name for name, _ in events] == ["error", "done"]
        assert "private diagnostic" not in response.text
        assert client.get("/health").status_code == 200
        assert client.post("/generate", json={"prompt": "a"}).status_code == 500
