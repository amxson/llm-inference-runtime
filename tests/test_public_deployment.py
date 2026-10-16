"""Deployment routing and real app contract, without a model download."""
import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient
from runtime_fakes import FakeModel, FakeTokenizer


def test_public_prefix_health_cors_snapshot_and_stream(monkeypatch):
    monkeypatch.setenv("LLM_RUNTIME_CORS_ORIGINS", "https://llm-inference-runtime.vercel.app")
    spec = importlib.util.spec_from_file_location("public_entrypoint", Path(__file__).parents[1] / "api/index.py")
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    loads = []
    def loader(name, device):
        loads.append((name, device))
        return FakeModel(), FakeTokenizer()
    with TestClient(entrypoint.create_public_app(loader=loader)) as client:
        health = client.get("/api/health")
        assert health.headers["content-type"].startswith("application/json")
        assert health.json()["status"] == "ok"
        assert health.json()["max_active"] == 1
        cors = {"Origin":"https://llm-inference-runtime.vercel.app", "Access-Control-Request-Method":"POST", "Access-Control-Request-Headers":"content-type"}
        assert client.options("/api/generate/stream", headers=cors).headers["access-control-allow-origin"] == cors["Origin"]
        cors["Origin"] = "https://untrusted.example"
        assert "access-control-allow-origin" not in client.options("/api/generate/stream",headers=cors).headers
        result = client.post("/api/generate/stream",json={"prompt":"a","max_new_tokens":2})
        assert result.headers["content-type"].startswith("text/event-stream")
        assert result.text.count("event: token") == 2
        assert result.text.count("event: done") == 1
        assert "event: error" not in result.text
        assert client.get("/api/runtime/snapshot").json()["tick"] == 2
        assert client.get("/api/not-an-api").status_code == 404
    assert loads == [("distilgpt2", "cpu")]
