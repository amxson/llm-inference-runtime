"""Real distilgpt2 CPU over a loopback TCP server: concurrent SSE and disconnect."""

from concurrent.futures import ThreadPoolExecutor
import json
import socket
import threading
import time

import httpx
import torch
import uvicorn

from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.server import create_app


def read_stream(base, prompt, budget):
    frames, kind = [], None
    with httpx.Client(timeout=120) as client, client.stream(
        "POST", base + "/generate/stream", json={"prompt": prompt, "max_new_tokens": budget}
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            if line.startswith("data: "):
                frames.append({"event": kind, "data": json.loads(line[6:])})
    return frames


def check():
    torch.set_num_threads(4)
    model, tokenizer = load_model_and_tokenizer("distilgpt2", "cpu")
    specs = [("The capital of France is", 3), ("An operating system manages", 6),
             ("In the early morning, the city", 5), ("A computer network connects", 4)]
    expected = [cached_greedy_decode(model, tokenizer, p, n)["generated_token_ids"] for p, n in specs]
    app = create_app(model, tokenizer, max_active=2, engine="continuous_cached_greedy")
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{listener.getsockname()[1]}"
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("Loopback server failed to start")
                time.sleep(.01)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(read_stream, base, p, n) for p, n in specs]
                streams = [f.result() for f in futures]
            for frames, ids in zip(streams, expected):
                tokens = [f["data"] for f in frames if f["event"] == "token"]
                assert [f["token_id"] for f in tokens] == ids
                assert [f["token_index"] for f in tokens] == list(range(1, len(ids) + 1))
                assert [f["generated_text"] for f in tokens] == [tokenizer.decode(ids[:i], skip_special_tokens=True) for i in range(1, len(ids) + 1)]
                assert frames[-1]["event"] == "done" and not any(f["event"] == "error" for f in frames)
            with httpx.Client(timeout=120) as client:
                health = client.get(base + "/health").json()
                snapshot = client.get(base + "/runtime/snapshot").json()
                assert health["serving_cache"] and snapshot["serving_cache"]
                assert any(t["decode_input_shape"] == [2, 1] for t in snapshot["traces"])
                assert any(t["admitted"] and t["decode_request_ids"] for t in snapshot["traces"])
                assert [f["event"] for f in read_stream(base, "Hello", 0)] == ["done"]
                with client.stream("POST", base + "/generate/stream", json={"prompt": "Hello", "max_new_tokens": 128}) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            cancelled_id = json.loads(line[6:])["request_id"]
                            break
                deadline = time.monotonic() + 15
                while client.get(base + "/runtime/snapshot").json()["requests"]:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Disconnected request was not released")
                    time.sleep(.02)
                assert not app.state.runtime.engine.requests
            return {"model": "distilgpt2", "health": health, "exact_token_parity": True,
                    "streams": streams, "concurrent_traces": snapshot["traces"],
                    "zero_tokens": "done only", "disconnect_request_released": cancelled_id}
        finally:
            server.should_exit = True
            thread.join(30)
            assert not thread.is_alive()


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
