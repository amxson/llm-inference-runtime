"""FastAPI service with one model lifetime and genuine token-level SSE delivery."""

import asyncio
from collections import deque
import copy
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from llm_runtime.continuous_batcher import ContinuousBatcher
from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from llm_runtime.request import RequestState
from llm_runtime.portfolio import ENGINE_COMMIT, benchmark_index, read_benchmark


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=10, ge=0, le=128)


class ServingRuntime:
    """One worker owns engine mutations; clients communicate through commands."""
    def __init__(self, model, tokenizer, max_active=4, max_pending=64):
        self.engine = ContinuousBatcher(model, tokenizer, max_active)
        self.tokenizer = tokenizer
        self.max_pending = max_pending
        self.commands = asyncio.Queue()
        self.channels = {}
        self.worker = None
        self.started = time.perf_counter()
        self.recent_traces = deque(maxlen=128)
        self.in_flight = False
        self._published = {}
        self._publish()

    def _publish(self):
        """Called only by the owning event loop, BETWEEN model forwards.

        Readers get a completed-boundary snapshot, not concurrently-mutating
        engine lists from the inference thread. Recent trace history is bounded.
        """
        self._published = {
            "tick": self.engine.tick,
            "active_request_ids": [state.request_id for state in self.engine.active],
            "waiting_request_ids": [state.request_id for state in self.engine.waiting],
            "queue_depth": len(self.engine.waiting),
            "batch_size": len(self.engine.active), "max_active": self.engine.max_active,
            "model_forward_count": self.engine.model_forward_count,
            "logical_allocator": None,
            "requests": [{"request_id": state.request_id, "status": state.status.value,
                          "generated_token_count": state.generated_token_count}
                         for state in self.engine.requests.values()],
            "traces": list(self.recent_traces),
        }

    def snapshot(self):
        # No await: only already-published plain data is read, even during a forward.
        return {**copy.deepcopy(self._published), "in_flight": self.in_flight,
                "snapshot_boundary": "last completed worker boundary",
                "engine": "continuous_uncached_greedy", "serving_cache": False}

    async def start(self):
        self.worker = asyncio.create_task(self._run())

    async def close(self):
        self.commands.put_nowait(("stop", None))
        await self.worker

    def submit(self, body: GenerateBody):
        if not body.prompt.strip():
            raise HTTPException(422, "Prompt must contain non-whitespace text.")
        if len(self.channels) >= self.max_pending:
            raise HTTPException(503, "Request capacity is full.")
        # Check the model's context boundary before returning streaming headers.
        count = self.tokenizer(body.prompt, return_tensors="pt")["input_ids"].shape[1]
        limit = getattr(getattr(self.engine.model, "config", None), "max_position_embeddings", None)
        if count == 0 or (limit is not None and count + body.max_new_tokens > limit):
            raise HTTPException(422, "Prompt and token budget exceed the supported context.")
        state = RequestState(body.prompt, body.max_new_tokens)
        channel = asyncio.Queue(maxsize=130)  # <=128 token events, error, done.
        self.channels[state.request_id] = channel
        self.commands.put_nowait(("add", state))
        return state, channel

    def release(self, request_id):
        self.channels.pop(request_id, None)
        self.commands.put_nowait(("cancel", request_id))

    def _complete(self, state):
        channel = self.channels.get(state.request_id)
        if channel is not None:
            channel.put_nowait(("done", state.to_result(self.tokenizer)))
        self.engine.forget_request(state.request_id)

    def _command(self, kind, value):
        if kind == "add":
            self.engine.add_request(value)
        elif kind == "cancel" and value in self.engine.requests:
            self.engine.cancel_request(value)
            self._complete(self.engine.requests[value])

    async def _run(self):
        while True:
            if not self.engine.has_work:
                kind, value = await self.commands.get()
            else:
                kind, value = "continue", None
            if kind == "stop":
                break
            self._command(kind, value)
            stopping = False
            while not self.commands.empty():
                kind, value = self.commands.get_nowait()
                if kind == "stop":
                    stopping = True
                    break
                self._command(kind, value)
            if stopping:
                break
            self._publish()
            if not self.engine.has_work:
                continue
            try:
                # The event loop remains responsive; only this worker executes forwards.
                step_start = time.perf_counter()
                self.in_flight = True
                trace = await asyncio.to_thread(self.engine.step)
                self.in_flight = False
                trace = copy.deepcopy(trace)
                trace["observed_at_ms"] = (time.perf_counter() - self.started) * 1000
                trace["worker_step_ms"] = (time.perf_counter() - step_start) * 1000
                trace["waiting_request_ids"] = [state.request_id for state in self.engine.waiting]
                trace["queue_depth"] = len(self.engine.waiting)
                for event in trace["events"]:
                    state = self.engine.requests[event["request_id"]]
                    channel = self.channels.get(state.request_id)
                    if channel is not None:
                        payload = {
                            "request_id": state.request_id,
                            "token_id": event["token_id"],
                            "token_index": state.generated_token_count,
                            "token_text": self.tokenizer.decode([event["token_id"]], skip_special_tokens=True),
                            "generated_text": self.tokenizer.decode(state.generated_token_ids, skip_special_tokens=True),
                            "tick": trace["tick"], "phase": event["phase"],
                        }
                        event.update(payload)
                        channel.put_nowait(("token", payload))
                self.recent_traces.append(trace)
                for request_id in trace["completed"]:
                    self._complete(self.engine.requests[request_id])
            except Exception:
                self.in_flight = False
                # A failed shared forward fails its active rows, not the worker/process.
                for state in list(self.engine.active):
                    if not state.is_finished:
                        state.finish("error")
                    channel = self.channels.get(state.request_id)
                    if channel is not None:
                        channel.put_nowait(("error", {"message": "Inference failed."}))
                    self._complete(state)
                self.engine.active.clear()
            self._publish()
            await asyncio.sleep(0)  # Publish events before scheduling the next forward.
        for state in list(self.engine.requests.values()):
            self.engine.cancel_request(state.request_id)
            self._complete(state)


def encode_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def create_app(model=None, tokenizer=None, *, loader=load_model_and_tokenizer,
               model_name=DEFAULT_MODEL_NAME, device="cpu", max_active=4,
               cors_origins=None, benchmark_directory=None) -> FastAPI:
    if (model is None) != (tokenizer is None) or max_active < 1:
        raise ValueError("Inject both model and tokenizer; max_active must be >= 1.")

    @asynccontextmanager
    async def lifespan(app):
        loaded_model, loaded_tokenizer = (loader(model_name, device) if model is None else (model, tokenizer))
        runtime = ServingRuntime(loaded_model, loaded_tokenizer, max_active)
        app.state.runtime = runtime
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="LLM Inference Runtime", lifespan=lifespan)
    origins = cors_origins if cors_origins is not None else [
        value.strip() for value in os.getenv("LLM_RUNTIME_CORS_ORIGINS", "").split(",") if value.strip()]
    if "*" in origins:
        raise ValueError("Configure explicit frontend CORS origins, not a wildcard.")
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    results_directory = (Path(benchmark_directory) if benchmark_directory is not None else
                         Path(__file__).resolve().parents[2] / "benchmarks" / "results")

    @app.get("/runtime/snapshot")
    async def runtime_snapshot():
        return {**app.state.runtime.snapshot(), "model": model_name,
                "device": str(next(app.state.runtime.engine.model.parameters()).device),
                "engine_commit": ENGINE_COMMIT}

    @app.get("/benchmarks")
    async def benchmarks():
        return benchmark_index(results_directory)

    @app.get("/benchmarks/{benchmark_name}")
    async def benchmark(benchmark_name: str):
        return read_benchmark(results_directory, benchmark_name)

    @app.get("/health")
    async def health():
        runtime = app.state.runtime
        return {
            "status": "ok", "model": model_name,
            "device": str(next(runtime.engine.model.parameters()).device),
            "engine": "continuous_uncached_greedy", "max_active": max_active,
            "model_forward_calls": runtime.engine.model_forward_count,
            "last_batch_size": len(runtime.engine.trace[-1]["model_request_ids"]) if runtime.engine.trace else 0,
            "max_observed_batch_size": max((len(item["model_request_ids"]) for item in runtime.engine.trace), default=0),
        }

    @app.post("/generate")
    async def generate(body: GenerateBody):
        runtime = app.state.runtime
        state, channel = runtime.submit(body)
        try:
            while True:
                event, data = await channel.get()
                if event == "done":
                    if data["stop_reason"] in {"error", "cancelled"}:
                        raise HTTPException(500, "Inference did not complete.")
                    return data
        finally:
            runtime.release(state.request_id)

    @app.post("/generate/stream")
    async def stream(body: GenerateBody):
        runtime = app.state.runtime
        state, channel = runtime.submit(body)

        async def events():
            try:
                while True:
                    event, data = await channel.get()
                    yield encode_sse(event, data)
                    if event == "done":
                        break
            finally:
                # StreamingResponse cancellation/disconnect reaches this cleanup.
                runtime.release(state.request_id)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
