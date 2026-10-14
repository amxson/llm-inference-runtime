"""FIFO admission into a changing, bounded, uncached active batch."""

from collections import deque

from llm_runtime.batching import batch_step
from llm_runtime.request import RequestState, RequestStatus


class ContinuousBatcher:
    def __init__(self, model, tokenizer, max_active: int = 4):
        if max_active < 1:
            raise ValueError("max_active must be >= 1.")
        self.model, self.tokenizer = model, tokenizer
        self.max_active = max_active
        self.waiting = deque()
        self.active = []
        self.requests = {}
        self.trace = deque(maxlen=1024)
        self.tick = 0
        self.model_forward_count = 0

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.active)

    def add_request(self, state: RequestState) -> None:
        if state.request_id in self.requests or state.status is not RequestStatus.WAITING:
            raise ValueError("Submit a new waiting request with a unique ID.")
        self.requests[state.request_id] = state
        self.waiting.append(state)

    def step(self) -> dict | None:
        """Admit at capacity and advance all current rows by one selected token."""
        admitted = []
        while self.waiting and len(self.active) < self.max_active:
            state = self.waiting.popleft()
            self.active.append(state)
            admitted.append(state.request_id)
        if not self.active:
            return None
        self.tick += 1
        active_ids = [state.request_id for state in self.active]
        events = batch_step(self.model, self.tokenizer, self.active)
        completed = [state.request_id for state in self.active if state.is_finished]
        self.active = [state for state in self.active if not state.is_finished]
        self.model_forward_count += int(bool(events))
        trace = {
            "tick": self.tick, "admitted": admitted, "active_request_ids": active_ids,
            "model_request_ids": [event["request_id"] for event in events],
            "completed": completed, "events": events,
        }
        self.trace.append(trace)
        return trace

    def cancel_request(self, request_id: str) -> None:
        state = self.requests[request_id]
        if not state.is_finished:
            state.finish("cancelled")
        self.waiting = deque(item for item in self.waiting if item.request_id != request_id)
        self.active = [item for item in self.active if item.request_id != request_id]

    def forget_request(self, request_id: str) -> None:
        """Let a serving owner release completed history after delivering the result."""
        if not self.requests[request_id].is_finished:
            raise ValueError("Only completed requests can be forgotten.")
        del self.requests[request_id]

    def run_until_complete(self) -> list[dict]:
        while self.has_work:
            self.step()
        return [state.to_result(self.tokenizer) for state in self.requests.values()]
