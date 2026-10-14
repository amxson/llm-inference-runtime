"""FIFO admission into a changing, bounded, uncached active batch."""

from collections import deque

from llm_runtime.block_allocator import BlockAllocator
from llm_runtime.batching import batch_step
from llm_runtime.request import RequestState, RequestStatus
from llm_runtime.scheduler import prepare_request


class ContinuousBatcher:
    def __init__(self, model, tokenizer, max_active: int = 4,
                 block_allocator: BlockAllocator | None = None):
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
        self.block_allocator = block_allocator

    def _release_blocks(self, state: RequestState) -> None:
        if self.block_allocator is not None and self.block_allocator.owns(state.request_id):
            self.block_allocator.free(state.request_id)
        state.kv_block_ids.clear()

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
        block_snapshot = None
        try:
            if self.block_allocator is not None:
                for state in self.active:
                    if state.status is RequestStatus.WAITING:
                        prepare_request(state, self.tokenizer, next(self.model.parameters()).device)
                    if not state.is_finished:
                        # A selected token gets KV storage when it is forwarded next tick.
                        state.kv_block_ids = self.block_allocator.reserve(
                            state.request_id, state.logical_context_length)
                block_snapshot = self.block_allocator.stats()
            events = batch_step(self.model, self.tokenizer, self.active)
        except Exception:
            if self.block_allocator is not None:
                # Fail the affected cohort and release even partially reserved ownership.
                for state in self.active:
                    if not state.is_finished:
                        state.finish("error")
                    self._release_blocks(state)
            raise
        completed = [state.request_id for state in self.active if state.is_finished]
        for state in self.active:
            if state.is_finished:
                self._release_blocks(state)
        self.active = [state for state in self.active if not state.is_finished]
        self.model_forward_count += int(bool(events))
        trace = {
            "tick": self.tick, "admitted": admitted, "active_request_ids": active_ids,
            "model_request_ids": [event["request_id"] for event in events],
            "completed": completed, "events": events,
        }
        if block_snapshot is not None:
            trace["logical_kv_before_forward"] = block_snapshot
            trace["logical_kv_after_completion"] = self.block_allocator.stats()
        self.trace.append(trace)
        return trace

    def cancel_request(self, request_id: str) -> None:
        state = self.requests[request_id]
        if not state.is_finished:
            state.finish("cancelled")
        self._release_blocks(state)
        self.waiting = deque(item for item in self.waiting if item.request_id != request_id)
        self.active = [item for item in self.active if item.request_id != request_id]

    def forget_request(self, request_id: str) -> None:
        """Let a serving owner release completed history after delivering the result."""
        if not self.requests[request_id].is_finished:
            raise ValueError("Only completed requests can be forgotten.")
        self._release_blocks(self.requests[request_id])
        del self.requests[request_id]

    def run_until_complete(self) -> list[dict]:
        while self.has_work:
            self.step()
        return [state.to_result(self.tokenizer) for state in self.requests.values()]
