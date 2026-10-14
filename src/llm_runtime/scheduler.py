"""FIFO admission and round-robin cached decoding: one request per forward."""

from collections import deque

import torch

from llm_runtime.request import RequestState, RequestStatus


def prepare_request(state: RequestState, tokenizer, device: torch.device) -> None:
    encoded = tokenizer(state.prompt, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in encoded.items() if isinstance(value, torch.Tensor)}
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Request tokenization must produce input_ids [1, T].")
    state.start_prefill(ids[0].tolist(), inputs)


class DecodeScheduler:
    def __init__(self, model, tokenizer, max_active: int = 4):
        if max_active < 1:
            raise ValueError("max_active must be >= 1.")
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.max_active = max_active
        self.waiting = deque()
        self.active = deque()
        self.requests = {}
        self.admission_order = []
        self.trace = []
        self.tick = 0

    def add_request(self, state: RequestState) -> None:
        if state.request_id in self.requests or state.status is not RequestStatus.WAITING:
            raise ValueError("Submit a new waiting request with a unique ID.")
        self.requests[state.request_id] = state
        self.waiting.append(state)

    def step(self) -> dict | None:
        """Advance one request by at most one model forward and one selected token."""
        while self.waiting and len(self.active) < self.max_active:
            admitted = self.waiting.popleft()
            self.active.append(admitted)
            self.admission_order.append(admitted.request_id)
        if not self.active:
            return None
        state = self.active.popleft()
        self.tick += 1
        phase = "prefill" if state.status is RequestStatus.WAITING else "decode"
        selected = None
        input_length = 0
        try:
            if state.status is RequestStatus.WAITING:
                prepare_request(state, self.tokenizer, self.device)
            logical_length = state.logical_context_length
            if not state.is_finished:
                input_length = state.runtime_inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    outputs = self.model(**state.runtime_inputs, past_key_values=state.kv_cache, use_cache=True)
                    state.kv_cache = outputs.past_key_values
                    if state.kv_cache is None:
                        raise RuntimeError("The model did not return a KV cache.")
                    next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                    selected = int(next_id.item())
                    state.runtime_inputs["input_ids"] = next_id.unsqueeze(-1)
                    if "attention_mask" in state.runtime_inputs:
                        mask = state.runtime_inputs["attention_mask"]
                        state.runtime_inputs["attention_mask"] = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
                    if "token_type_ids" in state.runtime_inputs:
                        state.runtime_inputs["token_type_ids"] = state.runtime_inputs["token_type_ids"][:, -1:]
                    state.record_token(selected, self.tokenizer.eos_token_id)
        except Exception:
            if not state.is_finished:
                state.finish("error")
            raise
        if not state.is_finished:
            self.active.append(state)
        event = {
            "tick": self.tick, "request_id": state.request_id, "phase": phase,
            "model_input_length": input_length, "logical_context_length": logical_length,
            "token_id": selected,
        }
        self.trace.append(event)
        return event

    def run_until_complete(self) -> list[dict]:
        while self.waiting or self.active:
            self.step()
        return [state.to_result(self.tokenizer) for state in self.requests.values()]
