"""FIFO continuous greedy batching with compact request-local GPT-2 KV caches."""

from collections import deque

import torch

from llm_runtime.cache_batch import collate_caches, split_cache
from llm_runtime.request import RequestState, RequestStatus
from llm_runtime.scheduler import prepare_request


class CachedContinuousBatcher:
    def __init__(self, model, tokenizer, max_active: int = 4):
        if max_active < 1:
            raise ValueError("max_active must be >= 1.")
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.max_active = max_active
        self.waiting, self.active, self.requests = deque(), [], {}
        self.trace = deque(maxlen=1024)
        self.tick = self.model_forward_count = 0

    @property
    def has_work(self):
        return bool(self.waiting or self.active)

    def add_request(self, state: RequestState):
        if state.request_id in self.requests or state.status is not RequestStatus.WAITING:
            raise ValueError("Submit a new waiting request with a unique ID.")
        self.requests[state.request_id] = state
        self.waiting.append(state)

    def cancel_request(self, request_id):
        state = self.requests[request_id]
        if not state.is_finished:
            state.finish("cancelled")
        self.waiting = deque(s for s in self.waiting if s.request_id != request_id)
        self.active = [s for s in self.active if s.request_id != request_id]

    def forget_request(self, request_id):
        if not self.requests[request_id].is_finished:
            raise ValueError("Only completed requests can be forgotten.")
        del self.requests[request_id]

    def _prefill_inputs(self, states):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Batch padding needs a PAD or EOS token ID.")
        width = max(len(s.prompt_token_ids) for s in states)
        ids = torch.full((len(states), width), pad_id, device=self.device, dtype=torch.long)
        mask, types = torch.zeros_like(ids), torch.zeros_like(ids)
        with_types = all("token_type_ids" in s.runtime_inputs for s in states)
        for row, state in enumerate(states):
            n = len(state.prompt_token_ids)
            ids[row, -n:] = state.runtime_inputs["input_ids"][0]
            original_mask = state.runtime_inputs.get("attention_mask")
            # The tokenizer is called once per unpadded prompt. Reject holes so
            # compact length always means actual forwarded prompt tokens.
            if original_mask is not None and not bool(original_mask.bool().all()):
                raise ValueError("Expected an unpadded, fully attended request prompt.")
            mask[row, -n:] = 1
            if with_types:
                types[row, -n:] = state.runtime_inputs["token_type_ids"][0]
        inputs = {"input_ids": ids, "attention_mask": mask,
                  "position_ids": (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)}
        if with_types:
            inputs["token_type_ids"] = types
        return inputs

    def _forward(self, states, phase):
        before = {s.request_id: 0 if phase == "prefill" else s.kv_cache.get_seq_length()
                  for s in states}
        lengths = [s.logical_context_length for s in states]
        if phase == "prefill":
            inputs = self._prefill_inputs(states)
        else:
            for s in states:
                if before[s.request_id] != s.logical_context_length - 1:
                    raise ValueError("Cache length must exclude the newest selected token.")
            batch = collate_caches([s.kv_cache for s in states])
            inputs = {"input_ids": torch.tensor([[s.generated_token_ids[-1]] for s in states],
                                                device=self.device, dtype=torch.long),
                      "past_key_values": batch.cache, "attention_mask": batch.attention_mask,
                      "position_ids": batch.position_ids}
            if all("token_type_ids" in s.runtime_inputs for s in states):
                inputs["token_type_ids"] = torch.cat([
                    s.runtime_inputs["token_type_ids"][:, -1:] for s in states])
        self.model_forward_count += 1
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=True)
            caches = split_cache(output.past_key_values, lengths)
            selected = output.logits[:, -1].argmax(-1).tolist()
        events = []
        for state, token, cache, length in zip(states, selected, caches, lengths):
            state.kv_cache = cache
            state.record_token(token, self.tokenizer.eos_token_id)  # finish releases cache
            events.append({"request_id": state.request_id, "phase": phase, "token_id": token,
                           "logical_context_length": length,
                           "logical_cache_length_before": before[state.request_id],
                           "logical_cache_length_after": length,
                           "model_input_width": inputs["input_ids"].shape[1],
                           "batch_size": len(states), "serving_cache": True})
        return events

    def step(self):
        admitted = []
        while self.waiting and len(self.active) < self.max_active:
            state = self.waiting.popleft()
            self.active.append(state)
            admitted.append(state.request_id)
        if not self.active:
            return None
        self.tick += 1
        active_ids = [s.request_id for s in self.active]
        forwards_before = self.model_forward_count
        events = []
        try:
            for state in self.active:
                if state.status is RequestStatus.WAITING:
                    prepare_request(state, self.tokenizer, self.device)
                limit = getattr(self.model.config, "max_position_embeddings", None)
                if limit is not None and len(state.prompt_token_ids) + state.max_new_tokens > limit:
                    raise ValueError("Prompt and token budget exceed the model's context limit.")
            # Partition ONCE: prefill may change status, but newcomers cannot
            # also enter this tick's decode forward.
            prefills = [s for s in self.active if s.status is RequestStatus.PREFILL]
            decodes = [s for s in self.active if s.status is RequestStatus.DECODING]
            for phase, rows in (("prefill", prefills), ("decode", decodes)):
                if rows:
                    events.extend(self._forward(rows, phase))
        except Exception:
            # Leave finished rows in active for the serving owner to deliver
            # sanitized error/done events. All cache ownership is released now.
            for state in self.active:
                if not state.is_finished:
                    state.finish("error")
            raise
        completed = [s.request_id for s in self.active if s.is_finished]
        self.active = [s for s in self.active if not s.is_finished]
        frame = {"tick": self.tick, "admitted": admitted, "active_request_ids": active_ids,
                 "model_request_ids": [e["request_id"] for e in events],
                 "completed": completed, "events": events, "serving_cache": True,
                 "model_forward_count_this_tick": self.model_forward_count - forwards_before,
                 "logical_cache_lengths_before": {e["request_id"]: e["logical_cache_length_before"] for e in events},
                 "logical_cache_lengths_after": {e["request_id"]: e["logical_cache_length_after"] for e in events}}
        for phase in ("prefill", "decode"):
            group = [e for e in events if e["phase"] == phase]
            frame[phase + "_request_ids"] = [e["request_id"] for e in group]
            frame[phase + "_input_shape"] = [len(group), group[0]["model_input_width"]] if group else None
        self.trace.append(frame)
        return frame

    def run_until_complete(self):
        while self.has_work:
            self.step()
        return [s.to_result(self.tokenizer) for s in self.requests.values()]
