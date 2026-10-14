"""Correct synchronous greedy batches, recollated without a combined KV cache."""

import torch

from llm_runtime.request import RequestState, RequestStatus
from llm_runtime.scheduler import prepare_request


def batch_step(model, tokenizer, states: list[RequestState]) -> list[dict]:
    """Advance every unfinished row once in one real multi-request forward."""
    device = next(model.parameters()).device
    for state in states:
        if state.status is RequestStatus.WAITING:
            prepare_request(state, tokenizer, device)
    active = [state for state in states if not state.is_finished]
    if not active:
        return []
    contexts = [state.prompt_token_ids + state.generated_token_ids for state in active]
    context_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if context_limit is not None and any(len(ids) > context_limit for ids in contexts):
        raise ValueError("A request exceeds the model's context limit.")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Batch padding needs a PAD or EOS token ID.")
    width = max(map(len, contexts))
    input_ids = torch.full((len(active), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids)
    token_types = torch.zeros_like(input_ids)
    has_token_types = all("token_type_ids" in state.runtime_inputs for state in active)
    for row, (state, ids) in enumerate(zip(active, contexts)):
        input_ids[row, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        prompt_mask = state.runtime_inputs.get("attention_mask")
        original_mask = prompt_mask[0] if prompt_mask is not None else torch.ones(len(state.prompt_token_ids), dtype=torch.long, device=device)
        full_mask = torch.cat([original_mask, original_mask.new_ones(state.generated_token_count)])
        mask[row, -len(ids):] = full_mask
        if has_token_types:
            original_types = state.runtime_inputs["token_type_ids"][0]
            full_types = torch.cat([original_types, original_types[-1:].expand(state.generated_token_count)])
            token_types[row, -len(ids):] = full_types
    # Left padding puts every active row's final real token at the last column.
    positions = (mask.cumsum(dim=-1) - 1).masked_fill(mask == 0, 0)
    inputs = {"input_ids": input_ids, "attention_mask": mask, "position_ids": positions}
    if has_token_types:
        inputs["token_type_ids"] = token_types
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
        selected = torch.argmax(outputs.logits[:, -1, :], dim=-1).tolist()
    events = []
    for state, token_id, ids in zip(active, selected, contexts):
        phase = "prefill" if state.status is RequestStatus.PREFILL else "decode"
        state.record_token(token_id, tokenizer.eos_token_id)
        events.append({
            "request_id": state.request_id, "phase": phase, "token_id": token_id,
            "logical_context_length": len(ids), "model_input_width": width,
            "batch_size": len(active),
        })
    return events


def batched_greedy_decode(model, tokenizer, states: list[RequestState]) -> dict:
    """Run one fixed cohort; finished rows leave, and no new requests join."""
    if len({state.request_id for state in states}) != len(states):
        raise ValueError("Batch request IDs must be unique.")
    if any(state.status is not RequestStatus.WAITING for state in states):
        raise ValueError("Submit fresh waiting requests.")
    trace = []
    while any(not state.is_finished for state in states):
        events = batch_step(model, tokenizer, states)
        if events:
            trace.append({"tick": len(trace) + 1, "events": events})
    return {
        "results": [state.to_result(tokenizer) for state in states],
        "trace": trace,
        "model_forward_call_count": len(trace),
        "use_cache": False,
    }
