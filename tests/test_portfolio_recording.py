"""Validate recorded ownership/lifecycle against real engine classes, without downloads."""
import json
from pathlib import Path

from llm_runtime.block_allocator import BlockAllocator
from llm_runtime.request import RequestState


def test_recording_matches_request_lifecycle_and_allocator_ownership():
    recording = json.loads((Path(__file__).resolve().parents[1] / "frontend/src/data/recorded-trace.json").read_text())
    pool = BlockAllocator(32, 4)
    states = {item["request_id"]: RequestState(item["prompt"], item["max_new_tokens"], item["request_id"])
              for item in recording["requests"]}
    metadata = {item["request_id"]: item for item in recording["requests"]}
    waiting, active = list(states), []
    for trace in recording["traces"]:
        admitted = []
        while waiting and len(active) < recording["max_active"]:
            request_id = waiting.pop(0)
            admitted.append(request_id)
            active.append(request_id)
            states[request_id].start_prefill(metadata[request_id]["prompt_token_ids"])
        assert trace["admitted"] == admitted
        assert trace["active_request_ids"] == active
        assert trace["waiting_request_ids"] == waiting
        for event in trace["events"]:
            state = states[event["request_id"]]
            assert state.logical_context_length == event["logical_context_length"]
            pool.reserve(state.request_id, state.logical_context_length)
        assert pool.stats() == trace["logical_kv_before_forward"]
        for event in trace["events"]:
            state = states[event["request_id"]]
            state.record_token(event["token_id"], 50256)
            assert event["token_index"] == state.generated_token_count
        completed = [request_id for request_id in active if states[request_id].is_finished]
        assert trace["completed"] == completed
        for request_id in completed:
            pool.free(request_id)
            active.remove(request_id)
        assert pool.stats() == trace["logical_kv_after_completion"]
    assert not active and not waiting
    assert pool.stats()["allocated_blocks"] == 0
