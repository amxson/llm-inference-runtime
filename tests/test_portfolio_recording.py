"""Validate cached recording lifecycle/invariants without downloading weights."""
import json
from pathlib import Path

from llm_runtime.request import RequestState


def test_recording_matches_request_lifecycle_and_logical_cache_lengths():
    recording = json.loads((Path(__file__).resolve().parents[1] / "frontend/src/data/recorded-trace.json").read_text())
    assert recording["serving_cache"] is True
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
            expected_before = 0 if event["phase"] == "prefill" else state.logical_context_length - 1
            assert event["logical_cache_length_before"] == expected_before
            assert event["logical_cache_length_after"] == state.logical_context_length
            if event["phase"] == "decode":
                assert event["model_input_width"] == 1
        for event in trace["events"]:
            state = states[event["request_id"]]
            state.record_token(event["token_id"], 50256)
            assert event["token_index"] == state.generated_token_count
        completed = [request_id for request_id in active if states[request_id].is_finished]
        assert trace["completed"] == completed
        for request_id in completed:
            active.remove(request_id)
    assert not active and not waiting
    assert any(t["admitted"] and t["decode_request_ids"] for t in recording["traces"])
    assert any(t["decode_input_shape"] and t["decode_input_shape"][0] > 1 for t in recording["traces"])
