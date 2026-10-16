"""Record deterministic greedy engine semantics; no fabricated timestamps/benchmarks."""
import argparse
import json
from pathlib import Path

from llm_runtime.block_allocator import BlockAllocator
from llm_runtime.continuous_batcher import ContinuousBatcher
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.portfolio import ENGINE_COMMIT
from llm_runtime.request import RequestState


def record():
    model, tokenizer = load_model_and_tokenizer("distilgpt2", "cpu")
    pool = BlockAllocator(32, 4)
    engine = ContinuousBatcher(model, tokenizer, max_active=3, block_allocator=pool)
    specs = [("The capital of France is", 3), ("Hello", 6), ("The tallest mountain is", 4),
             ("A computer is", 2), ("The purpose of an operating system is", 5), ("In the morning", 3)]
    requests = [RequestState(prompt, budget, f"r{index+1}") for index, (prompt, budget) in enumerate(specs)]
    for state in requests:
        engine.add_request(state)
    traces = []
    while engine.has_work:
        trace = engine.step()
        trace["waiting_request_ids"] = [state.request_id for state in engine.waiting]
        trace["queue_depth"] = len(engine.waiting)
        for event in trace["events"]:
            state = engine.requests[event["request_id"]]
            event.update(token_index=state.generated_token_count,
                         token_text=tokenizer.decode([event["token_id"]], skip_special_tokens=True),
                         generated_text=tokenizer.decode(state.generated_token_ids, skip_special_tokens=True))
        traces.append(trace)
    assert pool.stats()["free_blocks"] == pool.total_blocks
    return {"label": "Recorded trace", "provenance": "Actual ContinuousBatcher + distilgpt2 CPU greedy forwards; optional logical BlockAllocator; not an HTTP capture or a timing benchmark",
            "engine_commit": ENGINE_COMMIT, "model": "distilgpt2", "device": "cpu", "max_active": 3,
            "serving_cache": False, "time_axis": "discrete engine ticks; replay cadence is not measured elapsed time",
            "requests": [{"request_id": state.request_id, "prompt": state.prompt, "max_new_tokens": state.max_new_tokens,
                          "prompt_token_ids": state.prompt_token_ids} for state in requests], "traces": traces}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("frontend/src/data/recorded-trace.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Recorded trace saved to {args.output}")
