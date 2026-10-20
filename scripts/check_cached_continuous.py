"""Explicit real-weight CPU parity check; not part of the download-free suite."""

import json

import torch

from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.request import RequestState


def check():
    torch.set_num_threads(4)
    model, tokenizer = load_model_and_tokenizer("distilgpt2", "cpu")
    specs = [("The capital of France is", 2), ("An operating system manages", 5),
             ("In the early morning, the city", 5), ("A computer network connects", 4)]
    engine = CachedContinuousBatcher(model, tokenizer, 2)
    for index, (prompt, budget) in enumerate(specs):
        engine.add_request(RequestState(prompt, budget, chr(65 + index)))
    engine.run_until_complete()
    for state in engine.requests.values():
        expected = cached_greedy_decode(model, tokenizer, state.prompt, state.max_new_tokens)
        assert state.generated_token_ids == expected["generated_token_ids"], state.request_id
        assert state.kv_cache is None
    assert any(t["decode_request_ids"] == ["B", "C"] for t in engine.trace)
    assert any(t["decode_request_ids"] == ["C", "D"] for t in engine.trace)
    return {"model": "distilgpt2", "device": "cpu", "exact_token_parity": True,
            "results": [s.to_result(tokenizer) for s in engine.requests.values()], "traces": list(engine.trace)}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
