"""Real, randomly initialized Qwen2 CPU tests: architecture, not GPU evidence."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from fastapi.testclient import TestClient

from llm_runtime.model_adapter import model_adapter, eos_token_ids
from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.greedy_decode import greedy_decode
from llm_runtime.request import RequestState
from llm_runtime.server import create_app
from test_cached_continuous_batcher import TinyTokenizer, SPECS
from test_server import parse_events


def tiny_qwen():
    torch.manual_seed(17)
    config = Qwen2Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_qwen_real_gqa_rotary_ragged_dynamic_cache_parity(capacity):
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    specs = SPECS + [("3 4 5", 4, "equal"), ("9", 0, "zero")]
    engine = CachedContinuousBatcher(model, tokenizer, capacity)
    for prompt, budget, name in specs:
        engine.add_request(RequestState(prompt, budget, name))
    engine.run_until_complete()
    for state in engine.requests.values():
        ref = cached_greedy_decode(model, tokenizer, state.prompt, state.max_new_tokens)
        manual = greedy_decode(model, tokenizer, state.prompt, state.max_new_tokens)
        assert state.generated_token_ids == ref["generated_token_ids"] == manual["generated_token_ids"]
        assert state.kv_cache is None
    decodes = [f for f in engine.trace if f["decode_input_shape"]]
    assert all(f["decode_input_shape"][1] == 1 for f in decodes)
    if capacity == 2:
        assert any(f["admitted"] and f["decode_request_ids"] for f in engine.trace)


def test_qwen_positions_segment_ids_and_explicit_multiple_eos():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    adapter = model_adapter(model)
    inputs = {"input_ids": torch.tensor([[5], [6]]),
              "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
              "token_type_ids": torch.zeros(2, 1, dtype=torch.long)}
    prepared = adapter.forward_inputs(inputs)
    assert prepared["position_ids"].tolist() == [[2], [3]]
    assert prepared["logits_to_keep"] == 1 and "token_type_ids" not in prepared
    assert "token_type_ids" in inputs  # No caller-owned mutation.
    first = cached_greedy_decode(model, tokenizer, "1 2 3", 1)["generated_token_ids"][0]
    model.generation_config.eos_token_id = [47, first]
    assert eos_token_ids(model, tokenizer) == (47, first)
    engine = CachedContinuousBatcher(model, tokenizer, 2)
    state = RequestState("1 2 3", 5)
    engine.add_request(state)
    engine.run_until_complete()
    assert state.stop_reason == "eos" and state.generated_token_count == 1
    assert cached_greedy_decode(model, tokenizer, state.prompt, 5)["generated_token_ids"] == [first]


def test_sliding_qwen_is_rejected_not_silently_compacted():
    with pytest.raises(ValueError, match="full-attention"):
        model_adapter(SimpleNamespace(config=SimpleNamespace(
            model_type="qwen2", layer_types=["sliding_attention"])))


def test_qwen_http_sse_zero_and_live_token_ids():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    expected = cached_greedy_decode(model, tokenizer, "1 2 3", 4)["generated_token_ids"]
    with TestClient(create_app(model, tokenizer, model_name="tiny-random-qwen2")) as client:
        assert client.get("/health").json()["serving_cache"]
        assert client.post("/generate", json={"prompt": "1 2 3", "max_new_tokens": 4}).json()["generated_token_ids"] == expected
        events = parse_events(client.post("/generate/stream", json={
            "prompt": "1 2 3", "max_new_tokens": 4}).text)
        assert [data["token_id"] for kind, data in events if kind == "token"] == expected
        zero = client.post("/generate", json={"prompt": "1", "max_new_tokens": 0}).json()
        assert zero["generated_token_ids"] == [] and zero["model_execution_count"] == 0
