import json
from types import SimpleNamespace

import pytest
import torch

from llm_runtime.greedy_decode import greedy_decode
from llm_runtime.speculative_decode import speculative_greedy_decode, verify_compatibility
from llm_runtime.benchmark_speculative import run_comparison
from runtime_fakes import FakeTokenizer


class SequenceModel(torch.nn.Module):
    """Logits at EVERY position, so verification sees true next-token predictions."""
    def __init__(self, wrong_at=None, limit=64):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=16, max_position_embeddings=limit)
        self.wrong_at = wrong_at
        self.calls = []

    def forward(self, input_ids, use_cache, **kwargs):
        assert not use_cache and torch.is_inference_mode_enabled()
        self.calls.append(input_ids.tolist())
        selected = (input_ids + 1) % 16
        if self.wrong_at is not None:
            selected = torch.where(input_ids == self.wrong_at, (selected + 3) % 16, selected)
        logits = torch.zeros(*input_ids.shape, 16).scatter_(-1, selected.unsqueeze(-1), 10)
        return SimpleNamespace(logits=logits)


class CompatibleTokenizer(FakeTokenizer):
    backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps({"model": "fake", "normalizer": None}))

    def get_vocab(self):
        return {str(i): i for i in range(16)}


@pytest.mark.parametrize("wrong_at,k,budget,prompt", [(None, 3, 6, "a"), (4, 3, 7, "a"),
                                                    (2, 4, 5, "a"), (None, 4, 2, "a"),
                                                    (None, 3, 10, "eos"), (8, 3, 10, "eos"),
                                                    (None, 1, 5, "b"), (None, 3, 0, "a")])
def test_exact_target_parity_across_accept_mismatch_eos_and_budget(wrong_at, k, budget, prompt):
    target, draft, tokenizer = SequenceModel(), SequenceModel(wrong_at), FakeTokenizer()
    result = speculative_greedy_decode(target, draft, tokenizer, prompt, budget, k)
    expected = greedy_decode(SequenceModel(), tokenizer, prompt, budget)
    assert result["generated_token_ids"] == expected["generated_token_ids"]
    assert result["stop_reason"] == expected["stop_reason"]
    assert result["proposed_draft_tokens"] == result["accepted_draft_tokens"] + result["rejected_draft_tokens"]
    assert result["target_verification_calls"] == len(target.calls)
    assert result["draft_forward_calls"] == len(draft.calls)
    if budget == 0:
        assert not target.calls and not draft.calls and result["ttft_ms"] is None


def test_matching_prefix_correction_and_bonus_trace():
    result = speculative_greedy_decode(SequenceModel(), SequenceModel(4), FakeTokenizer(), "a", 7, 3)
    first, second = result["trace"]
    assert first["draft_token_ids"] == [3, 4, 8]
    assert first["accepted_count"] == 2 and first["mismatch_index"] == 2
    assert first["emitted_token_ids"] == [3, 4, 5]
    assert second["draft_token_ids"] == [6, 7, 8]
    assert second["emitted_token_ids"] == [6, 7, 8, 9]  # Bonus is EOS.
    assert result["rejected_draft_tokens"] == 1


def test_context_boundary_can_verify_without_draft_room():
    result = speculative_greedy_decode(SequenceModel(limit=2), SequenceModel(limit=2), FakeTokenizer(), "a", 1, 3)
    assert result["generated_token_ids"] == [3] and result["draft_forward_calls"] == 0
    with pytest.raises(ValueError, match="context"):
        speculative_greedy_decode(SequenceModel(limit=2), SequenceModel(limit=2), FakeTokenizer(), "a", 2)


@pytest.mark.parametrize("budget,k", [(-1, 3), (1, 0)])
def test_invalid_budgets(budget, k):
    with pytest.raises(ValueError):
        speculative_greedy_decode(SequenceModel(), SequenceModel(), FakeTokenizer(), "a", budget, k)


def test_compatibility_checks_id_mapping_specials_and_bpe():
    target, draft, tokenizer = SequenceModel(), SequenceModel(), CompatibleTokenizer()
    assert verify_compatibility(target, draft, tokenizer, tokenizer)["exact_vocab_id_mapping"]
    wrong = CompatibleTokenizer()
    wrong.get_vocab = lambda: {"other": 0}
    with pytest.raises(ValueError, match="vocabularies"):
        verify_compatibility(target, draft, tokenizer, wrong)
    wrong = CompatibleTokenizer()
    wrong.eos_token_id = 8
    with pytest.raises(ValueError, match="special"):
        verify_compatibility(target, draft, tokenizer, wrong)
    wrong = CompatibleTokenizer()
    wrong.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps({"model": "different"}))
    with pytest.raises(ValueError, match="BPE"):
        verify_compatibility(target, draft, tokenizer, wrong)


def test_paired_benchmark_enforces_parity_and_records_order():
    tokenizer = CompatibleTokenizer()
    result = run_comparison(SequenceModel(), SequenceModel(4), tokenizer, tokenizer,
                            "a", 5, 3, 0, 2, "fake-target", "fake-draft")
    assert result["comparison"]["exact_token_id_parity"]
    assert result["config"]["measured_execution_order"] == [["ordinary", "speculative"], ["speculative", "ordinary"]]


def test_paired_benchmark_refuses_invalid_parity(monkeypatch):
    import llm_runtime.benchmark_speculative as benchmark
    real = benchmark.speculative_greedy_decode
    def corrupted(*args, **kwargs):
        result = real(*args, **kwargs)
        result["generated_token_ids"] = [-1]
        return result
    monkeypatch.setattr(benchmark, "speculative_greedy_decode", corrupted)
    tokenizer = CompatibleTokenizer()
    with pytest.raises(ValueError, match="parity"):
        run_comparison(SequenceModel(), SequenceModel(), tokenizer, tokenizer, "a", 2, 3, 0, 1)
