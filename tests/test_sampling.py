import pytest
import torch

from llm_runtime.sampling import (
    apply_temperature, apply_top_k, apply_top_p, sample_next_token,
)


def test_temperature_scales_logits_without_mutating_them():
    logits = torch.tensor([1.0, -2.0, 3.0])
    assert torch.equal(apply_temperature(logits, 1.0), logits)
    assert torch.equal(apply_temperature(logits, 0.5), torch.tensor([2.0, -4.0, 6.0]))
    assert torch.equal(apply_temperature(logits, 2.0), torch.tensor([0.5, -1.0, 1.5]))
    assert torch.equal(logits, torch.tensor([1.0, -2.0, 3.0]))


def test_top_k_keeps_exact_candidates_and_clamps_large_k():
    logits = torch.tensor([1.0, 5.0, 2.0, 4.0])
    filtered = apply_top_k(logits, 2)
    assert torch.equal(filtered, torch.tensor([-torch.inf, 5.0, -torch.inf, 4.0]))
    assert torch.equal(apply_top_k(logits, 99), logits)
    assert torch.equal(apply_top_k(logits), logits)
    assert torch.isfinite(apply_top_k(torch.ones(4), 2)).sum().item() == 2
    assert torch.equal(logits, torch.tensor([1.0, 5.0, 2.0, 4.0]))


def test_top_p_keeps_threshold_crossing_token_and_restores_vocabulary_order():
    # Sorted probabilities are .5, .3, .15, .05: .7 needs the first two.
    logits = torch.log(torch.tensor([0.15, 0.5, 0.05, 0.3], dtype=torch.float64))
    filtered = apply_top_p(logits, 0.7)
    assert torch.equal(torch.isfinite(filtered), torch.tensor([False, True, False, True]))
    assert torch.equal(filtered[[1, 3]], logits[[1, 3]])
    assert torch.equal(apply_top_p(logits, 1.0), logits)
    assert torch.equal(apply_top_p(logits), logits)
    assert torch.isfinite(logits).all()


def test_top_p_keeps_one_token_for_tiny_nucleus_and_handles_exact_boundary():
    logits = torch.log(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64))
    assert torch.equal(torch.isfinite(apply_top_p(logits, 0.01)), torch.tensor([True, False, False]))
    assert torch.equal(torch.isfinite(apply_top_p(logits, 0.5)), torch.tensor([True, False, False]))


@pytest.mark.parametrize("settings, message", [
    ({"temperature": 0}, "temperature"),
    ({"temperature": -1}, "temperature"),
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": float("inf")}, "temperature"),
    ({"top_k": 0}, "top_k"),
    ({"top_k": -1}, "top_k"),
    ({"top_k": 1.5}, "top_k"),
    ({"top_k": True}, "top_k"),
    ({"top_p": 0}, "top_p"),
    ({"top_p": -0.1}, "top_p"),
    ({"top_p": 1.1}, "top_p"),
    ({"top_p": float("nan")}, "top_p"),
])
def test_invalid_parameters_raise_clear_errors(settings, message):
    with pytest.raises(ValueError, match=message):
        sample_next_token(torch.tensor([1.0, 2.0]), **settings)


def test_sampling_pipeline_passes_expected_probabilities_and_generator(monkeypatch):
    logits = torch.log(torch.tensor([0.5, 0.25, 0.15, 0.1], dtype=torch.float64))
    generator = torch.Generator().manual_seed(42)
    received = {}

    def fake_multinomial(probabilities, num_samples, generator):
        received.update(probabilities=probabilities, num_samples=num_samples, generator=generator)
        return torch.tensor([1])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    assert sample_next_token(logits, 2.0, 3, 0.7, generator) == 1
    # Temperature sqrt-transforms the weights; top-k removes .1; nucleus keeps
    # the first two of the renormalized remaining probabilities.
    weights = torch.sqrt(torch.tensor([0.5, 0.25], dtype=torch.float64))
    expected = torch.cat([weights / weights.sum(), torch.zeros(2, dtype=torch.float64)])
    assert torch.allclose(received["probabilities"], expected)
    assert received["num_samples"] == 1
    assert received["generator"] is generator


def test_sampling_is_reproducible_with_local_generator():
    logits = torch.tensor([1.0, 2.0, 0.0, 3.0])

    def draw_sequence():
        generator = torch.Generator().manual_seed(42)
        return [sample_next_token(logits, 0.8, 3, 0.95, generator) for _ in range(12)]

    assert draw_sequence() == draw_sequence()


def test_top_k_one_always_selects_highest_logit():
    logits = torch.tensor([1.0, 5.0, 2.0, 4.0])
    for seed in [0, 42, 123]:
        assert sample_next_token(logits, top_k=1, generator=torch.Generator().manual_seed(seed)) == 1


@pytest.mark.parametrize("logits", [
    torch.tensor([]), torch.tensor([[1.0, 2.0]]), torch.tensor([1, 2]),
    torch.tensor([float("nan"), 1.0]), torch.tensor([float("inf"), 1.0]),
    torch.tensor([-torch.inf, -torch.inf]),
])
def test_invalid_logit_vectors_fail_clearly(logits):
    with pytest.raises(ValueError, match="logits must"):
        sample_next_token(logits)
