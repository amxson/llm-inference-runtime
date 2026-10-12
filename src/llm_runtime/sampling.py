"""Explicit temperature scaling, candidate filtering, and token sampling."""

import math

import torch


def validate_sampling_parameters(
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> None:
    """Reject invalid settings, including when no tokens will be generated."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0.")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be None or an integer >= 1.")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be None or satisfy 0 < top_p <= 1.")


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Rescale preferences; lower temperatures sharpen the distribution."""
    validate_sampling_parameters(temperature=temperature)
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    """Keep exactly k highest-scoring entries of a vocabulary vector."""
    validate_sampling_parameters(top_k=top_k)
    if top_k is None:
        return logits.clone()

    k = min(top_k, logits.numel())
    values, indices = torch.topk(logits, k=k)
    filtered_logits = torch.full_like(logits, -torch.inf)
    return filtered_logits.scatter(0, indices, values)


def apply_top_p(logits: torch.Tensor, top_p: float | None = None) -> torch.Tensor:
    """Keep the smallest sorted probability prefix that reaches top_p."""
    validate_sampling_parameters(top_p=top_p)
    if top_p is None or top_p == 1.0:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    remove = cumulative_probabilities >= top_p
    # Shift the mask so the token that reaches/crosses the threshold stays.
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    original_order_remove = torch.zeros_like(remove).scatter(0, sorted_indices, remove)
    return logits.masked_fill(original_order_remove, -torch.inf)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token from a final-position vocabulary vector [V]."""
    if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating-point vocabulary vector [V].")
    if (
        torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any()
    ):
        raise ValueError("logits must contain a finite candidate and no NaN or positive infinity.")

    # Order matters: nucleus mass is computed after temperature and top-k.
    scaled_logits = apply_temperature(logits, temperature)
    filtered_logits = apply_top_k(scaled_logits, top_k)
    filtered_logits = apply_top_p(filtered_logits, top_p)
    probabilities = torch.softmax(filtered_logits, dim=-1)
    selected = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return int(selected.item())
