"""Ragged full-attention DynamicCache collation; no scheduler or model policy."""

from dataclasses import dataclass

import torch
from torch.nn.functional import pad
from transformers import DynamicCache


def cache_tensors(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Read the iterable tensor protocol inspected in Transformers 5.17.0.

    The constructor/iterator avoid private layer mutation. Reject incompatible
    formats rather than silently dropping sliding-window or other metadata.
    """
    if not isinstance(cache, DynamicCache):
        raise ValueError("Expected a full-attention DynamicCache.")
    layers = []
    for entry in cache:
        if len(entry) not in (2, 3) or (len(entry) == 3 and entry[2] is not None):
            raise ValueError("Only full-attention DynamicCache layers are supported.")
        key, value = entry[:2]
        if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
                or key.ndim != 4 or key.shape != value.shape or min(key.shape) < 1
                or key.device != value.device or key.dtype != value.dtype):
            raise ValueError("Expected nonempty matching K/V tensors [B, H, T, D].")
        if layers and (key.shape[0] != layers[0][0].shape[0]
                       or key.shape[-2] != layers[0][0].shape[-2]):
            raise ValueError("Cache layers must agree on batch and history lengths.")
        layers.append((key, value))
    if not layers:
        raise ValueError("Cannot batch an empty cache.")
    return layers


@dataclass(frozen=True)
class CacheBatch:
    cache: DynamicCache
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    logical_lengths: tuple[int, ...]


def collate_caches(caches: list[DynamicCache]) -> CacheBatch:
    """Make a fresh padded cache and mask/positions for ONE new token per row."""
    if not caches:
        raise ValueError("At least one request cache is required.")
    rows = [cache_tensors(cache) for cache in caches]
    reference = rows[0]
    for row in rows:
        if len(row) != len(reference):
            raise ValueError("Request caches have different layer counts.")
        for (key, _), (ref, _) in zip(row, reference):
            if (key.shape[0] != 1 or key.shape[1::2] != ref.shape[1::2]
                    or key.device != ref.device or key.dtype != ref.dtype):
                raise ValueError("Request caches need compatible single-row layers.")
    lengths = tuple(row[0][0].shape[-2] for row in rows)
    width = max(lengths)
    layers = []
    for layer in range(len(reference)):
        layers.append(tuple(torch.cat([
            pad(row[layer][kv], (0, 0, width - length, 0))
            for row, length in zip(rows, lengths)
        ], dim=0) for kv in (0, 1)))
    device = reference[0][0].device
    positions = torch.tensor(lengths, device=device, dtype=torch.long)[:, None]
    columns = torch.arange(width + 1, device=device)[None, :]
    mask = (columns >= width - positions).long()
    return CacheBatch(DynamicCache(ddp_cache_data=layers), mask, positions, lengths)


def split_cache(cache: DynamicCache, logical_lengths: list[int]) -> list[DynamicCache]:
    """Keep each row's real suffix, cloning away batch storage and leading pads.

    For decode, pass *previous length + 1*. For prefill, pass prompt lengths.
    Selected output tokens have not yet entered the cache.
    """
    layers = cache_tensors(cache)
    batch, _, width, _ = layers[0][0].shape
    if (len(logical_lengths) != batch or any(
            not isinstance(length, int) or isinstance(length, bool)
            or length < 1 or length > width for length in logical_lengths)):
        raise ValueError("Provide one positive logical length <= physical width per row.")
    return [DynamicCache(ddp_cache_data=[
        (key[row:row + 1, :, -length:, :].clone(),
         value[row:row + 1, :, -length:, :].clone())
        for key, value in layers
    ]) for row, length in enumerate(logical_lengths)]
