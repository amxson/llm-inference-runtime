import pytest
import torch
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

from llm_runtime.cache_batch import cache_tensors, collate_caches, split_cache


def make_cache(length, marker):
    return DynamicCache(ddp_cache_data=[
        (torch.full((1, 2, length, 4), float(marker + layer)),
         torch.full((1, 2, length, 4), float(-marker - layer)))
        for layer in range(2)
    ])


@pytest.mark.parametrize("lengths", [[3, 3], [3, 8, 5], [1]])
def test_collate_split_positions_padding_order_and_ownership(lengths):
    originals = [make_cache(n, i + 1) for i, n in enumerate(lengths)]
    batched = collate_caches(originals)
    assert batched.logical_lengths == tuple(lengths)
    assert batched.position_ids.tolist() == [[n] for n in lengths]
    assert batched.attention_mask.tolist() == [
        [0] * (max(lengths) - n) + [1] * (n + 1) for n in lengths
    ]
    split = split_cache(batched.cache, lengths)
    for original, restored in zip(originals, split):
        assert restored.get_seq_length() == original.get_seq_length()
        for (key, value), (rkey, rvalue) in zip(cache_tensors(original), cache_tensors(restored)):
            torch.testing.assert_close(key, rkey)
            torch.testing.assert_close(value, rvalue)
            assert key.data_ptr() != rkey.data_ptr()
    cache_tensors(split[0])[0][0].fill_(777)
    assert cache_tensors(originals[0])[0][0].max() == 1
    assert cache_tensors(batched.cache)[0][0].max() <= len(lengths)
    if len(split) > 1:
        assert cache_tensors(split[1])[0][0].max() == 2


def test_documented_row_selection_then_reorder_does_not_mix_histories():
    batched = collate_caches([make_cache(3, 10), make_cache(8, 20), make_cache(5, 30)])
    batched.cache.batch_select_indices(torch.tensor([2, 0]))
    restored = split_cache(batched.cache, [5, 3])
    assert [c.get_seq_length() for c in restored] == [5, 3]
    assert [cache_tensors(c)[0][0].unique().item() for c in restored] == [30, 10]


def test_one_forward_extends_each_logical_history_not_selected_token():
    batched = collate_caches([make_cache(3, 10), make_cache(8, 20)])
    for layer in range(2):
        new = torch.ones(2, 2, 1, 4) * 99
        batched.cache.update(new, new, layer)
    restored = split_cache(batched.cache, [4, 9])
    assert [cache.get_seq_length() for cache in restored] == [4, 9]
    assert cache_tensors(restored[0])[0][0][0, 0, :, 0].tolist() == [10, 10, 10, 99]


@pytest.mark.parametrize("caches", [[], [None], [DynamicCache()], [make_cache(0, 1)]])
def test_empty_and_invalid_groups_fail(caches):
    with pytest.raises(ValueError):
        collate_caches(caches)


def test_incompatible_shapes_sliding_metadata_and_split_lengths_fail():
    bad = DynamicCache(ddp_cache_data=[(torch.zeros(2, 2, 3, 4), torch.zeros(2, 2, 3, 4))])
    for group in [[bad], [make_cache(2, 1), bad]]:
        with pytest.raises(ValueError):
            collate_caches(group)
    sliding = DynamicCache(ddp_cache_data=[
        (torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4), torch.tensor(8))])
    with pytest.raises(ValueError, match="full-attention"):
        collate_caches([sliding])
    for lengths in [[], [0], [4], [True]]:
        with pytest.raises(ValueError):
            split_cache(make_cache(3, 1), lengths)


def tiny_gpt2():
    torch.manual_seed(42)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=64, n_embd=16, n_layer=2, n_head=2,
        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
    )).eval()


def test_real_tiny_gpt2_ragged_cached_forward_matches_independent_logits():
    model = tiny_gpt2()  # Random local weights: no downloads.
    prompts = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])]
    with torch.inference_mode():
        prefills = [model(ids, use_cache=True) for ids in prompts]
        next_ids = torch.cat([out.logits[:, -1].argmax(-1)[:, None] for out in prefills])
        batch = collate_caches([out.past_key_values for out in prefills])
        output = model(next_ids, past_key_values=batch.cache, attention_mask=batch.attention_mask,
                       position_ids=batch.position_ids, use_cache=True)
        restored = split_cache(output.past_key_values, [4, 9])
        for row, prefill in enumerate(prefills):
            independent = model(next_ids[row:row + 1], past_key_values=prefill.past_key_values,
                                use_cache=True)
            torch.testing.assert_close(output.logits[row], independent.logits[0], atol=1e-6, rtol=1e-5)
            for (key, value), (rkey, rvalue) in zip(
                    cache_tensors(independent.past_key_values), cache_tensors(restored[row])):
                torch.testing.assert_close(key, rkey, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(value, rvalue, atol=1e-6, rtol=1e-5)
