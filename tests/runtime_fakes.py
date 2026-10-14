"""Small deterministic tokenizer/model shared by runtime integration tests."""

from types import SimpleNamespace

import torch


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = None
    padding_side = "right"

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        ids = {"a": [1, 2], "b": [4], "c": [7, 1, 2], "eos": [8]}.get(prompt, [1, 2])
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "token_type_ids": torch.zeros(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens):
        return "".join(f" t{token}" for token in ids if not skip_special_tokens or token != self.eos_token_id)


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=1024)

    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.caches = []

    def parameters(self):
        return iter([self.parameter])

    def __call__(self, input_ids, use_cache, attention_mask=None, past_key_values=None, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.calls.append({
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone() if attention_mask is not None else None,
            "cache": past_key_values, "use_cache": use_cache, **kwargs,
        })
        logits = torch.zeros(*input_ids.shape, 16)
        contexts = []
        for row in range(input_ids.shape[0]):
            ids = input_ids[row].tolist()
            if past_key_values is not None:
                assert row == 0
                ids = past_key_values.token_ids + ids
            elif attention_mask is not None:
                ids = input_ids[row][attention_mask[row].bool()].tolist()
            contexts.append(ids)
            logits[row, -1, (ids[-1] + 1) % 16] = 10
        cache = SimpleNamespace(token_ids=contexts[0]) if use_cache else None
        self.caches.append(cache)
        return SimpleNamespace(logits=logits, past_key_values=cache)
