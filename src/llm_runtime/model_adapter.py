"""Small forward-policy boundary; scheduling and cache ownership stay elsewhere.

Inspected against Transformers 5.17: Qwen2 applies RoPE using position_ids,
then appends K/V to DynamicCache. Its forward no longer takes cache_position.
Physical left-padding is masked; logical positions must not count that padding.
"""

from typing import Any


class ModelAdapter:
    """GPT-2 and test doubles retain their existing forward inputs."""

    def forward_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return dict(inputs)


class Qwen2Adapter(ModelAdapter):
    def forward_inputs(self, inputs):
        result = dict(inputs)
        result.pop("token_type_ids", None)  # Qwen has no GPT-2 segment embeddings.
        mask = result.get("attention_mask")
        if "position_ids" not in result and mask is not None:
            width = result["input_ids"].shape[-1]
            positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
            result["position_ids"] = positions[:, -width:]
        # Only the final position is selected by these explicit token loops.
        # Avoid materializing [B, prompt_length, 151936] prefill logits.
        result["logits_to_keep"] = 1
        return result


def model_adapter(model) -> ModelAdapter:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "qwen2":
        if getattr(config, "use_sliding_window", False) or "sliding_attention" in (
                getattr(config, "layer_types", None) or []):
            raise ValueError("Only full-attention Qwen2 caches are supported.")
        return Qwen2Adapter()
    return ModelAdapter()


def eos_token_ids(model, tokenizer) -> tuple[int, ...]:
    """Qwen uses generation EOS (possibly a list); preserve GPT-2's old policy."""
    generation = (getattr(model, "generation_config", None)
                  if getattr(getattr(model, "config", None), "model_type", None) == "qwen2" else None)
    value = getattr(generation, "eos_token_id", None)
    if not isinstance(value, (int, list, tuple)):
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return ()
    return (value,) if isinstance(value, int) else tuple(value)
