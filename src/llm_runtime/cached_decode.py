"""Single-request greedy decoding with explicit prefill and KV-cache reuse."""

import argparse
from typing import Any

import torch

from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer

DEFAULT_PROMPT = "The capital of France is"


def cached_greedy_decode(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    """Prefill once, then process only the newest token plus the returned cache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    device = next(model.parameters()).device
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    model_inputs = {
        name: tensor.to(device)
        for name, tensor in encoded_prompt.items()
        if isinstance(tensor, torch.Tensor)
    }
    full_input_ids = model_inputs["input_ids"]
    if full_input_ids.ndim != 2 or full_input_ids.shape[0] != 1:
        raise ValueError("cached_greedy_decode supports input_ids shape [1, T].")
    if full_input_ids.shape[1] == 0:
        raise ValueError("The prompt must contain at least one token.")

    prompt_token_ids = full_input_ids[0].tolist()
    generated_token_ids = []
    steps = []
    cache = None  # A fresh cache for each request; no cache on prefill.
    stopped_on_eos = False

    with torch.inference_mode():
        for step in range(max_new_tokens):
            input_length = model_inputs["input_ids"].shape[1]
            logical_length = full_input_ids.shape[1]
            outputs = model(**model_inputs, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            if cache is None:
                raise RuntimeError("The model did not return a KV cache with use_cache=True.")

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            selected_id = int(next_token_id.item())
            newest_token = next_token_id.unsqueeze(-1)
            full_input_ids = torch.cat([full_input_ids, newest_token], dim=-1)
            # Only this newly selected token will enter the next model call.
            model_inputs["input_ids"] = newest_token
            if "attention_mask" in model_inputs:
                mask = model_inputs["attention_mask"]
                model_inputs["attention_mask"] = torch.cat(
                    [mask, torch.ones_like(mask[:, :1])], dim=-1
                )
            if "token_type_ids" in model_inputs:
                model_inputs["token_type_ids"] = model_inputs["token_type_ids"][:, -1:]

            generated_token_ids.append(selected_id)
            steps.append({
                "step": step + 1,
                "phase": "prefill" if step == 0 else "decode",
                "model_input_length": input_length,
                "logical_context_length": logical_length,
                "token_id": selected_id,
                "token_text": tokenizer.decode([selected_id], skip_special_tokens=True),
            })
            if tokenizer.eos_token_id is not None and selected_id == tokenizer.eos_token_id:
                stopped_on_eos = True
                break

    full_token_ids = full_input_ids[0].tolist()
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "full_token_ids": full_token_ids,
        "generated_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "full_text": tokenizer.decode(full_token_ids, skip_special_tokens=True),
        "max_new_tokens": max_new_tokens,
        "generated_token_count": len(generated_token_ids),
        "stopped_on_eos": stopped_on_eos,
        "stop_reason": "eos" if stopped_on_eos else "max_new_tokens",
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0 (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be >= 0")
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    result = cached_greedy_decode(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"Prompt: {ascii(result['prompt'])}")
    print(f"Prompt token count: {len(result['prompt_token_ids'])}")
    print(f"Generated token IDs: {result['generated_token_ids']}")
    print(f"Generated text: {ascii(result['generated_text'])}")
    print(f"Generated token count: {result['generated_token_count']}")
    print(f"Stop reason: {result['stop_reason']}")
    for step in result["steps"]:
        print(
            f"  {step['step']}: {step['phase']}, input length={step['model_input_length']}, "
            f"logical context={step['logical_context_length']}, "
            f"token ID={step['token_id']}, token={ascii(step['token_text'])}"
        )
    print(f"Final text: {ascii(result['full_text'])}")


if __name__ == "__main__":
    main()
