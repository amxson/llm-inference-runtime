"""Manual stochastic decoding with full-sequence recomputation."""

import argparse
from typing import Any

import torch

from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from llm_runtime.sampling import sample_next_token, validate_sampling_parameters

DEFAULT_PROMPT = "The capital of France is"


def sample_decode(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Sample one request until EOS or the token limit, without a KV cache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    validate_sampling_parameters(temperature, top_k, top_p)

    model_device = next(model.parameters()).device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=model_device)
        generator.manual_seed(seed)

    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    model_inputs = {
        name: tensor.to(model_device)
        for name, tensor in encoded_prompt.items()
        if isinstance(tensor, torch.Tensor)
    }
    input_ids = model_inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("sample_decode supports one request with input_ids shape [1, T].")
    if input_ids.shape[1] == 0:
        raise ValueError("The prompt must contain at least one token.")

    prompt_token_ids = input_ids[0].tolist()
    generated_token_ids = []
    steps = []
    stopped_on_eos = False

    with torch.inference_mode():
        for step in range(max_new_tokens):
            sequence_length = input_ids.shape[1]
            # Intentionally process the full growing input on every iteration.
            outputs = model(**model_inputs, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :]
            selected_id = sample_next_token(
                next_token_logits[0], temperature, top_k, top_p, generator
            )
            next_token_id = input_ids.new_tensor([[selected_id]])
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            model_inputs["input_ids"] = input_ids
            if "attention_mask" in model_inputs:
                mask = model_inputs["attention_mask"]
                model_inputs["attention_mask"] = torch.cat(
                    [mask, torch.ones_like(mask[:, :1])], dim=-1
                )
            if "token_type_ids" in model_inputs:
                token_types = model_inputs["token_type_ids"]
                model_inputs["token_type_ids"] = torch.cat(
                    [token_types, token_types[:, -1:]], dim=-1
                )

            generated_token_ids.append(selected_id)
            steps.append({
                "step": step + 1,
                "input_sequence_length": sequence_length,
                "token_id": selected_id,
                "token_text": tokenizer.decode([selected_id], skip_special_tokens=True),
            })
            if tokenizer.eos_token_id is not None and selected_id == tokenizer.eos_token_id:
                stopped_on_eos = True
                break

    full_token_ids = input_ids[0].tolist()
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
        "sampling_config": {
            "temperature": temperature, "top_k": top_k, "top_p": top_p, "seed": seed,
        },
        "steps": steps,
    }


def main() -> None:
    """Load a model and show a short sampling run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0 (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        if args.max_new_tokens < 0:
            raise ValueError("--max-new-tokens must be >= 0")
        validate_sampling_parameters(args.temperature, args.top_k, args.top_p)
    except ValueError as error:
        parser.error(str(error))

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    result = sample_decode(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.temperature, args.top_k, args.top_p, args.seed,
    )
    print(f"Prompt: {ascii(result['prompt'])}")
    print(f"Prompt token count: {len(result['prompt_token_ids'])}")
    print(f"Sampling configuration: {result['sampling_config']}")
    print(f"Requested new tokens: {result['max_new_tokens']}")
    print(f"Generated token IDs: {result['generated_token_ids']}")
    print(f"Generated text: {ascii(result['generated_text'])}")
    print(f"Generated token count: {result['generated_token_count']}")
    print(f"Stop reason: {result['stop_reason']}")
    print(f"Stopped on EOS: {result['stopped_on_eos']}")
    print(f"Model forward calls: {len(result['steps'])}")
    print("Steps (full input sequence is recomputed each time):")
    for step in result["steps"]:
        print(
            f"  {step['step']}: input length={step['input_sequence_length']}, "
            f"token ID={step['token_id']}, token={ascii(step['token_text'])}"
        )
    print(f"Final text: {ascii(result['full_text'])}")


if __name__ == "__main__":
    main()
