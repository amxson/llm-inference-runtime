"""Inspect one causal language model forward pass."""

import argparse
from typing import Any

import torch

from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer

DEFAULT_PROMPT = "The capital of France is"


def inspect_forward_pass(
    model: Any,
    tokenizer: Any,
    prompt: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run one forward pass and summarize its tokens, tensors, and logits."""
    model_device = next(model.parameters()).device
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    model_inputs = {
        name: tensor.to(model_device)
        for name, tensor in encoded_prompt.items()
        if isinstance(tensor, torch.Tensor)
    }
    input_ids = model_inputs["input_ids"]

    with torch.inference_mode():
        outputs = model(**model_inputs)

    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    candidate_count = min(top_k, next_token_logits.shape[-1])
    top_values, top_token_ids = torch.topk(
        next_token_logits[0], k=candidate_count
    )

    top_candidates = []
    for rank, (token_id, raw_logit) in enumerate(
        zip(top_token_ids.tolist(), top_values.tolist()), start=1
    ):
        top_candidates.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "raw_logit": raw_logit,
            }
        )

    greedy_token_id = int(torch.argmax(next_token_logits[0]).item())
    greedy_candidate = {
        "token_id": greedy_token_id,
        "token_text": tokenizer.decode([greedy_token_id]),
        "raw_logit": float(next_token_logits[0, greedy_token_id].item()),
    }

    token_ids = input_ids[0].tolist()
    return {
        "prompt": prompt,
        "token_ids": token_ids,
        "token_pieces": tokenizer.convert_ids_to_tokens(token_ids),
        "input_ids": {
            "shape": tuple(input_ids.shape),
            "dtype": str(input_ids.dtype),
            "device": str(input_ids.device),
            "batch_size": input_ids.shape[0],
            "sequence_length": input_ids.shape[1],
        },
        "logits": {
            "shape": tuple(logits.shape),
            "dtype": str(logits.dtype),
            "device": str(logits.device),
        },
        "last_position_logits": {
            "shape": tuple(next_token_logits.shape),
        },
        "top_candidates": top_candidates,
        "greedy_candidate": greedy_candidate,
    }


def print_inspection(inspection: dict[str, Any]) -> None:
    """Print a compact, concept-oriented forward-pass report."""
    input_details = inspection["input_ids"]
    logits_details = inspection["logits"]

    print("Prompt")
    print("------")
    print(inspection["prompt"])

    print("\nTokenization")
    print("------------")
    print(f"token IDs: {inspection['token_ids']}")
    print(f"token pieces: {ascii(inspection['token_pieces'])}")

    print("\nInput tensor")
    print("------------")
    print(f"input_ids shape: {list(input_details['shape'])}")
    print(f"batch size: {input_details['batch_size']}")
    print(f"sequence length: {input_details['sequence_length']}")
    print(f"dtype: {input_details['dtype']}")
    print(f"device: {input_details['device']}")

    print("\nModel output")
    print("------------")
    print(f"logits shape: {list(logits_details['shape'])}")
    print(f"dtype: {logits_details['dtype']}")
    print(f"device: {logits_details['device']}")

    print("\nLast-position logits")
    print("--------------------")
    print(f"shape: {list(inspection['last_position_logits']['shape'])}")
    print("transformation: [B, T, V] -> [B, V]")

    print("\nTop candidates (raw logits, not probabilities)")
    print("---------------------------------------------")
    for candidate in inspection["top_candidates"]:
        print(
            f"{candidate['rank']}. token_id={candidate['token_id']}, "
            f"token={ascii(candidate['token_text'])}, "
            f"raw_logit={candidate['raw_logit']:.6f}"
        )

    greedy = inspection["greedy_candidate"]
    print("\nGreedy candidate")
    print("----------------")
    print(f"token ID: {greedy['token_id']}")
    print(f"token: {ascii(greedy['token_text'])}")
    print(f"raw logit: {greedy['raw_logit']:.6f}")
    print("The candidate was not appended; no additional forward pass was run.")


def main() -> None:
    """Load a model and print one forward-pass inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt to inspect (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model identifier (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        help="Device to use, such as 'cpu', 'cuda', or 'cuda:0' (default: auto)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    inspection = inspect_forward_pass(model, tokenizer, args.prompt)
    print_inspection(inspection)


if __name__ == "__main__":
    main()
