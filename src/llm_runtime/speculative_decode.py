"""Explicit greedy draft/verify decoding, using full uncached contexts."""

import argparse
import hashlib
import json
import time

import torch

from llm_runtime.benchmark_naive import calculate_timing_metrics, synchronize_device
from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer

DEFAULT_DRAFT = "sshleifer/tiny-gpt2"


def verify_compatibility(target, draft, target_tokenizer, draft_tokenizer) -> dict:
    """Check ID meanings/BPE rules, then use ONLY the target tokenizer at runtime."""
    vocab = target_tokenizer.get_vocab()
    if vocab != draft_tokenizer.get_vocab():
        raise ValueError("Draft and target token-ID vocabularies differ.")
    for model in (target, draft):
        if model.config.vocab_size != len(vocab):
            raise ValueError("Model logits vocabulary does not match the tokenizers.")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(target_tokenizer, name, None) != getattr(draft_tokenizer, name, None):
            raise ValueError("Tokenizer special-token IDs differ.")
    backends = [json.loads(t.backend_tokenizer.to_str()) for t in (target_tokenizer, draft_tokenizer)]
    # Postprocessing metadata can differ (e.g. offset handling). Shared target
    # tokenization avoids relying on it; core BPE/normalizer/decoder must agree.
    core_keys = ("model", "normalizer", "pre_tokenizer", "decoder")
    core = {key: backends[0].get(key) for key in core_keys}
    if core != {key: backends[1].get(key) for key in core_keys}:
        raise ValueError("Tokenizer core BPE/normalization/decoding rules differ.")
    probes = ["The capital of France is", "Hello", " hello\nworld", "\u00e9 \U0001f600 <|endoftext|>"]
    for prompt in probes:
        if not torch.equal(target_tokenizer(prompt, return_tensors="pt")["input_ids"],
                           draft_tokenizer(prompt, return_tensors="pt")["input_ids"]):
            raise ValueError("Tokenizer probe encodings differ.")
    return {"vocab_size": len(vocab), "exact_vocab_id_mapping": True,
            "core_bpe_rules_identical": True, "probe_encoding_parity": True,
            "core_sha256": hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest(),
            "backend_metadata_differing_fields": [key for key in backends[0] if backends[0][key] != backends[1].get(key)],
            "runtime_tokenizer": "target tokenizer shared by both models"}


def _append(inputs: dict, tokens: list[int]) -> dict:
    result = dict(inputs)
    ids = inputs["input_ids"]
    result["input_ids"] = torch.cat([ids, ids.new_tensor([tokens])], dim=-1)
    if "attention_mask" in inputs:
        mask = inputs["attention_mask"]
        result["attention_mask"] = torch.cat([mask, mask.new_ones((1, len(tokens)))], dim=-1)
    if "token_type_ids" in inputs:
        types = inputs["token_type_ids"]
        result["token_type_ids"] = torch.cat([types, types[:, -1:].expand(1, len(tokens))], dim=-1)
    return result


def speculative_greedy_decode(target, draft, tokenizer, prompt: str,
                              max_new_tokens: int, draft_tokens: int = 3) -> dict:
    """Caller must verify compatibility before supplying a shared tokenizer.

    Verification position T-1+j predicts proposal j. The final position predicts
    a bonus token if the whole proposed prefix matches. No draft KV needs rollback.
    """
    if max_new_tokens < 0 or draft_tokens < 1:
        raise ValueError("max_new_tokens must be >= 0 and draft_tokens >= 1.")
    device = next(target.parameters()).device
    if next(draft.parameters()).device != device:
        raise ValueError("Draft and target must be on the same device.")
    synchronize_device(device)
    start = time.perf_counter()
    inputs = {key: value.to(device) for key, value in tokenizer(prompt, return_tensors="pt").items()
              if isinstance(value, torch.Tensor)}
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Expected a nonempty single prompt with shape [1, T].")
    limits = [getattr(getattr(model, "config", None), "max_position_embeddings", None) for model in (target, draft)]
    context_limit = min((value for value in limits if value is not None), default=None)
    if context_limit is not None and ids.shape[1] + max(0, max_new_tokens - 1) > context_limit:
        raise ValueError("Prompt and token budget exceed the shared model context limit.")
    prompt_ids = ids[0].tolist()
    generated, timestamps, trace = [], [], []
    proposed_count = accepted_count = draft_calls = 0
    target_lengths, draft_lengths = [], []
    stopped = False
    eos = tokenizer.eos_token_id
    with torch.inference_mode():
        while len(generated) < max_new_tokens and not stopped:
            prefix_length = inputs["input_ids"].shape[1]
            remaining = max_new_tokens - len(generated)
            count = min(draft_tokens, remaining)
            if context_limit is not None:
                count = min(count, context_limit - prefix_length)
            proposed, draft_inputs = [], inputs
            for _ in range(count):
                draft_lengths.append(draft_inputs["input_ids"].shape[1])
                output = draft(**draft_inputs, use_cache=False)
                token = int(output.logits[0, -1].argmax().item())
                draft_calls += 1
                proposed.append(token)
                draft_inputs = _append(draft_inputs, [token])
                if eos is not None and token == eos:
                    break
            verify_inputs = _append(inputs, proposed) if proposed else inputs
            target_lengths.append(verify_inputs["input_ids"].shape[1])
            output = target(**verify_inputs, use_cache=False)
            predictions = output.logits[0, prefix_length - 1:prefix_length + len(proposed)].argmax(dim=-1).tolist()
            emitted, accepted, mismatch = [], 0, None
            for index, proposal in enumerate(proposed):
                if proposal != predictions[index]:
                    emitted.append(predictions[index])
                    mismatch = index
                    break  # Discard the mismatching proposal and all its suffix.
                emitted.append(proposal)
                accepted += 1
                if proposal == eos or len(emitted) == remaining:
                    break
            if mismatch is None and accepted == len(proposed) and len(emitted) < remaining and (not emitted or emitted[-1] != eos):
                emitted.append(predictions[-1])  # All accepted (or no room to draft): target bonus.
            synchronize_device(device)
            available = time.perf_counter()
            generated.extend(emitted)
            timestamps.extend([available] * len(emitted))
            inputs = _append(inputs, emitted)
            proposed_count += len(proposed)
            accepted_count += accepted
            stopped = bool(emitted and eos is not None and emitted[-1] == eos)
            trace.append({"prefix_length": prefix_length, "draft_token_ids": proposed,
                          "target_predictions": predictions, "accepted_count": accepted,
                          "mismatch_index": mismatch, "emitted_token_ids": emitted})
    return {"prompt_token_count": len(prompt_ids), "generated_token_count": len(generated),
            "max_new_tokens": max_new_tokens, "generated_token_ids": generated,
            "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
            "full_text": tokenizer.decode(prompt_ids + generated, skip_special_tokens=True),
            "proposed_draft_tokens": proposed_count, "accepted_draft_tokens": accepted_count,
            "rejected_draft_tokens": proposed_count - accepted_count,
            "acceptance_rate": accepted_count / proposed_count if proposed_count else None,
            "target_verification_calls": len(target_lengths), "draft_forward_calls": draft_calls,
            "target_forward_input_lengths": target_lengths, "draft_forward_input_lengths": draft_lengths,
            "stopped_on_eos": stopped, "stop_reason": "eos" if stopped else "max_new_tokens",
            "trace": trace, **calculate_timing_metrics(start, timestamps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--draft-tokens", type=int, default=3)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.draft_tokens < 1:
        parser.error("Token budget must be nonnegative and draft block positive.")
    target, tokenizer = load_model_and_tokenizer(args.target, args.device)
    draft, draft_tokenizer = load_model_and_tokenizer(args.draft, args.device)
    compatibility = verify_compatibility(target, draft, tokenizer, draft_tokenizer)
    print(json.dumps({"compatibility": compatibility, "result": speculative_greedy_decode(
        target, draft, tokenizer, args.prompt, args.max_new_tokens, args.draft_tokens)}, indent=2))


if __name__ == "__main__":
    main()
