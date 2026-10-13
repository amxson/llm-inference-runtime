"""Fresh paired benchmarks of uncached and KV-cached single-request greedy decoding."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import torch
import transformers

from llm_runtime.benchmark_naive import (
    calculate_timing_metrics,
    summarize_runs,
    synchronize_device,
    timed_greedy_decode,
    validate_benchmark_parameters,
)
from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer

DEFAULT_PROMPT = "The capital of France is"


def timed_cached_greedy_decode(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    """Time M6 prefill and decode with precisely the M5 token-selection boundaries."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    device = next(model.parameters()).device
    token_timestamps = []
    forward_lengths = []
    logical_lengths = []
    generated_token_ids = []
    cache = None
    stopped_on_eos = False

    synchronize_device(device)
    request_start = time.perf_counter()
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    model_inputs = {
        name: tensor.to(device)
        for name, tensor in encoded_prompt.items()
        if isinstance(tensor, torch.Tensor)
    }
    full_input_ids = model_inputs["input_ids"]
    if full_input_ids.ndim != 2 or full_input_ids.shape[0] != 1:
        raise ValueError("The cached benchmark supports input_ids shape [1, T].")
    if full_input_ids.shape[1] == 0:
        raise ValueError("The prompt must contain at least one token.")
    prompt_token_ids = full_input_ids[0].tolist()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            forward_lengths.append(model_inputs["input_ids"].shape[1])
            logical_lengths.append(full_input_ids.shape[1])
            outputs = model(**model_inputs, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            if cache is None:
                raise RuntimeError("The model did not return a KV cache with use_cache=True.")
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            selected_id = int(next_token_id.item())
            synchronize_device(device)
            token_timestamps.append(time.perf_counter())

            newest_token = next_token_id.unsqueeze(-1)
            full_input_ids = torch.cat([full_input_ids, newest_token], dim=-1)
            model_inputs["input_ids"] = newest_token
            if "attention_mask" in model_inputs:
                mask = model_inputs["attention_mask"]
                model_inputs["attention_mask"] = torch.cat(
                    [mask, torch.ones_like(mask[:, :1])], dim=-1
                )
            if "token_type_ids" in model_inputs:
                model_inputs["token_type_ids"] = model_inputs["token_type_ids"][:, -1:]
            generated_token_ids.append(selected_id)
            if tokenizer.eos_token_id is not None and selected_id == tokenizer.eos_token_id:
                stopped_on_eos = True
                break

    return {
        "prompt_token_count": len(prompt_token_ids),
        "generated_token_count": len(generated_token_ids),
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": generated_token_ids,
        "generated_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "full_text": tokenizer.decode(full_input_ids[0].tolist(), skip_special_tokens=True),
        "model_forward_call_count": len(forward_lengths),
        "forward_input_lengths": forward_lengths,
        "logical_context_lengths": logical_lengths,
        "total_input_token_positions_processed": sum(forward_lengths),
        "stopped_on_eos": stopped_on_eos,
        "stop_reason": "eos" if stopped_on_eos else "max_new_tokens",
        **calculate_timing_metrics(request_start, token_timestamps),
    }


def compare_runs(
    uncached_runs: list[dict[str, Any]], cached_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Enforce all-run token parity before calculating any comparison ratios."""
    if not uncached_runs or len(uncached_runs) != len(cached_runs):
        raise ValueError("Comparison needs equal nonzero measured run counts.")
    reference_ids = uncached_runs[0]["generated_token_ids"]
    if any(run["generated_token_ids"] != reference_ids for run in uncached_runs + cached_runs):
        raise ValueError("Token parity failed; the benchmark comparison is invalid.")

    uncached_summary = summarize_runs(uncached_runs)
    cached_summary = summarize_runs(cached_runs)
    uncached_positions = statistics.mean(run["total_input_token_positions_processed"] for run in uncached_runs)
    cached_positions = statistics.mean(run["total_input_token_positions_processed"] for run in cached_runs)

    def ratio(numerator, denominator):
        if numerator is None or denominator is None or denominator <= 0:
            return None
        return numerator / denominator

    return {
        "exact_token_id_parity": True,
        "mean_uncached_input_positions": uncached_positions,
        "mean_cached_input_positions": cached_positions,
        "input_position_ratio_cached_over_uncached": ratio(cached_positions, uncached_positions),
        "latency_speedup_uncached_over_cached": ratio(
            uncached_summary["mean_total_latency_ms"], cached_summary["mean_total_latency_ms"]
        ),
        "throughput_speedup_cached_over_uncached": ratio(
            cached_summary["mean_generated_tokens_per_second"],
            uncached_summary["mean_generated_tokens_per_second"],
        ),
        "itl_ratio_uncached_over_cached": ratio(
            uncached_summary["mean_itl_ms"], cached_summary["mean_itl_ms"]
        ),
        "ttft_ratio_uncached_over_cached": ratio(
            uncached_summary["mean_ttft_ms"], cached_summary["mean_ttft_ms"]
        ),
    }


def run_comparison(
    model: Any,
    tokenizer: Any,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 10,
    warmup_runs: int = 1,
    runs: int = 3,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Warm up both modes and alternate measured order using one loaded model."""
    validate_benchmark_parameters(max_new_tokens, warmup_runs, runs)
    device = next(model.parameters()).device
    for _ in range(warmup_runs):
        timed_greedy_decode(model, tokenizer, prompt, max_new_tokens)
        timed_cached_greedy_decode(model, tokenizer, prompt, max_new_tokens)

    measured = {"uncached": [], "cached": []}
    execution_order = []
    for index in range(runs):
        order = ["uncached", "cached"] if index % 2 == 0 else ["cached", "uncached"]
        execution_order.append(order)
        for mode in order:
            synchronize_device(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if mode == "uncached":
                result = timed_greedy_decode(model, tokenizer, prompt, max_new_tokens)
                result["logical_context_lengths"] = result["forward_input_lengths"].copy()
            else:
                result = timed_cached_greedy_decode(model, tokenizer, prompt, max_new_tokens)
            synchronize_device(device)
            result["cuda_max_memory_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            measured[mode].append(result)

    # A mismatch raises before results can be saved or presented as valid.
    comparison = compare_runs(measured["uncached"], measured["cached"])
    return {
        "benchmark": "cached_vs_uncached_greedy",
        "environment": {
            "model_name": model_name,
            "device": str(device),
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "platform": platform.platform(),
            "torch_thread_count": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "config": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "warmup_runs_per_mode": warmup_runs,
            "measured_runs_per_mode": runs,
            "measured_execution_order": execution_order,
            "timing": "M5 perf_counter boundaries; synchronized CUDA; start before tokenization; end at last selection",
            "final_text_decoding_included": False,
            "model_loading_included": False,
            "throughput_scope": "single_request",
            "mean_itl_aggregation": "pooled intervals across measured requests",
            "zero_token_convention": "total latency and throughput 0; TTFT/ITL and undefined ratios null",
            "gpu_peak_scope": "each measured request, includes resident model allocations",
        },
        "uncached": {"runs": measured["uncached"], "summary": summarize_runs(measured["uncached"])},
        "cached": {"runs": measured["cached"], "summary": summarize_runs(measured["cached"])},
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0 (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_benchmark_parameters(args.max_new_tokens, args.warmup_runs, args.runs)
    except ValueError as error:
        parser.error(str(error))
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    result = run_comparison(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.warmup_runs, args.runs, args.model,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"JSON saved to: {args.output}")
    print(f"Model: {args.model}; device: {result['environment']['device']}")
    for mode in ["uncached", "cached"]:
        print(f"{mode}: {args.runs} measured requests, {args.warmup_runs} warmups")
        for name, value in result[mode]["summary"].items():
            print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print("Comparison (ratios use the named numerator/denominator):")
    for name, value in result["comparison"].items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")


if __name__ == "__main__":
    main()
