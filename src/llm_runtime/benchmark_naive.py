"""Benchmark single-request uncached greedy decoding with explicit timing boundaries."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import torch
import transformers

from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer

DEFAULT_PROMPT = "The capital of France is"


def synchronize_device(device: torch.device) -> None:
    """Wait for CUDA work before reading a wall-clock timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def calculate_timing_metrics(
    request_start: float, token_timestamps: list[float]
) -> dict[str, Any]:
    """Derive milliseconds and single-request throughput from selection timestamps."""
    if not token_timestamps:
        return {
            "total_latency_ms": 0.0,
            "ttft_ms": None,
            "inter_token_latencies_ms": [],
            "mean_itl_ms": None,
            "generated_tokens_per_second": 0.0,
            "token_availability_ms": [],
        }

    elapsed = token_timestamps[-1] - request_start
    itls = [
        (current - previous) * 1000
        for previous, current in zip(token_timestamps, token_timestamps[1:])
    ]
    return {
        "total_latency_ms": elapsed * 1000,
        "ttft_ms": (token_timestamps[0] - request_start) * 1000,
        "inter_token_latencies_ms": itls,
        "mean_itl_ms": statistics.mean(itls) if itls else None,
        "generated_tokens_per_second": len(token_timestamps) / elapsed if elapsed > 0 else 0.0,
        "token_availability_ms": [(stamp - request_start) * 1000 for stamp in token_timestamps],
    }


def timed_greedy_decode(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    """Measure the M3 full-sequence greedy mechanism, excluding final text decoding."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    device = next(model.parameters()).device
    token_timestamps = []
    forward_lengths = []
    generated_token_ids = []
    stopped_on_eos = False

    synchronize_device(device)
    request_start = time.perf_counter()
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    model_inputs = {
        name: tensor.to(device)
        for name, tensor in encoded_prompt.items()
        if isinstance(tensor, torch.Tensor)
    }
    input_ids = model_inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The benchmark supports one request with input_ids shape [1, T].")
    if input_ids.shape[1] == 0:
        raise ValueError("The prompt must contain at least one token.")
    prompt_token_ids = input_ids[0].tolist()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            forward_lengths.append(input_ids.shape[1])
            outputs = model(**model_inputs, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            selected_id = int(next_token_id.item())
            synchronize_device(device)
            # Token availability is selection, before append/bookkeeping or text decoding.
            token_timestamps.append(time.perf_counter())

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
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
            if tokenizer.eos_token_id is not None and selected_id == tokenizer.eos_token_id:
                stopped_on_eos = True
                break

    # All final text decoding and result construction occur after the last timestamp.
    return {
        "prompt_token_count": len(prompt_token_ids),
        "generated_token_count": len(generated_token_ids),
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": generated_token_ids,
        "generated_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "full_text": tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True),
        "model_forward_call_count": len(forward_lengths),
        "forward_input_lengths": forward_lengths,
        "total_input_token_positions_processed": sum(forward_lengths),
        "stopped_on_eos": stopped_on_eos,
        "stop_reason": "eos" if stopped_on_eos else "max_new_tokens",
        **calculate_timing_metrics(request_start, token_timestamps),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize measured requests; pool available ITLs, without tail percentiles."""
    latencies = [run["total_latency_ms"] for run in runs]
    ttfts = [run["ttft_ms"] for run in runs if run["ttft_ms"] is not None]
    itls = [value for run in runs for value in run["inter_token_latencies_ms"]]
    return {
        "mean_total_latency_ms": statistics.mean(latencies),
        "min_total_latency_ms": min(latencies),
        "max_total_latency_ms": max(latencies),
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
        "mean_itl_ms": statistics.mean(itls) if itls else None,
        "mean_generated_tokens_per_second": statistics.mean(
            run["generated_tokens_per_second"] for run in runs
        ),
        "generated_token_ids_identical": all(
            run["generated_token_ids"] == runs[0]["generated_token_ids"] for run in runs
        ),
    }


def validate_benchmark_parameters(max_new_tokens: int, warmup_runs: int, runs: int) -> None:
    if max_new_tokens < 0 or warmup_runs < 0 or runs < 1:
        raise ValueError("max_new_tokens and warmup_runs must be >= 0; runs must be >= 1.")


def run_benchmark(
    model: Any,
    tokenizer: Any,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 10,
    warmup_runs: int = 1,
    runs: int = 3,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Warm up an already loaded model, then retain only measured requests."""
    validate_benchmark_parameters(max_new_tokens, warmup_runs, runs)
    device = next(model.parameters()).device
    for _ in range(warmup_runs):
        timed_greedy_decode(model, tokenizer, prompt, max_new_tokens)

    synchronize_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measured_runs = [
        timed_greedy_decode(model, tokenizer, prompt, max_new_tokens)
        for _ in range(runs)
    ]
    synchronize_device(device)
    return {
        "benchmark": "naive_uncached_greedy",
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
            "cuda_max_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        },
        "config": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "warmup_runs": warmup_runs,
            "runs": runs,
            "use_cache": False,
            "timing": "perf_counter; synchronized on CUDA; start before tokenization; end at last token selection",
            "final_text_decoding_included": False,
            "throughput_scope": "single_request",
            "mean_itl_aggregation": "pooled intervals across measured requests",
            "zero_token_convention": "total latency and throughput 0; TTFT and mean ITL null",
        },
        "runs": measured_runs,
        "summary": summarize_runs(measured_runs),
    }


def main() -> None:
    """Load once and run a modest benchmark, optionally saving JSON."""
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

    # Model loading is outside every request timing boundary.
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    result = run_benchmark(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.warmup_runs, args.runs, args.model,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"JSON saved to: {args.output}")

    print(f"Model: {args.model}; device: {result['environment']['device']}")
    print(f"Warmups: {args.warmup_runs}; measured requests: {args.runs}")
    first = result["runs"][0]
    print(f"Prompt tokens: {first['prompt_token_count']}; generated tokens: {first['generated_token_count']}")
    print(f"Input token positions processed per request: {first['total_input_token_positions_processed']}")
    print("Single-request uncached greedy baseline:")
    for name, value in result["summary"].items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    if not result["summary"]["generated_token_ids_identical"]:
        print("WARNING: generated token IDs differed across measured requests; inspect the raw runs.")


if __name__ == "__main__":
    main()
