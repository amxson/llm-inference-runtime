"""Paired ordinary-target vs uncached speculative greedy CPU benchmark."""

import argparse
import json
from pathlib import Path
import platform
import time

import torch
import transformers

from llm_runtime.benchmark_naive import summarize_runs, timed_greedy_decode, validate_benchmark_parameters
from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from llm_runtime.speculative_decode import DEFAULT_DRAFT, speculative_greedy_decode, verify_compatibility


def run_comparison(target, draft, tokenizer, draft_tokenizer,
                   prompt="The capital of France is", max_new_tokens=10,
                   draft_tokens=3, warmup_runs=1, runs=3,
                   target_name=DEFAULT_MODEL_NAME, draft_name=DEFAULT_DRAFT):
    validate_benchmark_parameters(max_new_tokens, warmup_runs, runs)
    if draft_tokens < 1:
        raise ValueError("draft_tokens must be positive.")
    compatibility = verify_compatibility(target, draft, tokenizer, draft_tokenizer)
    decoders = {
        "ordinary": lambda: timed_greedy_decode(target, tokenizer, prompt, max_new_tokens),
        "speculative": lambda: speculative_greedy_decode(target, draft, tokenizer, prompt, max_new_tokens, draft_tokens),
    }
    for _ in range(warmup_runs):
        for decode in decoders.values():
            decode()
    measured = {name: [] for name in decoders}
    order = []
    for index in range(runs):
        pair = ["ordinary", "speculative"] if index % 2 == 0 else ["speculative", "ordinary"]
        order.append(pair)
        for name in pair:
            measured[name].append(decoders[name]())
    reference = measured["ordinary"][0]["generated_token_ids"]
    if any(run["generated_token_ids"] != reference for values in measured.values() for run in values):
        raise ValueError("Speculative/ordinary greedy token parity failed; comparison is invalid.")
    summaries = {name: summarize_runs(values) for name, values in measured.items()}
    ordinary_ms = summaries["ordinary"]["mean_total_latency_ms"]
    speculative_ms = summaries["speculative"]["mean_total_latency_ms"]
    return {
        "benchmark": "ordinary_vs_speculative_greedy", "compatibility": compatibility,
        "environment": {"target_model": target_name, "draft_model": draft_name,
                        "target_parameters": sum(p.numel() for p in target.parameters()),
                        "draft_parameters": sum(p.numel() for p in draft.parameters()),
                        "device": str(next(target.parameters()).device),
                        "python_version": platform.python_version(), "pytorch_version": str(torch.__version__),
                        "transformers_version": transformers.__version__, "platform": platform.platform(),
                        "torch_thread_count": torch.get_num_threads()},
        "config": {"prompt": prompt, "max_new_tokens": max_new_tokens, "draft_tokens": draft_tokens,
                   "warmup_runs_per_mode": warmup_runs, "measured_runs_per_mode": runs,
                   "measured_execution_order": order, "use_cache": False,
                   "timing": "M5 before tokenization to last selection; speculative block tokens available together after verification",
                   "model_loading_and_compatibility_included": False,
                   "rejected_definition": "all proposed tokens not emitted as accepted draft tokens, including discarded suffix",
                   "scope": "single-request greedy only; uncached target and draft; final text decode excluded"},
        **{name: {"runs": measured[name], "summary": summaries[name]} for name in decoders},
        "comparison": {"exact_token_id_parity": True,
                       "latency_delta_speculative_minus_ordinary_ms": speculative_ms - ordinary_ms,
                       "latency_speedup_ordinary_over_speculative": ordinary_ms / speculative_ms if speculative_ms > 0 else None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_benchmark_parameters(args.max_new_tokens, args.warmup_runs, args.runs)
        if args.draft_tokens < 1:
            raise ValueError("draft-tokens must be positive.")
    except ValueError as error:
        parser.error(str(error))
    start = time.perf_counter()
    target, tokenizer = load_model_and_tokenizer(args.target, "cpu")
    draft, draft_tokenizer = load_model_and_tokenizer(args.draft, "cpu")
    load_ms = (time.perf_counter() - start) * 1000
    result = run_comparison(target, draft, tokenizer, draft_tokenizer, args.prompt,
                            args.max_new_tokens, args.draft_tokens, args.warmup_runs,
                            args.runs, args.target, args.draft)
    result["setup"] = {"both_models_tokenizers_load_ms": load_ms}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"compatibility": result["compatibility"], "comparison": result["comparison"],
                      "summaries": {name: result[name]["summary"] for name in ("ordinary", "speculative")},
                      "speculative_counters": {key: result["speculative"]["runs"][0][key] for key in
                                               ("proposed_draft_tokens", "accepted_draft_tokens", "rejected_draft_tokens", "acceptance_rate", "target_verification_calls")}}, indent=2))


if __name__ == "__main__":
    main()
