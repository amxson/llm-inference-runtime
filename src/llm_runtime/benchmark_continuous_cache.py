"""Paired cached/uncached continuous CPU cohorts; preserve raw runs and losses."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
import transformers

from llm_runtime.benchmark_cache_sweep import distribution, execution_order, host_observation, paired_ratios
from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.continuous_batcher import ContinuousBatcher
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.request import RequestState

WORKLOAD = [
    ("The capital of France is", 16),
    ("An operating system manages memory and", 24),
    ("When sunlight passes through a glass prism, the colors", 32),
    ("A computer network connects", 16),
    ("In the early morning, the city streets were quiet and", 24),
    ("A database transaction should", 32),
    ("The purpose of a scientific experiment is to test a hypothesis by", 16),
    ("In a library, books are organized", 24),
]
ENGINES = {"uncached": ContinuousBatcher, "cached": CachedContinuousBatcher}
METRICS = ("wall_ms", "requests_per_second", "generated_tokens_per_second", "model_forward_calls",
           "total_input_positions", "cached_decode_input_positions", "uncached_input_positions", "peak_active")


def trace_accounting(traces, cached):
    """Count physical padded model inputs, not attention FLOPs or logical KV size."""
    calls, prefill_sizes, decode_sizes, summary = [], [], [], []
    for frame in traces:
        events = frame["events"]
        prefill = [e for e in events if e["phase"] == "prefill"]
        decode = [e for e in events if e["phase"] == "decode"]
        if prefill:
            prefill_sizes.append(len(prefill))
        if decode:
            decode_sizes.append(len(decode))
        groups = [("prefill", prefill), ("decode", decode)] if cached else [("uncached", events)]
        for phase, group in groups:
            if not group:
                continue
            width = group[0]["model_input_width"]
            if any(e["model_input_width"] != width or e["batch_size"] != len(group) for e in group):
                raise ValueError("Trace does not describe one rectangular model forward.")
            if phase == "decode" and width != 1:
                raise ValueError("Cached decode must forward exactly one newest token per row.")
            calls.append({"tick": frame["tick"], "phase": phase, "batch_size": len(group),
                          "input_width": width, "input_positions": len(group) * width})
        summary.append({key: frame[key] for key in (
            "tick", "admitted", "active_request_ids", "completed", "model_request_ids")})
        if cached:
            summary[-1].update({key: frame[key] for key in (
                "prefill_request_ids", "decode_request_ids", "prefill_input_shape", "decode_input_shape",
                "logical_cache_lengths_before", "logical_cache_lengths_after")})
    return {"calls": calls, "prefill_batch_sizes": prefill_sizes, "decode_batch_sizes": decode_sizes,
            "total_input_positions": sum(c["input_positions"] for c in calls),
            "cached_decode_input_positions": sum(c["input_positions"] for c in calls if c["phase"] == "decode"),
            "uncached_input_positions": sum(c["input_positions"] for c in calls if c["phase"] == "uncached"),
            "trace_summary": summary}


def run_cohort(model, tokenizer, mode, max_active, workload=WORKLOAD):
    if mode not in ENGINES:
        raise ValueError("Unknown benchmark mode.")
    engine = ENGINES[mode](model, tokenizer, max_active)
    start = time.perf_counter()
    states = [RequestState(prompt, budget, f"r{i + 1}") for i, (prompt, budget) in enumerate(workload)]
    for state in states:
        engine.add_request(state)
    traces, completion_ms = [], {}
    while engine.has_work:
        frame = engine.step()
        elapsed = (time.perf_counter() - start) * 1000
        completion_ms.update({request_id: elapsed for request_id in frame["completed"]})
        traces.append(frame)
    wall_ms = (time.perf_counter() - start) * 1000
    results = [s.to_result(tokenizer) for s in states]  # Final text export outside timing.
    if any(s.stop_reason not in ("eos", "max_new_tokens") for s in states):
        raise ValueError("A cohort request failed.")
    accounting = trace_accounting(traces, mode == "cached")
    if len(accounting["calls"]) != engine.model_forward_count:
        raise ValueError("Model forward accounting mismatch.")
    tokens = sum(s.generated_token_count for s in states)
    return {"mode": mode, "max_active": max_active, "request_count": len(states),
            "successful_count": len(states), "generated_token_count": tokens,
            "wall_ms": wall_ms, "requests_per_second": len(states) * 1000 / wall_ms,
            "generated_tokens_per_second": tokens * 1000 / wall_ms,
            "request_latencies_ms": completion_ms, "model_forward_calls": engine.model_forward_count,
            "peak_active": max((len(t["active_request_ids"]) for t in traces), default=0),
            "results": results, **accounting}


def assert_parity(uncached, cached):
    if uncached["request_count"] != cached["request_count"]:
        raise ValueError("Different request counts.")
    for u, c in zip(uncached["results"], cached["results"]):
        if any(u[key] != c[key] for key in (
                "request_id", "prompt_token_ids", "generated_token_ids", "stop_reason")):
            raise ValueError("Exact token/lifecycle parity failed; do not publish a ratio.")


def batch_summary(values):
    values = list(values)
    return {"distribution": distribution(values), "counts": dict(sorted(Counter(values).items()))}


def aggregate_pairs(max_active, pairs):
    if not pairs:
        raise ValueError("Need measured pairs.")
    for index, pair in enumerate(pairs):
        if pair["pair_index"] != index or pair["execution_order"] != execution_order(index):
            raise ValueError("Invalid pair order.")
        assert_parity(pair["uncached"], pair["cached"])
        for mode in ENGINES:
            run = pair[mode]
            if (run["max_active"] != max_active or run["mode"] != mode
                    or run["successful_count"] != run["request_count"]
                    or len(run["results"]) != run["request_count"]
                    or run["model_forward_calls"] != len(run["calls"])
                    or run["total_input_positions"] != sum(c["input_positions"] for c in run["calls"])):
                raise ValueError("Invalid cohort accounting/schema.")
            if run["wall_ms"] <= 0:
                raise ValueError("Invalid wall time.")
    modes = {}
    for mode in ENGINES:
        runs = [pair[mode] for pair in pairs]
        modes[mode] = {key: distribution(r[key] for r in runs) for key in METRICS}
        modes[mode]["request_latency_ms"] = distribution(v for r in runs for v in r["request_latencies_ms"].values())
        for phase in ("prefill", "decode"):
            modes[mode][phase + "_batch_sizes"] = batch_summary(v for r in runs for v in r[phase + "_batch_sizes"])
    return {"max_active": max_active, "measured_pair_count": len(pairs), "exact_token_id_parity": True,
            "pairs": pairs, "aggregates": modes,
            "paired_wall_time_ratios": paired_ratios(
                [p["uncached"]["wall_ms"] for p in pairs], [p["cached"]["wall_ms"] for p in pairs])}


def run_benchmark(model, tokenizer, levels=(1, 2, 4), warmups=2, pairs=10, progress=print):
    if warmups < 2 or pairs < 7:
        raise ValueError("Need two warmup cohorts/mode and at least seven measured pairs.")
    output = []
    for capacity in levels:
        for index in range(warmups):
            u = run_cohort(model, tokenizer, "uncached", capacity)
            c = run_cohort(model, tokenizer, "cached", capacity)
            assert_parity(u, c)
            progress(f"max_active={capacity}: warmup {index + 1}/{warmups} complete", flush=True)
        measured = []
        for index in range(pairs):
            pair = {"pair_index": index, "execution_order": execution_order(index)}
            for mode in pair["execution_order"]:
                pair[mode] = run_cohort(model, tokenizer, mode, capacity)
            assert_parity(pair["uncached"], pair["cached"])
            measured.append(pair)
            progress(f"max_active={capacity}: pair {index + 1}/{pairs}; "
                     f"uncached={pair['uncached']['wall_ms']:.1f}ms cached={pair['cached']['wall_ms']:.1f}ms; exact IDs", flush=True)
        output.append(aggregate_pairs(capacity, measured))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-ac-quiet", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_ac_quiet:
        parser.error("Confirm current AC power and quiet measurement window first.")
    if args.output.exists():
        parser.error("Output exists; prior measurements must not be overwritten.")
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = ("benchmark_continuous_cache.py", "cached_continuous_batcher.py", "cache_batch.py",
             "continuous_batcher.py", "batching.py", "model_loader.py")
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in files}
    torch.set_num_threads(4)
    before = host_observation()
    started = datetime.now(timezone.utc).isoformat()
    model, tokenizer = load_model_and_tokenizer("distilgpt2", "cpu")
    levels = run_benchmark(model, tokenizer)
    result = {"schema_version": 1, "benchmark": "cached_continuous_batching_cpu",
              "started_at_utc": started, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_revision": revision, "source_file_sha256": hashes,
              "source_note": "Committed runtime baseline; new harness identified by hashes and committed with artifact.",
              "environment": {"model_name": "distilgpt2", "device": "cpu", "python_version": platform.python_version(),
                              "pytorch_version": str(torch.__version__), "transformers_version": transformers.__version__,
                              "platform": platform.platform(), "torch_thread_count": torch.get_num_threads(),
                              "torch_interop_thread_count": torch.get_num_interop_threads(),
                              "dtype": str(next(model.parameters()).dtype), "attention_implementation": model.config._attn_implementation},
              "host": {"before": before, "after": host_observation(), "user_confirmed_ac_quiet": True,
                       "isolation": "No tests/builds/browser review during measurement; OS/background work not controlled."},
              "config": {"workload": [{"prompt": p, "max_new_tokens": n} for p, n in WORKLOAD],
                         "max_active": [1, 2, 4], "warmup_cohorts_per_mode_per_level": 2, "measured_pairs_per_level": 10,
                         "timing": "perf_counter from fresh RequestState construction/admission through all completions; includes tokenization, collation, engine trace overhead; excludes model loading and final text export; no HTTP.",
                         "request_latency": "All requests arrive at cohort start; completion boundary latency includes waiting. Batch-size distributions count phase rows per tick; uncached prefill/decode rows may share one mixed forward.",
                         "primary_statistic": "Median paired uncached/cached cohort wall-time ratio; linear empirical quantiles; sample stdev n-1."},
              "exact_token_id_parity": True, "levels": levels,
              "limitations": ["One small CPU/model/workload; not a GPU or production capacity result.",
                              "Ten pairs per capacity; p90/p95 are exploratory, not tail SLAs/confidence intervals.",
                              "Fixed ascending capacity order can confound thermal drift/background activity.",
                              "Input positions include temporary padding; work proxy, NOT FLOPs or attention reads.",
                              "Cache collation/split copy K/V every tick; no physical paging or optimized attention kernel.",
                              "Exact parity within each pair is required; EOS honored and observed counts retained."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; all measured pairs passed exact token parity.")


if __name__ == "__main__":
    main()
