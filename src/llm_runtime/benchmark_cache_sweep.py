"""Controlled CPU cache-length sweep using the existing timed greedy decoders."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

import torch
import transformers

from llm_runtime.benchmark_cache import DEFAULT_PROMPT, timed_cached_greedy_decode
from llm_runtime.benchmark_naive import timed_greedy_decode
from llm_runtime.model_loader import load_model_and_tokenizer

METRICS = ("total_latency_ms", "ttft_ms", "mean_itl_ms", "generated_tokens_per_second")


def distribution(values):
    """Empirical linear-interpolated quantiles; sample stdev needs two values."""
    values = list(values)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Distribution values must be finite numbers.")
    ordered = sorted(values)
    fields = ("mean", "median", "min", "max", "p25", "p75", "p90", "p95", "sample_stdev")
    if not ordered:
        return {"count": 0, **dict.fromkeys(fields)}

    def quantile(q):
        index = (len(ordered) - 1) * q
        lower = int(index)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)

    return {"count": len(ordered), "mean": statistics.mean(ordered),
            "median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1],
            **{name: quantile(q) for name, q in (("p25", .25), ("p75", .75), ("p90", .9), ("p95", .95))},
            "sample_stdev": statistics.stdev(ordered) if len(ordered) > 1 else None}


def execution_order(pair_index):
    """Zero-based index: reverse the order on each successive pair."""
    if type(pair_index) is not int or pair_index < 0:
        raise ValueError("Pair index must be a nonnegative integer.")
    return ["uncached", "cached"] if pair_index % 2 == 0 else ["cached", "uncached"]


def position_counts(prompt_tokens, generated_tokens):
    """Count forwarded positions, not FLOPs; final selected token is not forwarded."""
    if type(prompt_tokens) is not int or prompt_tokens < 1 or type(generated_tokens) is not int or generated_tokens < 0:
        raise ValueError("Need a positive prompt length and nonnegative output length.")
    uncached = generated_tokens * prompt_tokens + generated_tokens * (generated_tokens - 1) // 2
    cached = prompt_tokens + generated_tokens - 1 if generated_tokens else 0
    return {"uncached_positions": uncached, "cached_positions": cached,
            "position_reduction_fraction": 1 - cached / uncached if uncached else None,
            "position_ratio_uncached_over_cached": uncached / cached if cached else None}


def paired_ratios(uncached, cached):
    if not uncached or len(uncached) != len(cached):
        raise ValueError("Need equal nonzero pair counts.")
    distribution([*uncached, *cached])  # Reject nonfinite and nonnumeric inputs.
    if any(value <= 0 for value in [*uncached, *cached]):
        raise ValueError("Paired latencies must be positive.")
    values = [u / c for u, c in zip(uncached, cached)]
    return {"values": values, "summary": distribution(values)}


def aggregate_level(budget, pairs):
    """Validate raw pairs before computing summaries or publishing latency ratios."""
    if type(budget) is not int or budget < 1 or not pairs:
        raise ValueError("Need a positive budget and measured pairs.")
    counts = []
    for index, pair in enumerate(pairs):
        if pair["pair_index"] != index or pair["execution_order"] != execution_order(index):
            raise ValueError("Invalid pair identity/order.")
        uncached, cached = pair["uncached"], pair["cached"]
        if uncached["generated_token_ids"] != cached["generated_token_ids"]:
            raise ValueError("Token parity failed; no speedup may be published.")
        n = len(uncached["generated_token_ids"])
        prompt = uncached["prompt_token_count"]
        if not 1 <= n <= budget or cached["prompt_token_count"] != prompt:
            raise ValueError("Invalid observed token counts.")
        expected = position_counts(prompt, n)
        for mode in ("uncached", "cached"):
            run = pair[mode]
            lengths = list(range(prompt, prompt + n)) if mode == "uncached" else [prompt] + [1] * (n - 1)
            if (run["generated_token_count"] != n or run["forward_input_lengths"] != lengths
                    or run["total_input_token_positions_processed"] != expected[mode + "_positions"]):
                raise ValueError("Raw run disagrees with forwarded-position counts.")
            if run["max_new_tokens"] != budget or (n < budget and run["stop_reason"] != "eos"):
                raise ValueError("Short output must record EOS, not a fabricated full-length run.")
            for metric in METRICS:
                value = run[metric]
                if value is None:
                    if metric != "mean_itl_ms" or n != 1:
                        raise ValueError("Missing timing metric.")
                elif distribution([value])["min"] < 0:
                    raise ValueError("Negative timing metric.")
        counts.append(expected)
    aggregates = {
        mode: {metric: distribution(pair[mode][metric] for pair in pairs if pair[mode][metric] is not None)
               for metric in METRICS}
        for mode in ("uncached", "cached")
    }
    ratios = paired_ratios([p["uncached"]["total_latency_ms"] for p in pairs],
                           [p["cached"]["total_latency_ms"] for p in pairs])
    return {"requested_tokens": budget, "measured_pair_count": len(pairs),
            "exact_token_id_parity": True,
            "observed_generated_tokens": sorted({p["uncached"]["generated_token_count"] for p in pairs}),
            "all_runs_reached_budget": all(p["uncached"]["generated_token_count"] == budget for p in pairs),
            "pairs": pairs, "aggregates": aggregates, "paired_latency_ratios": ratios,
            "work_positions_by_pair": counts,
            "work_positions": counts[0] if all(c == counts[0] for c in counts) else None}


def run_sweep(model, tokenizer, budgets=(32, 128, 512), warmups=2, pairs=10, progress=print):
    """One loaded model, fresh request caches, adjacent paired measurements."""
    if type(warmups) is not int or warmups < 2 or type(pairs) is not int or pairs < 10:
        raise ValueError("At least two warmups/mode and ten measured pairs are required.")
    if not budgets or len(set(budgets)) != len(budgets) or any(type(n) is not int or n < 1 for n in budgets):
        raise ValueError("Budgets must be distinct positive integers.")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("This sweep is CPU-only.")
    prompt_tokens = tokenizer(DEFAULT_PROMPT, return_tensors="pt")["input_ids"].shape[1]
    limit = model.config.max_position_embeddings
    if prompt_tokens < 1 or prompt_tokens + max(budgets) > limit:
        raise ValueError("Prompt plus output budget exceeds the model context window.")
    decoders = {"uncached": timed_greedy_decode, "cached": timed_cached_greedy_decode}
    levels = []
    for budget in budgets:
        progress(f"{budget} tokens: {warmups} excluded warmups per mode", flush=True)
        for _ in range(warmups):
            for mode in ("uncached", "cached"):
                decoders[mode](model, tokenizer, DEFAULT_PROMPT, budget)
        measured = []
        for index in range(pairs):
            order = execution_order(index)
            progress(f"{budget} tokens: pair {index + 1}/{pairs}, {' -> '.join(order)}", flush=True)
            pair = {"pair_index": index, "execution_order": order}
            for mode in order:
                pair[mode] = decoders[mode](model, tokenizer, DEFAULT_PROMPT, budget)
            # Fail at the first mismatch, before another pair or any file is written.
            if pair["uncached"]["generated_token_ids"] != pair["cached"]["generated_token_ids"]:
                raise ValueError("Token parity failed; no speedup may be published.")
            measured.append(pair)
        levels.append(aggregate_level(budget, measured))
    return levels


def host_observation():
    """Read-only observations, matching the prior harness; no names/process lists."""
    if platform.system() != "Windows":
        return {"power_plan": "not observed", "cpu_load_percent": None}
    command = (
        "$cpu=Get-CimInstance Win32_Processor; $b=Get-CimInstance Win32_Battery; "
        "[pscustomobject]@{cpu=($cpu|Select-Object -First 1).Name;"
        "cpu_load_percent=($cpu|Measure-Object LoadPercentage -Average).Average;"
        "power_plan=(powercfg /getactivescheme);battery_status=$b.BatteryStatus;"
        "battery_percent=$b.EstimatedChargeRemaining}|ConvertTo-Json -Compress"
    )
    return json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-ac-quiet", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_ac_quiet:
        parser.error("Confirm AC power and pause heavy work with --confirm-ac-quiet.")
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve prior measurements.")
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in ("benchmark_cache_sweep.py", "benchmark_cache.py", "benchmark_naive.py", "model_loader.py")}
    torch.set_num_threads(4)
    before = host_observation()
    started = datetime.now(timezone.utc).isoformat()
    print("Loading distilgpt2 once; model loading is outside request timing.", flush=True)
    model, tokenizer = load_model_and_tokenizer("distilgpt2", "cpu")
    levels = run_sweep(model, tokenizer)
    after = host_observation()
    result = {"schema_version": 1, "benchmark": "cache_length_sweep_cpu",
              "started_at_utc": started, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_revision": revision, "source_file_sha256": hashes,
              "source_note": "Revision is the committed runtime baseline; harness in this worktree is identified by file hashes and committed with this artifact.",
              "environment": {"model_name": "distilgpt2", "device": "cpu", "dtype": str(next(model.parameters()).dtype),
                              "python_version": platform.python_version(), "pytorch_version": str(torch.__version__),
                              "transformers_version": transformers.__version__, "platform": platform.platform(),
                              "torch_thread_count": torch.get_num_threads(), "torch_interop_thread_count": torch.get_num_interop_threads(),
                              "attention_implementation": model.config._attn_implementation,
                              "model_context_limit": model.config.max_position_embeddings},
              "host": {"before": before, "after": after, "user_confirmed_ac_quiet": True,
                       "isolation": "No tests, builds or browser review during measurement; ordinary OS/background activity not controlled."},
              "config": {"prompt": DEFAULT_PROMPT, "generated_token_budgets": [32, 128, 512],
                         "length_execution_order": [32, 128, 512], "warmup_runs_per_mode_per_length": 2,
                         "measured_pairs_per_length": 10, "warmup_order_each_round": ["uncached", "cached"],
                         "timing": "Existing timed_greedy_decode/timed_cached_greedy_decode; perf_counter before tokenization through last selection; excludes loading and final text decode.",
                         "aggregation": "Per-request metrics; empirical linear-interpolated quantiles, sample stdev (n-1); not confidence intervals.",
                         "primary_statistic": "Median of ten matched uncached/cached latency ratios, not ratio of aggregate means."},
              "exact_token_id_parity": all(level["exact_token_id_parity"] for level in levels), "levels": levels,
              "limitations": ["One CPU/model/prompt; fixed ascending length order may confound length with thermal/background drift.",
                              "Ten pairs per length are exploratory; percentiles are not tail SLAs or confidence intervals.",
                              "Forwarded positions are not FLOPs or latency predictions; attention still reads prior K/V.",
                              "Local single-request selection timing excludes HTTP; this is not cached continuous batching or a GPU result.",
                              "EOS is honored; observed token counts and early stopping are retained without padding results."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; exact parity verified for every measured pair.", flush=True)


if __name__ == "__main__":
    main()
