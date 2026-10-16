"""Finalization measurements, reusing paired CLI boundaries and real HTTP SSE.

No engine changes. Run alone on a plugged-in, otherwise quiet CPU host.
"""
import argparse
import datetime
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time

import httpx
import torch

from llm_runtime.benchmark_cache import run_comparison
from llm_runtime.model_loader import load_model_and_tokenizer


def distribution(values):
    """Linear-interpolated empirical quantiles, not confidence intervals."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}

    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = int(index)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)

    return {"count": len(ordered), "min": ordered[0], "mean": statistics.mean(ordered),
            "median": statistics.median(ordered), "p90": percentile(.90),
            "p95": percentile(.95), "max": ordered[-1]}


def host_observation():
    """Read-only Windows state, deliberately excluding names/paths/process lists."""
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


def receive_sse(client, base, body):
    """Timestamp each parsed complete event on arrival; preserve wire lines."""
    start = time.perf_counter()
    events, wire = [], []
    with client.stream("POST", base + "/generate/stream", json=body) as response:
        response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raise ValueError("Expected real SSE response")
        kind, data = "message", []
        for line in response.iter_lines():
            wire.append(line)
            if not line:
                if data:
                    events.append({"event": kind, "arrival_ms": (time.perf_counter() - start) * 1000,
                                   "data": json.loads("\n".join(data))})
                kind, data = "message", []
            elif line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
    tokens = [event for event in events if event["event"] == "token"]
    done = [event for event in events if event["event"] == "done"]
    if len(done) != 1 or any(event["event"] == "error" for event in events):
        raise ValueError("Stream needs exactly one done and no errors")
    ids = [event["data"]["token_id"] for event in tokens]
    if [event["data"]["token_index"] for event in tokens] != list(range(1, len(tokens) + 1)):
        raise ValueError("Non-contiguous SSE token indexes")
    if ids != done[0]["data"]["generated_token_ids"] or len(ids) != body["max_new_tokens"]:
        raise ValueError("SSE completion/token mismatch or early stopping; inspect workload")
    request_id = done[0]["data"]["request_id"]
    snapshot = client.get(base + "/runtime/snapshot").raise_for_status().json()
    traces = [frame for frame in snapshot["traces"]
              if any(event["request_id"] == request_id for event in frame["events"])]
    if len(traces) != len(tokens):
        raise ValueError("Worker trace history incomplete; do not publish partial distribution")
    return {"events": events, "wire_lines": wire,
            "token_gaps_ms": [b["arrival_ms"] - a["arrival_ms"] for a, b in zip(tokens, tokens[1:])],
            "worker_steps": [{"tick": frame["tick"], "worker_step_ms": frame["worker_step_ms"],
                              "phase": frame["events"][0]["phase"]} for frame in traces],
            "token_events": len(tokens), "done_events": 1, "error_events": 0}


def run(base, warmups, runs, tokens):
    torch.set_num_threads(4)
    prompt = "The capital of France is"
    before = host_observation()
    print("Loading one model for the existing paired harness...", flush=True)
    model, tokenizer = load_model_and_tokenizer("distilgpt2", "cpu")
    print("Paired cache experiment; no tests/builds run concurrently...", flush=True)
    paired = run_comparison(model, tokenizer, prompt, tokens, warmups, runs, "distilgpt2")
    del model, tokenizer
    print("Serial network SSE warmup and measured requests...", flush=True)
    with httpx.Client(timeout=180) as client:
        health = client.get(base + "/health").raise_for_status().json()
        if health["engine"] != "continuous_uncached_greedy":
            raise ValueError("Unexpected serving path")
        body = {"prompt": prompt, "max_new_tokens": tokens}
        for _ in range(warmups):
            receive_sse(client, base, body)
        network = []
        for index in range(runs):
            print(f"SSE measured request {index + 1}/{runs}", flush=True)
            network.append(receive_sse(client, base, body))
    values = [step["worker_step_ms"] for request in network for step in request["worker_steps"]]
    decode = [step["worker_step_ms"] for request in network for step in request["worker_steps"] if step["phase"] == "decode"]
    return {"benchmark": "portfolio_finalization_cpu", "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "environment": {**paired["environment"], "host_before": before, "host_after": host_observation(),
                            "user_confirmed_ac_power_and_quiet_window": True,
                            "isolation": "No test/build/review jobs during measured execution; ordinary OS/background activity not controlled.",
                            "server": health},
            "config": {"prompt": prompt, "max_new_tokens": tokens, "warmup_runs_per_mode": warmups,
                       "measured_runs_per_mode": runs, "dtype": "float32",
                       "paired_scope": "Existing benchmark_cache.run_comparison; alternating order, single request; no HTTP.",
                       "network_scope": "Serial real HTTP uncached SSE; warmup excluded; final snapshots read after done, not polled during request.",
                       "worker_step_scope": "dispatch + forward + worker observation overhead; not pure kernel latency",
                       "quantiles": "linear interpolation, pooled observations; not independent-trial confidence bounds"},
            "paired": paired, "network_runs": network,
            "worker_step_ms": distribution(values), "decode_worker_step_ms": distribution(decode),
            "network_token_gap_ms": distribution([v for request in network for v in request["token_gaps_ms"]]),
            "limitations": ["One CPU/model/prompt; not production capacity or universal speedup.",
                            "Cached CLI selection gaps and uncached HTTP arrival gaps are different measurements; never compare them as one speedup.",
                            "Host-contention interpretation of historical data remains a hypothesis, not established causality."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--confirm-ac-quiet", action="store_true", help="Confirm AC power and pause heavy jobs before execution")
    args = parser.parse_args()
    if not args.confirm_ac_quiet:
        parser.error("Confirm AC power and a quiet run window with --confirm-ac-quiet; do not assume host isolation.")
    result = run(args.base_url, warmups=2, runs=5, tokens=128)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"worker_step_ms": result["worker_step_ms"],
                      "comparison": result["paired"]["comparison"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
