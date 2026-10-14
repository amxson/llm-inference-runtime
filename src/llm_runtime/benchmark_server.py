"""Measure actual concurrent HTTP/SSE requests, including network and server queueing."""

import argparse
import asyncio
import json
from pathlib import Path
import platform
import statistics
import time

import httpx


def percentile(values: list[float], percent: float) -> float | None:
    """Linear interpolation over sorted samples; these are exploratory percentiles."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100.")
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def aggregate_requests(records: list[dict], duration_seconds: float) -> dict:
    if duration_seconds <= 0:
        raise ValueError("Benchmark duration must be positive.")
    successful = [record for record in records if record["success"]]
    latencies = [record["total_latency_ms"] for record in successful]
    ttfts = [record["ttft_ms"] for record in successful if record["ttft_ms"] is not None]
    tokens = sum(record["generated_token_count"] for record in successful)
    return {
        "total_requests": len(records), "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "duration_seconds": duration_seconds, "aggregate_generated_tokens": tokens,
        "failed_partial_tokens": sum(record["generated_token_count"] for record in records if not record["success"]),
        "requests_per_second": len(successful) / duration_seconds,
        "generated_tokens_per_second": tokens / duration_seconds,
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
        "p50_ttft_ms": percentile(ttfts, 50), "p95_ttft_ms": percentile(ttfts, 95),
        "latency_sample_count": len(latencies), "ttft_sample_count": len(ttfts),
        "successful_token_ids_identical": all(
            record["generated_token_ids"] == successful[0]["generated_token_ids"] for record in successful
        ) if successful else None,
    }


async def measure_request(client: httpx.AsyncClient, body: dict, index: int) -> dict:
    start = time.perf_counter()
    first_token = None
    terminal_time = None
    token_ids = []
    completion = "error"
    error_type = None
    http_status = None
    success = False
    try:
        async with client.stream("POST", "/generate/stream", json=body) as response:
            http_status = response.status_code
            response.raise_for_status()
            name, data_lines = None, []
            done = None
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line and name:
                    data = json.loads("\n".join(data_lines))
                    now = time.perf_counter()
                    if name == "token":
                        if first_token is None:
                            first_token = now
                        token_ids.append(int(data["token_id"]))
                    elif name == "done":
                        done, terminal_time = data, now
                    elif name == "error":
                        error_type = "InferenceError"
                    name, data_lines = None, []
            if done is None:
                raise ValueError("Missing terminal done event.")
            completion = done["stop_reason"]
            success = (
                error_type is None and completion in {"eos", "max_new_tokens"}
                and token_ids == done["generated_token_ids"]
                and len(token_ids) == done["generated_token_count"]
            )
            if not success and error_type is None:
                error_type = "InvalidCompletion"
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        # Exception messages may contain URLs/credentials; persist only the type.
        error_type = type(error).__name__
        success = False
    end = terminal_time if terminal_time is not None else time.perf_counter()
    return {
        "request_index": index, "success": success, "completion_status": completion,
        "http_status": http_status, "error_type": error_type,
        "total_latency_ms": (end - start) * 1000,
        "ttft_ms": (first_token - start) * 1000 if first_token is not None else None,
        "generated_token_count": len(token_ids), "generated_token_ids": token_ids,
    }


async def run_benchmark(
    url: str, prompt: str, max_new_tokens: int, concurrencies: list[int],
    requests_per_level: int = 16, warmup_runs: int = 1, timeout: float = 60.0,
) -> dict:
    if requests_per_level < 1 or warmup_runs < 0 or max_new_tokens < 0 or timeout <= 0:
        raise ValueError("Invalid request counts, token budget, warmup count, or timeout.")
    if not concurrencies or any(value < 1 for value in concurrencies) or len(set(concurrencies)) != len(concurrencies):
        raise ValueError("Concurrency levels must be positive and unique.")
    body = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    levels = []
    limits = httpx.Limits(max_connections=max(concurrencies), max_keepalive_connections=max(concurrencies))
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits) as client:
        health_response = await client.get("/health")
        health_response.raise_for_status()
        health = health_response.json()
        for concurrency in concurrencies:
            for _ in range(warmup_runs):
                warmup = await measure_request(client, body, -1)
                if not warmup["success"]:
                    raise RuntimeError("Warmup failed; refusing to benchmark an unhealthy service.")
            before = (await client.get("/health")).raise_for_status().json()
            semaphore = asyncio.Semaphore(concurrency)

            async def execute(index):
                # Client-side semaphore waiting is outside individual latency.
                async with semaphore:
                    return await measure_request(client, body, index)

            start = time.perf_counter()
            records = await asyncio.gather(*(execute(index) for index in range(requests_per_level)))
            duration = time.perf_counter() - start
            after = (await client.get("/health")).raise_for_status().json()
            levels.append({
                "concurrency": concurrency, "requests": records,
                "summary": aggregate_requests(records, duration),
                "server_forward_calls_during_level": after["model_forward_calls"] - before["model_forward_calls"],
                "server_max_observed_batch_size_since_start": after.get("max_observed_batch_size"),
            })
    parsed_url = httpx.URL(url)
    return {
        "benchmark": "serving_concurrency_greedy",
        "environment": {
            "client_python_version": platform.python_version(),
            "client_platform": platform.platform(), "httpx_version": httpx.__version__,
            "http_server_header": health_response.headers.get("server"),
            "server": {name: health.get(name) for name in ["model", "device", "engine", "max_active"]},
        },
        "config": {
            "transport": "network_http_sse", "server_host": parsed_url.host,
            "server_port": parsed_url.port, "scheme": parsed_url.scheme,
            "prompt": prompt, "max_new_tokens": max_new_tokens,
            "concurrency_levels": concurrencies, "requests_per_level": requests_per_level,
            "serial_warmups_per_level": warmup_runs,
            "request_latency": "after client semaphore admission through terminal SSE done arrival",
            "ttft": "request start through first SSE token arrival",
            "throughput": "successful requests/tokens divided by full measured level wall time, including failures",
            "latency_statistics": "successful requests only; failed raw metrics retained",
            "percentiles": "linear interpolation; exploratory, workload-specific, small samples",
        },
        "levels": levels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_benchmark(
        args.url, args.prompt, args.max_new_tokens, args.concurrency,
        args.requests, args.warmup_runs, args.timeout,
    ))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"JSON saved to: {args.output}")
    print("Actual HTTP/SSE concurrency results (exploratory percentiles):")
    for level in result["levels"]:
        print(json.dumps({"concurrency": level["concurrency"], **level["summary"]}))


if __name__ == "__main__":
    main()
