"""One loopback HTTP measurement path for both SSE protocols.

vLLM can bundle tokens in one event. Keep actual event arrival times instead of
inventing a timestamp per token. Mean ITL is span/(tokens-1), a chunk-limited
estimate; the raw event sizes expose that limitation.
"""

import asyncio
import json
import statistics
import time

import httpx


async def sse_events(response):
    kind, lines = "message", []
    async for line in response.aiter_lines():
        if not line:
            if lines:
                yield kind, "\n".join(lines)
            kind, lines = "message", []
        elif line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        yield kind, "\n".join(lines)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median": None, "p50": None, "p90": None, "p95": None, "max": None}
    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"count": len(values), "median": statistics.median(values), "p50": quantile(.5),
            "p90": quantile(.9), "p95": quantile(.95), "max": values[-1]}


async def stream_request(client, engine, spec, *, eos_ids, model_name="phase3-qwen"):
    if engine == "runtime":
        endpoint = "/generate/stream"
        payload = {"prompt": spec["prompt"], "max_new_tokens": spec["max_new_tokens"]}
    elif engine == "vllm":
        endpoint = "/v1/completions"
        payload = {"model": model_name, "prompt": spec["prompt_token_ids"],
                   "max_tokens": spec["max_new_tokens"], "temperature": 0,
                   "top_p": 1, "seed": 0, "stream": True, "return_token_ids": True,
                   "stop_token_ids": list(eos_ids), "ignore_eos": False,
                   "skip_special_tokens": True, "stream_options": {"include_usage": True}}
    else:
        raise ValueError("Unknown engine")
    result = {"id": spec["id"], "generated_token_ids": [], "arrival_events": [],
              "engine_text": "", "success": False, "stop_reason": None, "usage": None,
              "prompt_ids_verified": False}
    started = time.perf_counter()  # Immediately before HTTP send, after concurrency admission.
    terminal = False
    terminal_arrival_ms = None
    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ValueError("Expected actual SSE response")
            async for kind, raw in sse_events(response):
                elapsed = (time.perf_counter() - started) * 1000
                if raw == "[DONE]":
                    if engine != "vllm":
                        raise ValueError("Unexpected terminator")
                    terminal = True
                    terminal_arrival_ms = elapsed
                    break
                data = json.loads(raw)
                if engine == "runtime":
                    if kind == "error":
                        raise ValueError(f"Runtime SSE error: {data}")
                    if kind == "token":
                        ids = [data["token_id"]]
                        result["engine_text"] = data["generated_text"]
                    elif kind == "done":
                        if data.get("prompt_token_ids") != spec["prompt_token_ids"]:
                            raise ValueError("Runtime input token IDs differ from the shared workload")
                        result["prompt_ids_verified"] = True
                        if data["generated_token_ids"] != result["generated_token_ids"]:
                            raise ValueError("Stream token IDs disagree with terminal result")
                        result["stop_reason"] = data["stop_reason"]
                        terminal = True
                        terminal_arrival_ms = elapsed
                        break
                    else:
                        continue
                else:
                    if "error" in data:
                        raise ValueError(f"vLLM SSE error: {data['error']}")
                    if data.get("usage"):
                        result["usage"] = data["usage"]
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    prompt_ids = choice.get("prompt_token_ids")
                    if prompt_ids is not None:
                        if prompt_ids != spec["prompt_token_ids"]:
                            raise ValueError("vLLM input token IDs differ from the shared workload")
                        result["prompt_ids_verified"] = True
                    ids = choice.get("token_ids") or []
                    result["engine_text"] += choice.get("text", "")
                    if choice.get("finish_reason"):
                        result["stop_reason"] = choice["finish_reason"]
                if ids:
                    if not all(isinstance(token, int) for token in ids):
                        raise ValueError("Engine must return exact integer token IDs")
                    result["generated_token_ids"].extend(ids)
                    result["arrival_events"].append({"arrival_ms": elapsed, "token_ids": ids})
        if not terminal or result["stop_reason"] not in {"eos", "max_new_tokens", "stop", "length"}:
            raise ValueError("Missing successful terminal SSE event")
        if not result["prompt_ids_verified"]:
            raise ValueError("vLLM did not confirm the shared prompt IDs")
        n = len(result["generated_token_ids"])
        if not 0 < n <= spec["max_new_tokens"]:
            raise ValueError("Missing IDs or invalid generated token count")
        if result["usage"] and result["usage"]["completion_tokens"] != n:
            raise ValueError("vLLM usage and returned token counts disagree")
        result["success"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    # Terminal arrival, not connection teardown or post-response validation.
    result["latency_ms"] = (terminal_arrival_ms if result["success"] else
                             (time.perf_counter() - started) * 1000)
    arrivals = result["arrival_events"]
    n = len(result["generated_token_ids"])
    result["generated_tokens"] = n
    result["ttft_ms"] = arrivals[0]["arrival_ms"] if arrivals else None
    result["mean_itl_ms"] = ((arrivals[-1]["arrival_ms"] - arrivals[0]["arrival_ms"]) / (n - 1)
                             if n > 1 else None)
    result["tokens_per_second"] = n / (result["latency_ms"] / 1000)
    return result


async def cohort(base_url, engine, specs, concurrency, eos_ids):
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=300, limits=limits, trust_env=False) as client:
        async def send(spec):
            async with semaphore:
                return await stream_request(client, engine, spec, eos_ids=eos_ids)
        started = time.perf_counter()
        results = await asyncio.gather(*(send(spec) for spec in specs))
        seconds = time.perf_counter() - started
    successful = [item for item in results if item["success"]]
    return {"requests": results, "wall_time_ms": seconds * 1000, "concurrency": concurrency,
            "successful_requests": len(successful), "failed_requests": len(results) - len(successful),
            "request_throughput": len(successful) / seconds,
            "generated_token_throughput": sum(r["generated_tokens"] for r in successful) / seconds,
            "latency_ms": distribution([r["latency_ms"] for r in successful]),
            "ttft_ms": distribution([r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]),
            "mean_itl_ms": distribution([r["mean_itl_ms"] for r in successful if r["mean_itl_ms"] is not None])}


def first_divergence(left, right):
    for position, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return position
    return min(len(left), len(right)) if len(left) != len(right) else None
