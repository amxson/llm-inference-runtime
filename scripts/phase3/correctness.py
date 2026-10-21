"""Real Qwen/FP16/CUDA gate, plus explicitly ineligible numerical controls."""

import argparse
import asyncio
import inspect
from pathlib import Path
import socket
import threading
import time

import httpx
import torch
import uvicorn

from common import environment, save_json, source_provenance, torch_memory, utc_now, workload
from stream_client import cohort, sse_events
from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.greedy_decode import greedy_decode
from llm_runtime.model_adapter import eos_token_ids
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.request import RequestState
from llm_runtime.server import create_app


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numerical_settings():
    """Record process-wide policy, including the loopback server's thread."""
    cuda = torch.backends.cuda
    return {"fp16_reduced_precision_reduction": cuda.matmul.allow_fp16_reduced_precision_reduction,
            "flash_sdp": cuda.flash_sdp_enabled(), "mem_efficient_sdp": cuda.mem_efficient_sdp_enabled(),
            "math_sdp": cuda.math_sdp_enabled(), "cudnn_sdp": cuda.cudnn_sdp_enabled(),
            "math_sdp_reduced_precision": cuda.fp16_bf16_reduction_math_sdp_allowed()}


def configure_experiment(profile):
    """Explicit worker-only experiments; never change default serving policy."""
    if profile == "default":
        return None
    if profile not in {"sdpa-full-reduction", "math-full-reduction", "eager-full-reduction"}:
        raise ValueError(f"Unknown numerical experiment: {profile}")
    cuda = torch.backends.cuda
    cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if profile == "math-full-reduction":
        cuda.enable_flash_sdp(False)
        cuda.enable_mem_efficient_sdp(False)
        cuda.enable_cudnn_sdp(False)
        cuda.enable_math_sdp(True)
        cuda.allow_fp16_bf16_reduction_math_sdp(False)
    return "eager" if profile == "eager-full-reduction" else "sdpa"


def continuous(model, tokenizer, specs, capacity):
    engine = CachedContinuousBatcher(model, tokenizer, capacity)
    for spec in specs:
        engine.add_request(RequestState(spec["prompt"], spec["max_new_tokens"], spec["id"]))
    results = engine.run_until_complete()
    require(all(s.kv_cache is None and not s.runtime_inputs for s in engine.requests.values()),
            "Completed request leaked owned tensors/cache")
    return results, list(engine.trace)


def internal_checks(model, tokenizer, specs, capacities, report):
    first = specs[0]
    encoded = {k: v.to("cuda") for k, v in tokenizer(first["prompt"], return_tensors="pt").items()}
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
        require(bool(torch.isfinite(output.logits).all()), "Nonfinite Qwen logits")
        report["forward"] = {"input_shape": list(encoded["input_ids"].shape),
                             "logits_shape": list(output.logits.shape), "finite": True}
    del output, encoded
    references = {}
    report["references"] = []
    for spec in specs:
        ref = cached_greedy_decode(model, tokenizer, spec["prompt"], spec["max_new_tokens"])
        references[spec["id"]] = ref["generated_token_ids"]
        report["references"].append({"id": spec["id"], **ref})
    report["manual_greedy"] = []
    for spec in specs[:8]:
        manual = greedy_decode(model, tokenizer, spec["prompt"], min(8, spec["max_new_tokens"]))
        report["manual_greedy"].append({"id": spec["id"], **manual})
        require(manual["generated_token_ids"] == references[spec["id"]][:8],
                f"Uncached/manual parity failed: {spec['id']}")
    report["continuous"] = []
    for capacity in capacities:
        results, trace = continuous(model, tokenizer, specs, capacity)
        entry = {"capacity": capacity, "results": results, "exact_token_parity": False}
        report["continuous"].append(entry)
        for item in results:
            require(item["generated_token_ids"] == references[item["request_id"]],
                    f"Internal cached/batched parity failed: capacity={capacity}, request={item['request_id']}")
        entry["exact_token_parity"] = True
        print(f"Exact GPU internal parity passed: {len(specs)} requests, max_active={capacity}", flush=True)
    # Explicit equal-length, zero-token and EOS cases, independent of early
    # stopping in the mixed workload. Forced EOS changes policy only in this test.
    equal = [{"id": f"equal-{i}", "prompt": first["prompt"], "max_new_tokens": n}
             for i, n in enumerate((3, 5, 0))]
    results, _ = continuous(model, tokenizer, equal, 4)
    require(results[-1]["generated_token_ids"] == [] and results[-1]["model_execution_count"] == 0,
            "Zero-budget request forwarded tokens")
    for item in results:
        require(item["generated_token_ids"] == references[first["id"]][:equal[int(item["request_id"][-1])]["max_new_tokens"]],
                "Equal-length batch parity failed")
    report["equal_and_zero"] = results
    original = model.generation_config.eos_token_id
    try:
        model.generation_config.eos_token_id = [references[first["id"]][0]]
        forced, _ = continuous(model, tokenizer, [equal[0]], 1)
        require(forced[0]["stop_reason"] == "eos" and forced[0]["generated_token_count"] == 1,
                "Forced actual-selected-token EOS test failed")
        report["forced_eos_test_only"] = forced
    finally:
        model.generation_config.eos_token_id = original
    changing = [{"id": f"trace-{i}", "prompt": s["prompt"], "max_new_tokens": n}
                for i, (s, n) in enumerate(zip(specs[:4], (2, 6, 6, 5)))]
    results, trace = continuous(model, tokenizer, changing, 2)
    report["changing_cohort_trace"] = trace
    require(any(f["prefill_input_shape"] and f["prefill_input_shape"][0] == 2 for f in trace), "Missing batched prefill")
    require(any(f["decode_input_shape"] == [2, 1] for f in trace), "Missing cached [B,1] decode")
    joined = False
    for i, frame in enumerate(trace):
        if frame["admitted"] and frame["decode_request_ids"]:
            joined |= any(set(frame["admitted"]) & set(later["decode_request_ids"]) and
                          set(frame["decode_request_ids"]) & set(later["decode_request_ids"])
                          for later in trace[i + 1:])
        for event in frame["events"]:
            if event["phase"] == "decode":
                require(event["logical_cache_length_after"] == event["logical_cache_length_before"] + 1,
                        "Cache did not advance one logical position")
    require(joined, "No actual new/old request shared later decode; inspect EOS/trace before benchmarking")
    report["changing_cohort_results"] = results


async def http_checks(base_url, specs, eos_ids, references):
    async with httpx.AsyncClient(base_url=base_url, timeout=300, trust_env=False) as client:
        health = (await client.get("/health")).json()
        require(health["device"].startswith("cuda") and health["serving_cache"], "HTTP engine is not cached CUDA")
        spec = specs[0]
        response = await client.post("/generate", json={"prompt": spec["prompt"], "max_new_tokens": spec["max_new_tokens"]})
        response.raise_for_status()
        ordinary = response.json()
        require(ordinary["generated_token_ids"] == references[spec["id"]], "HTTP reference parity failed")
        before = (await client.get("/health")).json()["model_forward_calls"]
        async with client.stream("POST", "/generate/stream", json={"prompt": spec["prompt"], "max_new_tokens": 0}) as zero:
            zero.raise_for_status()
            events = [(kind, data) async for kind, data in sse_events(zero)]
        require(len(events) == 1 and events[0][0] == "done", "Zero-budget SSE emitted token events")
        require((await client.get("/health")).json()["model_forward_calls"] == before, "Zero-budget SSE executed forward")
    streaming = await cohort(base_url, "runtime", specs, 4, eos_ids)
    for item in streaming["requests"]:
        require(item["success"] and item["generated_token_ids"] == references[item["id"]],
                f"Real loopback SSE parity failed: {item}")
    return {"health": health, "ordinary_http": ordinary, "zero_sse": events, "streaming": streaming}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--numerical-experiment", default="default", choices=(
        "default", "sdpa-full-reduction", "math-full-reduction", "eager-full-reduction"),
        help="Diagnostic only: nondefault reports cannot authorize a benchmark")
    parser.add_argument("--diagnostic-fp32", action="store_true",
                        help="FP32 sensitivity control only, never an FP16 benchmark gate")
    args = parser.parse_args()
    if args.diagnostic_fp32 and args.numerical_experiment != "math-full-reduction":
        parser.error("The FP32 control requires the recorded math-full-reduction policy")
    if args.output.exists():
        parser.error("Choose a new output file; prior evidence is immutable")
    dtype = "float32" if args.diagnostic_fp32 else "float16"
    report = {"schema_version": 1, "status": "running", "started_at_utc": utc_now(),
              "model": args.model, "revision": args.revision, "dtype": dtype,
              "checkpoint_load_dtype": "float16",
              "promote_fp16_weights_for_control": args.diagnostic_fp32,
              "diagnostic_fp32": args.diagnostic_fp32,
              "numerical_experiment": args.numerical_experiment,
              "experiment_only": args.numerical_experiment != "default" or args.diagnostic_fp32}
    server, thread, listener = None, None, None
    try:
        report.update(source_provenance())
        report["environment"] = environment()
        report["numerical_settings_before"] = numerical_settings()
        attention = configure_experiment(args.numerical_experiment)
        report["numerical_settings"] = numerical_settings()
        options = {} if attention is None else {"attn_implementation": attention}
        model, tokenizer = load_model_and_tokenizer(args.model, "cuda", dtype="float16", revision=args.revision, **options)
        if args.diagnostic_fp32:
            # Keep exactly the already-rounded FP16 parameter values. The
            # control changes execution precision, not checkpoint information.
            model.float()
        require(next(model.parameters()).dtype == getattr(torch, dtype), "Model weights differ from recorded precision")
        require(model.config.model_type == "qwen2", "Expected real Qwen2 checkpoint")
        require(model.config._commit_hash == args.revision, "Resolved checkpoint revision differs")
        if attention is not None:
            require(model.config._attn_implementation == attention, "Experimental attention backend differs")
        report["tokenizer"] = type(tokenizer).__name__
        report["pad_token_id"] = tokenizer.pad_token_id
        report["eos_token_ids"] = list(eos_token_ids(model, tokenizer))
        report["model_context_limit"] = model.config.max_position_embeddings
        report["attention_implementation"] = model.config._attn_implementation
        report["installed_forward_signature"] = str(inspect.signature(model.forward))
        report["workload_config"], specs = workload(tokenizer)
        report["workload"] = specs
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        internal_checks(model, tokenizer, specs, args.concurrencies, report)
        references = {item["id"]: item["generated_token_ids"] for item in report["references"]}
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        app = create_app(model, tokenizer, model_name=args.model)
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Correctness loopback server failed to start")
            time.sleep(.05)
        report["http_sse"] = asyncio.run(http_checks(f"http://127.0.0.1:{port}", specs,
                                                     report["eos_token_ids"], references))
        report["torch_memory"] = torch_memory()
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(15)
        if listener is not None:
            listener.close()
        report["finished_at_utc"] = utc_now()
        save_json(args.output, report)
        print(f"Correctness status={report['status']}; raw evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
