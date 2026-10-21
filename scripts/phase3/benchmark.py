"""Paired-session loopback streaming comparison. Requires a real GPU gate pass.

Both servers run sequentially on one GPU. ABBA sessions and rotated concurrency
order reduce (not eliminate) drift without loading two models into VRAM.
"""

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

import httpx

from common import (ROOT, child_environment, command_output, gpu_observation,
                    save_json, source_provenance, utc_now)
from stream_client import cohort, distribution, first_divergence


class ExternalMemorySampler:
    """Sample ALL visible GPU process rows, not exact peaks or allocator bytes."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        started = time.perf_counter()
        while not self.stop.is_set():
            sample = command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                     "--format=csv,noheader,nounits"])
            self.samples.append({"elapsed_ms": (time.perf_counter() - started) * 1000, **sample})
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(35)


@contextmanager
def server(command, log_path, port):
    if sys.platform != "linux":
        raise RuntimeError("This workflow targets the authorized Linux/CUDA Colab host.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))  # Refuse occupied ports; never kill another server.
    with Path(log_path).open("x", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            # Only the process group created above (including vLLM workers).
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            # The API parent can exit before its GPU worker. Do not start the
            # next model until this owned process group has no live members.
            deadline = time.monotonic() + 30
            while live_group_members(process.pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned server workers did not exit; stop before loading another model")
                time.sleep(.1)


def live_group_members(group, proc_directory=Path("/proc")):
    """Linux /proc pgrp membership, ignoring exited zombie entries."""
    members = []
    for entry in proc_directory.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            # stat after comm: state, ppid, pgrp, session, ...
            if fields[0] != "Z" and int(fields[2]) == group:
                members.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue  # Process exited while inspected, or proc entry unavailable.
    return members


async def ready(base_url, process):
    deadline = time.monotonic() + 900
    async with httpx.AsyncClient(base_url=base_url, timeout=10, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Server exited; inspect the preserved server log.")
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Server startup exceeded 15 minutes; no request latencies published")


async def probe(base_url, method, endpoint, **kwargs):
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        for _ in range(100):
            response = await client.request(method, endpoint, **kwargs)
            if response.status_code != 409:
                response.raise_for_status()
                return response.json()
            await asyncio.sleep(.05)
    raise RuntimeError("Runtime never reached idle measurement boundary")


def check_runtime_outputs(row, references):
    for item in row["requests"]:
        if item["generated_token_ids"] != references[item["id"]]:
            raise ValueError(f"Runtime internal parity failed during measurement: {item['id']}")


def aggregate(rows, capacities):
    summary = []
    for capacity in capacities:
        engines = {}
        for engine in ("runtime", "vllm"):
            group = [r for r in rows if r["engine"] == engine and r["concurrency"] == capacity]
            requests = [request for row in group for request in row["requests"]]
            engines[engine] = {
                "measured_repetitions": len(group), "successful_requests": sum(r["success"] for r in requests),
                "failed_requests": sum(not r["success"] for r in requests),
                "request_latency_ms": distribution([r["latency_ms"] for r in requests]),
                "ttft_ms": distribution([r["ttft_ms"] for r in requests if r["ttft_ms"] is not None]),
                "mean_itl_ms": distribution([r["mean_itl_ms"] for r in requests if r["mean_itl_ms"] is not None]),
                "cohort_wall_time_ms": distribution([r["wall_time_ms"] for r in group]),
                "request_throughput": distribution([r["request_throughput"] for r in group]),
                "generated_token_throughput": distribution([r["generated_token_throughput"] for r in group]),
                "torch_peak_allocated_bytes": max((r["engine_stats"]["torch_memory"]["peak_allocated_bytes"] for r in group), default=None) if engine == "runtime" else None,
                "torch_peak_reserved_bytes": max((r["engine_stats"]["torch_memory"]["peak_reserved_bytes"] for r in group), default=None) if engine == "runtime" else None,
                "external_vram": "See raw sampled process rows. No exact/comparable peak VRAM claim."
            }
        comparisons = []
        runtime = [r for r in rows if r["engine"] == "runtime" and r["concurrency"] == capacity]
        production = {r["repetition"]: r for r in rows if r["engine"] == "vllm" and r["concurrency"] == capacity}
        for ours in runtime:
            theirs = {r["id"]: r for r in production[ours["repetition"]]["requests"]}
            for request in ours["requests"]:
                other = theirs[request["id"]]
                divergence = first_divergence(request["generated_token_ids"], other["generated_token_ids"])
                comparisons.append({"repetition": ours["repetition"], "id": request["id"],
                                    "exact_token_parity": divergence is None, "first_divergence_position_zero_based": divergence})
        engines["comparison"] = {
            "vllm_over_runtime_token_throughput": engines["vllm"]["generated_token_throughput"]["median"] / engines["runtime"]["generated_token_throughput"]["median"],
            "runtime_over_vllm_latency": engines["runtime"]["request_latency_ms"]["median"] / engines["vllm"]["request_latency_ms"]["median"],
            "exact_output_parity_rate": statistics.mean(c["exact_token_parity"] for c in comparisons),
            "outputs": comparisons}
        summary.append({"concurrency": capacity, **engines})
    return summary


def validate_gate(gate, provenance, capacities):
    if gate.get("experiment_only") or gate.get("numerical_experiment", "default") != "default":
        raise ValueError("Numerical experiment reports cannot authorize benchmarks; serving policy must be reconciled first")
    if gate.get("status") != "pass" or not gate.get("environment", {}).get("cuda_available"):
        raise ValueError("A passing REAL CUDA correctness artifact is required")
    if gate["source_commit"] != provenance["source_commit"] or gate["source_file_sha256"] != provenance["source_file_sha256"]:
        raise ValueError("Correctness gate source differs from this checkout")
    proven = {r["capacity"] for r in gate["continuous"] if r["exact_token_parity"]}
    if not set(capacities) <= proven:
        raise ValueError("Run correctness at every requested benchmark capacity first")
    if gate["dtype"] != "float16" or gate["model"] != "Qwen/Qwen2.5-0.5B":
        raise ValueError("Expected unquantized Qwen2.5-0.5B FP16 correctness evidence")


async def run(args, gate, provenance):
    specs = gate["workload"]
    references = {r["id"]: r["generated_token_ids"] for r in gate["references"]}
    if args.smoke:
        specs = [{**s, "max_new_tokens": 4} for s in specs[:4]]
        references = {key: value[:4] for key, value in references.items()}
    rows, sessions, warmups = [], [], []
    result = {"schema_version": 1, "benchmark": "gpu_final_comparison",
              "final_eligible": not args.smoke, "started_at_utc": utc_now(), **provenance,
              "model": gate["model"], "revision": gate["revision"], "dtype": gate["dtype"],
              "correctness_evidence": {"file": args.correctness.name, "status": gate["status"]},
              "hardware": gate["environment"], "workload": specs, "concurrencies": args.concurrencies,
              "sessions": sessions, "warmups": warmups, "raw_repetitions": rows,
              "limitations": [
                  "Only one 0.5B model, GPU and fixed mixed workload; not a production capacity claim.",
                  "10 repetitions (smoke: fewer). Request percentiles pool correlated requests; not tail SLAs or confidence intervals.",
                  "ABBA model-residency sessions with rotated concurrency order reduce, not eliminate, thermal/order bias. No strict per-repetition engine alternation.",
                  "TTFT starts immediately before HTTP send, after client semaphore admission, and ends at first nonempty token-ID event. No internet latency.",
                  "HTTP latency includes serialization, queueing, tokenization/detokenization and terminal SSE delivery, not model load/startup/warmup.",
                  "vLLM receives the same pretokenized prompt IDs; our HTTP endpoint tokenizes text. This favors vLLM slightly and is explicitly not an identical preprocessing path.",
                  "Mean ITL = first-to-last token EVENT arrival span/(tokens-1). Bundled chunks limit per-token timing resolution; raw event sizes/times are retained.",
                  "Our server synchronizes CUDA at idle reset/stats boundaries. Both engines must deliver real token IDs to CPU before SSE delivery; client-side synchronize cannot synchronize another process.",
                  "PyTorch allocated/reserved peaks describe only our allocator. External nvidia-smi process rows are sampled every ~0.5s plus query cost, not exact peaks or allocator-equivalent metrics.",
                  "Runtime forward observation and bounded telemetry are included in its serving cost. vLLM uses its ordinary native server.",
                  "vLLM 0.17.1 uses its supported CUDA wheel/Torch stack, Transformers <5 and FLASHINFER on T4; our adapter uses Transformers 5.17.0. All actual versions are recorded.",
                  "Cross-engine greedy divergence is recorded, not rejected; internal runtime/reference parity is mandatory.",
                  "EOS is honored. Different output lengths change throughput denominators; inspect raw generated-token counts.",
                  "Our logical BlockAllocator is not PagedAttention; actual compact cache collation copies K/V every decode tick."
              ]}
    try:
        # ABBA: each engine gets early/late residency. Config order rotates too.
        for session_index, engine in enumerate(("runtime", "vllm", "vllm", "runtime")):
            port = 8766 if engine == "runtime" else 8767
            base_url = f"http://127.0.0.1:{port}"
            if engine == "runtime":
                command = [str(args.runtime_python), "scripts/phase3/serve_runtime.py", "--model", gate["model"],
                           "--revision", gate["revision"], "--port", str(port)]
            else:
                command = [str(args.vllm_python), "-m", "vllm.entrypoints.openai.api_server",
                           "--model", gate["model"], "--revision", gate["revision"],
                           "--tokenizer-revision", gate["revision"], "--dtype", "half",
                           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "phase3-qwen",
                           "--max-model-len", "2048", "--max-num-seqs", str(max(args.concurrencies)),
                           "--gpu-memory-utilization", "0.70", "--generation-config", "vllm",
                           "--no-enable-prefix-caching", "--attention-config", '{"backend":"FLASHINFER"}']
            offset = session_index % len(args.concurrencies)
            order = args.concurrencies[offset:] + args.concurrencies[:offset]
            session = {"index": session_index, "engine": engine, "command": command,
                       "configuration_order": order, "started_at_utc": utc_now(), "before": gpu_observation()}
            sessions.append(session)
            log = args.output / f"server-{session_index}-{engine}.txt"
            with server(command, log, port) as process:
                await ready(base_url, process)
                session["server_pid"] = process.pid
                if engine == "runtime":
                    session["environment"] = await probe(base_url, "GET", "/phase3/environment")
                else:
                    captured = subprocess.run([str(args.vllm_python), "scripts/phase3/preflight.py"],
                                              cwd=ROOT, env=child_environment(), text=True, capture_output=True, check=True)
                    session["environment"] = json.loads(captured.stdout)
                    session["models"] = await probe(base_url, "GET", "/v1/models")
                    if session["environment"]["versions"]["vllm"] != "0.17.1":
                        raise ValueError("Expected the reviewed vLLM 0.17.1 environment")
                half = args.repetitions // 2
                repetition_range = range(half) if session_index < 2 else range(half, args.repetitions)
                for capacity in order:
                    if engine == "runtime":
                        await probe(base_url, "POST", f"/phase3/reset?max_active={capacity}")
                    for warmup in range(2):
                        row = await cohort(base_url, engine, specs, capacity, gate["eos_token_ids"])
                        row.update(engine=engine, session=session_index, warmup=warmup)
                        warmups.append(row)
                        save_json(args.output / f"warmup-{session_index}-{capacity}-{warmup}.json", row)
                        if row["failed_requests"]:
                            raise ValueError("Warmup failed; inspect raw SSE request errors")
                        if engine == "runtime":
                            check_runtime_outputs(row, references)
                    for repetition in repetition_range:
                        if engine == "runtime":
                            await probe(base_url, "POST", f"/phase3/reset?max_active={capacity}")
                        before = gpu_observation()
                        with ExternalMemorySampler() as sampler:
                            row = await cohort(base_url, engine, specs, capacity, gate["eos_token_ids"])
                        row.update(engine=engine, session=session_index, repetition=repetition,
                                   external_process_vram_samples=sampler.samples, gpu_before=before, gpu_after=gpu_observation())
                        row["engine_stats"] = (await probe(base_url, "GET", "/phase3/stats") if engine == "runtime" else
                                               {"torch_memory": None, "model_forward_count": None,
                                                "note": "No equivalent external executor counters/allocator statistics claimed"})
                        rows.append(row)
                        save_json(args.output / f"raw-{engine}-{capacity}-{repetition}.json", row)
                        if row["failed_requests"]:
                            raise ValueError("Measured request failed; no successful final artifact will be emitted")
                        if engine == "runtime":
                            check_runtime_outputs(row, references)
                        print(f"session={session_index} {engine} concurrency={capacity} repetition={repetition + 1}/{args.repetitions} complete", flush=True)
            session["finished_at_utc"] = utc_now()
            session["after"] = gpu_observation()
        result["aggregates"] = aggregate(rows, args.concurrencies)
        if source_provenance() != provenance:
            raise ValueError("Source checkout changed during measurement")
        result["finished_at_utc"] = utc_now()
        result["status"] = "pass"
        from validate import validate_artifact
        validate_artifact(result, gate, provenance)
        save_json(args.output / ("smoke_comparison.json" if args.smoke else "gpu_final_comparison.json"), result)
        return result
    except Exception as error:
        result["status"] = "fail"
        result["error"] = f"{type(error).__name__}: {error}"
        result["finished_at_utc"] = utc_now()
        save_json(args.output / "failed-comparison.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--smoke", action="store_true", help="4 requests x 4 tokens; NOT final evidence")
    args = parser.parse_args()
    if args.repetitions < (2 if args.smoke else 8) or args.repetitions % 2:
        parser.error("Use an even count >=8 for balanced ABBA (default 10); smoke permits 2")
    if not args.smoke and not {1, 2, 4} <= set(args.concurrencies):
        parser.error("Final comparison requires concurrency 1, 2 and 4")
    if len(set(args.concurrencies)) != len(args.concurrencies) or any(n not in (1, 2, 4, 8) for n in args.concurrencies):
        parser.error("Choose distinct capacities from 1, 2, 4, 8")
    if args.output.exists():
        parser.error("Choose a new output directory; do not overwrite raw evidence")
    provenance = source_provenance()
    gate = json.loads(args.correctness.read_text(encoding="utf8"))
    validate_gate(gate, provenance, args.concurrencies)
    args.output.mkdir(parents=True)
    asyncio.run(run(args, gate, provenance))


if __name__ == "__main__":
    main()
