"""Recompute GPU artifact aggregates and enforce evidence/provenance gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from common import ROOT, source_provenance


def committed_provenance(commit, root=ROOT):
    """Review published evidence against immutable Git blobs, not today's files.

    Collection still requires a clean checkout. This read-only review path lets
    later UI/docs commits coexist with evidence from its original measured code.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full measured source commit SHA")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    resolved = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise ValueError("Measured source commit did not resolve exactly")
    names = git("ls-tree", "-r", "--name-only", commit, "--", "src", "scripts/phase3", "pyproject.toml").decode().splitlines()
    if not names or "pyproject.toml" not in names:
        raise ValueError("Measured source tree is incomplete")
    return {"source_commit": commit, "source_file_sha256": {
        name: hashlib.sha256(git("show", f"{commit}:{name}").replace(b"\r\n", b"\n")).hexdigest()
        for name in names}}


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def validate_artifact(data, gate, provenance):
    from benchmark import aggregate, validate_gate
    validate_gate(gate, provenance, data["concurrencies"])
    if data.get("status") != "pass" or data.get("schema_version") != 1:
        raise ValueError("Not a successful comparison artifact")
    for name in ("source_commit", "source_file_sha256"):
        if data[name] != provenance[name]:
            raise ValueError("Artifact source provenance mismatch")
    if (data["revision"], data["model"], data["dtype"]) != (gate["revision"], gate["model"], gate["dtype"]):
        raise ValueError("Model/revision/dtype mismatch")
    if data.get("execution_policy", "fp16-sdpa") != gate.get("execution_policy", "fp16-sdpa"):
        raise ValueError("Execution policy differs from the correctness gate")
    final = data["final_eligible"]
    if final and (len(data["workload"]) < 16 or not {1, 2, 4} <= set(data["concurrencies"]) or
                  not {32, 64, 128} <= {s["max_new_tokens"] for s in data["workload"]}):
        raise ValueError("Final workload/configuration is insufficient")
    specs = {s["id"]: s for s in data["workload"]}
    refs = {r["id"]: r["generated_token_ids"] for r in gate["references"]}
    seen = set()
    counts = {}
    for row in data["raw_repetitions"]:
        key = (row["engine"], row["concurrency"], row["repetition"])
        if key in seen:
            raise ValueError("Duplicate engine/configuration/repetition")
        seen.add(key)
        counts[key[:2]] = counts.get(key[:2], 0) + 1
        if {r["id"] for r in row["requests"]} != set(specs) or len(row["requests"]) != len(specs):
            raise ValueError("Missing/duplicate workload request")
        total = 0
        for request in row["requests"]:
            ids = request["generated_token_ids"]
            arrivals = request["arrival_events"]
            if not request["success"] or not ids or request["generated_tokens"] != len(ids):
                raise ValueError("Failed/malformed raw request")
            if len(ids) > specs[request["id"]]["max_new_tokens"]:
                raise ValueError("Token budget exceeded")
            if [token for event in arrivals for token in event["token_ids"]] != ids:
                raise ValueError("Event IDs disagree with result")
            times = [event["arrival_ms"] for event in arrivals]
            if not times or times != sorted(times) or times[0] < 0 or times[-1] > request["latency_ms"]:
                raise ValueError("Invalid event timing")
            if not close(request["ttft_ms"], times[0]):
                raise ValueError("TTFT definition violated")
            if len(ids) > 1 and not close(request["mean_itl_ms"], (times[-1] - times[0]) / (len(ids) - 1)):
                raise ValueError("ITL cannot be recomputed")
            if not close(request["tokens_per_second"], len(ids) * 1000 / request["latency_ms"]):
                raise ValueError("Per-request throughput cannot be recomputed")
            if row["engine"] == "runtime" and ids != refs[request["id"]][:specs[request["id"]]["max_new_tokens"]]:
                raise ValueError("Runtime reference parity failed")
            total += len(ids)
        if row["failed_requests"] != 0 or row["successful_requests"] != len(specs):
            raise ValueError("Failed requests in a successful benchmark")
        seconds = row["wall_time_ms"] / 1000
        if not close(row["generated_token_throughput"], total / seconds) or not close(row["request_throughput"], len(specs) / seconds):
            raise ValueError("Cohort throughput cannot be recomputed")
        if row["engine"] == "runtime":
            stats = row["engine_stats"]
            if stats["model_forward_count"] != len(stats["forwards"]):
                raise ValueError("Forward counts disagree")
            if any(f["phase"] == "decode" and f["input_shape"][1] != 1 for f in stats["forwards"]):
                raise ValueError("Cached decode forwarded more than one token per row")
            if stats["forwarded_token_positions"] != sum(f["input_shape"][0] * f["input_shape"][1] for f in stats["forwards"]):
                raise ValueError("Forwarded positions disagree")
            if stats["torch_memory"]["peak_allocated_bytes"] <= 0:
                raise ValueError("Missing GPU allocator evidence")
    for capacity in data["concurrencies"]:
        for engine in ("runtime", "vllm"):
            if counts.get((engine, capacity), 0) < (7 if final else 2):
                raise ValueError("Insufficient measured repetitions")
        if counts[("runtime", capacity)] != counts[("vllm", capacity)]:
            raise ValueError("Unbalanced measured engine counts")
    if [s["engine"] for s in data["sessions"]] != ["runtime", "vllm", "vllm", "runtime"]:
        raise ValueError("Missing balanced ABBA session order")
    for session in data["sessions"]:
        if data.get("execution_policy") == "fp32-math":
            from policies import validate_fp32_record
            if session["engine"] == "runtime":
                validate_fp32_record(session["environment"])
                if session["environment"].get("execution_policy") != "fp32-math":
                    raise ValueError("Runtime did not report the approved FP32 policy")
            else:
                command = session["command"]
                if (command[command.index("--dtype") + 1] != "float32" or "--enforce-eager" not in command or
                        json.loads(command[command.index("--attention-config") + 1])["backend"] != "TRITON_ATTN"):
                    raise ValueError("vLLM FP32 configuration differs from the recorded policy")
        env = session["environment"]
        env = env["environment"] if session["engine"] == "runtime" else env
        if not env["cuda_available"] or env["gpus"] != data["hardware"]["gpus"]:
            raise ValueError("Engine GPU differs from correctness host")
        if env["versions"]["torch"] != data["hardware"]["versions"]["torch"] or env["cuda_runtime"] != data["hardware"]["cuda_runtime"]:
            raise ValueError("Engine Torch/CUDA package set differs")
        for capacity in data["concurrencies"]:
            warmups = [r for r in data["warmups"] if r["session"] == session["index"] and r["concurrency"] == capacity]
            if len(warmups) < 2 or any(r["failed_requests"] for r in warmups):
                raise ValueError("Missing successful configuration warmups")
    if aggregate(data["raw_repetitions"], data["concurrencies"]) != data["aggregates"]:
        raise ValueError("Aggregate metrics/parity cannot be recomputed")
    return {"status": "pass", "final_eligible": final, "source_commit": data["source_commit"],
            "measured_cohorts": len(data["raw_repetitions"]), "note": "Consistency/provenance validation, not independent attestation of remote execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--source-commit", help="Review against the full measured Git SHA; never used for collection")
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text(encoding="utf8"))
    gate = json.loads(args.correctness.read_text(encoding="utf8"))
    provenance = committed_provenance(args.source_commit) if args.source_commit else source_provenance()
    print(json.dumps(validate_artifact(data, gate, provenance), indent=2))


if __name__ == "__main__":
    main()
