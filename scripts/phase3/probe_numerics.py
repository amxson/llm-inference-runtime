"""Bounded numerical controls, not benchmarks or automatic fixes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from common import environment, save_json, source_provenance, utc_now
from run_colab import export_evidence, run_command

PROFILES = ("sdpa-full-reduction", "math-full-reduction", "eager-full-reduction")
HARNESS_CHANGES = {"scripts/phase3/correctness.py", "scripts/phase3/benchmark.py"}


def check_prior(prior, provenance, env):
    if (prior.get("status"), prior.get("model"), prior.get("dtype")) != (
            "fail", "Qwen/Qwen2.5-0.5B", "float16"):
        raise ValueError("Expected the original failed Qwen FP16 artifact")
    for package in ("torch", "transformers"):
        if env["versions"][package] != prior["environment"]["versions"][package]:
            raise ValueError(f"Installed {package} changed; do not reinstall silently")
    if env["gpus"] != prior["environment"]["gpus"]:
        raise ValueError("GPU inventory differs from the failed run")
    changed = []
    for name, digest in prior["source_file_sha256"].items():
        if provenance["source_file_sha256"].get(name) != digest:
            if name not in HARNESS_CHANGES:
                raise ValueError(f"Unexpected source change: {name}")
            changed.append(name)
    return sorted(changed)


def run_profiles(output, prior, execute=run_command, *, fp32_control=False):
    """Expected parity failures continue; installation/environment errors stop."""
    rows = []
    profiles = ("math-fp32-control",) if fp32_control else PROFILES
    for index, profile in enumerate(profiles, 1):
        target = output / f"{profile}.json"
        policy = "math-full-reduction" if fp32_control else profile
        command = [sys.executable, "scripts/phase3/correctness.py", "--model", prior["model"],
                   "--revision", prior["revision"], "--concurrencies", "1", "2", "4",
                   "--numerical-experiment", policy, "--output", target]
        if fp32_control:
            command.append("--diagnostic-fp32")
        command_error = None
        try:
            execute(command, output / f"{profile}.txt",
                    f"{index}/{len(profiles)}: strict diagnostic correctness: {profile}")
        except RuntimeError as error:
            command_error = str(error)
            if not target.is_file():
                raise
        result = json.loads(target.read_text(encoding="utf8"))
        if (result.get("workload") != prior["workload"] or
                result.get("experiment_only") is not True or
                result.get("numerical_experiment") != policy or
                result.get("dtype") != ("float32" if fp32_control else "float16") or
                bool(result.get("diagnostic_fp32")) != fp32_control):
            raise ValueError("Experiment configuration/workload differs unexpectedly")
        status, error = result["status"], result.get("error")
        # Only failed assertions about parity are expected experimental results.
        # A missing driver, cache, model, OOM or HTTP error is not a parity result.
        if status != "pass" and not (status == "fail" and error and
                error.startswith("ValueError:") and "parity failed" in error.lower()):
            raise RuntimeError(f"Unexpected failure in {profile}: {error or command_error}")
        if status == "pass" and command_error:
            raise RuntimeError(f"Worker failed after writing a pass: {command_error}")
        rows.append({"profile": profile, "status": status, "error": error,
                     "dtype": result["dtype"],
                     "passing_capacities": [r["capacity"] for r in result.get("continuous", [])
                                            if r["exact_token_parity"]],
                     "http_sse_passed": status == "pass" and "http_sse" in result})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-control", action="store_true",
                        help="Run only the FP32 math-policy sensitivity control, not FP16 experiments")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh directory; prior evidence is immutable")
    args.output.mkdir(parents=True)
    report = {"kind": "numerical_experiment_not_benchmark", "benchmark_valid": False,
              "status": "running", "started_at_utc": utc_now(),
              "profiles": ["math-fp32-control"] if args.fp32_control else list(PROFILES),
              "fp32_control": args.fp32_control}
    try:
        if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            raise ValueError("Require offline cached weights; no installation/download is performed")
        report.update(source_provenance())
        prior_bytes = args.prior.read_bytes()
        prior = json.loads(prior_bytes)
        report["prior_correctness_sha256"] = hashlib.sha256(prior_bytes).hexdigest()
        report["environment"] = environment()
        report["prior_source_commit"] = prior["source_commit"]
        report["changed_prior_source_files"] = check_prior(prior, report, report["environment"])
        report["results"] = run_profiles(args.output, prior, fp32_control=args.fp32_control)
        report["status"] = "experiments_complete"
        report["note"] = "Review results before changing serving policy. No benchmark or automatic promotion ran."
        print(json.dumps(report["results"], indent=2), flush=True)
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at_utc"] = utc_now()
        save_json(args.output / "experiments.json", report)
        export_evidence(args.output)


if __name__ == "__main__":
    main()
