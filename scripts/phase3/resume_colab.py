"""Resume the approved FP32 comparison in existing venvs; never reinstall."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from common import save_json, source_provenance, utc_now
from run_colab import export_evidence, run_command


def check_existing_run(prior, runtime, production, provenance):
    if prior.get("model") != "Qwen/Qwen2.5-0.5B" or len(prior.get("revision", "")) != 40:
        raise ValueError("Expected the original pinned Qwen run")
    if runtime["versions"]["torch"] != "2.10.0+cu129" or runtime["versions"]["transformers"] != "5.17.0":
        raise ValueError("Existing runtime environment differs; stop, do not reinstall")
    if (production["versions"]["torch"] != runtime["versions"]["torch"] or
            production["versions"]["vllm"] != "0.17.1" or
            production["versions"]["transformers"] != "4.57.6"):
        raise ValueError("Existing production environment differs; stop, do not reinstall")
    if runtime["gpus"] != production["gpus"] or runtime["gpus"] != prior["environment"]["gpus"]:
        raise ValueError("GPU inventory differs from the reviewed T4 run")
    for name, digest in prior["source_file_sha256"].items():
        if (name.startswith("src/") or name == "pyproject.toml") and provenance["source_file_sha256"].get(name) != digest:
            raise ValueError(f"Runtime source unexpectedly changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", choices=("correctness", "smoke", "all"), default="all")
    args = parser.parse_args()
    if sys.platform != "linux" or args.output.exists():
        parser.error("Use Linux and a fresh output directory")
    runtime_python, production_python = (args.env_root / name / "bin/python" for name in ("runtime", "vllm"))
    if not all(p.is_file() for p in (args.prior, runtime_python, production_python)):
        parser.error("Original run/installed environments missing; do not reinstall automatically")
    args.output.mkdir(parents=True)
    report = {"status": "running", "started_at_utc": utc_now(), "execution_policy": "fp32-math",
              "through": args.through, "dtype": "float32", "environment_reinstalled": False}
    stage = "inventory"

    def execute(command, filename, label):
        run_command(command, args.output / filename, label)

    try:
        if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            raise ValueError("Require offline cached checkpoint execution")
        report.update(source_provenance())
        prior_bytes = args.prior.read_bytes()
        prior = json.loads(prior_bytes)
        report["original_correctness_sha256"] = hashlib.sha256(prior_bytes).hexdigest()
        report["model"], report["revision"] = prior["model"], prior["revision"]
        execute([runtime_python, "scripts/phase3/preflight.py", "--output", args.output / "runtime-environment.json"],
                "runtime-preflight.txt", "1/5: existing runtime CUDA environment")
        execute([production_python, "scripts/phase3/preflight.py", "--check-fp32-triton", "--output",
                 args.output / "vllm-environment.json"], "vllm-preflight.txt", "1/5: existing vLLM CUDA/backend declarations")
        runtime = json.loads((args.output / "runtime-environment.json").read_text())
        production = json.loads((args.output / "vllm-environment.json").read_text())
        check_existing_run(prior, runtime, production, report)
        stage = "regression tests"
        execute([runtime_python, "-m", "pytest", "-q"], "python-tests.txt", "2/5: regression tests")
        execute([runtime_python, "-m", "compileall", "-q", "src", "scripts", "tests"], "compileall.txt", "Compile source/tests")
        stage = "native FP32 correctness and HTTP/SSE"
        execute([runtime_python, "scripts/phase3/correctness.py", "--model", prior["model"], "--revision", prior["revision"],
                 "--execution-policy", "fp32-math", "--concurrencies", "1", "2", "4", "--output",
                 args.output / "correctness.json"], "correctness.txt", "3/5: native FP32 correctness and HTTP/SSE")
        if args.through != "correctness":
            benchmark = [runtime_python, "scripts/phase3/benchmark.py", "--correctness", args.output / "correctness.json",
                         "--runtime-python", runtime_python, "--vllm-python", production_python]
            stage = "real two-engine streaming smoke"
            execute([*benchmark, "--smoke", "--repetitions", "2", "--concurrencies", "1", "--output", args.output / "smoke"],
                    "smoke.txt", "4/5: real two-engine streaming smoke; NOT final evidence")
            if args.through == "all":
                stage = "full comparison"
                execute([*benchmark, "--repetitions", "10", "--concurrencies", "1", "2", "4", "--output", args.output / "comparison"],
                        "benchmark.txt", "5/5: full FP32 comparison; 10 repetitions per engine/capacity")
                stage = "validation"
                execute([runtime_python, "scripts/phase3/validate.py", args.output / "comparison/gpu_final_comparison.json",
                         "--correctness", args.output / "correctness.json"], "validation.txt", "Validate raw measured evidence")
        report["status"] = "pass"
        print("Requested stages passed. Evidence still requires review; project not marked complete.", flush=True)
    except BaseException as error:
        report["status"], report["failed_stage"] = "fail", stage
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"STOP: {stage}: {error}; later stages did not run.", flush=True)
        raise
    finally:
        report["finished_at_utc"] = utc_now()
        save_json(args.output / "run-status.json", report)
        export_evidence(args.output)


if __name__ == "__main__":
    main()
