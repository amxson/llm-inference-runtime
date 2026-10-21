"""One fail-fast Colab launcher, with live stages and automatic evidence export."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

from common import ROOT, child_environment, save_json, source_provenance, utc_now


def run_command(command, log_path, label):
    """Stream child output and retain it; quiet downloads/JIT get a heartbeat."""
    print(f"\n{label}\nLog: {log_path}", flush=True)
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(30):
            print(f"  Still running: {label} ({time.monotonic() - started:.0f}s).", flush=True)

    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    try:
        with Path(log_path).open("x", encoding="utf8") as log:
            with subprocess.Popen([str(arg) for arg in command], cwd=ROOT, env=child_environment(),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
        if code:
            raise RuntimeError(f"{label} failed (exit {code}); see {log_path}")
        print(f"PASS: {label} ({time.monotonic() - started:.1f}s)", flush=True)
    finally:
        stopped.set()
        ticker.join(1)


def verify_resolution(directory):
    expected = {"runtime": {"torch": "2.10.0+cu129", "transformers": "5.17.0"},
                "vllm": {"torch": "2.10.0+cu129", "transformers": "4.57.6", "vllm": "0.17.1"}}
    for name, pins in expected.items():
        report = json.loads((directory / f"resolve-{name}.json").read_text(encoding="utf8"))
        selected = {entry["metadata"]["name"].lower().replace("_", "-"): entry["metadata"]["version"]
                    for entry in report["install"]}
        if any(selected.get(package) != version for package, version in pins.items()):
            raise RuntimeError(f"Unexpected {name} package resolution; stop before installing: {selected}")
        print(f"Reviewed pins matched for {name}: {pins}", flush=True)


def export_evidence(directory):
    """Export only this fresh run's JSON/text evidence, never caches/weights."""
    files = sorted(p for p in directory.rglob("*")
                   if p.is_file() and not p.is_symlink() and p.suffix in {".json", ".txt"})
    save_json(directory / "SHA256SUMS.json", {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    archive = directory.parent / f"phase3-colab-{directory.name}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in {".json", ".txt"}:
                target.write(path, arcname=str(path.relative_to(directory)))
    print(f"\nEVIDENCE ZIP: {archive}", flush=True)
    print(f"ZIP SHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--through", choices=("setup", "correctness", "smoke", "all"), default="all")
    args = parser.parse_args()
    if sys.platform != "linux" or args.output.exists() or args.env_root.exists():
        parser.error("Use Linux and fresh output/environment paths. Prior attempts are never overwritten.")
    provenance = source_provenance()
    args.output.mkdir(parents=True)
    status = {"status": "running", "started_at_utc": utc_now(), "through": args.through, **provenance}
    runtime_python = args.env_root / "runtime/bin/python"
    vllm_python = args.env_root / "vllm/bin/python"
    stage = "initialization"

    def execute(command, name, label):
        run_command(command, args.output / name, label)

    try:
        stage = "1/7: inventory actual GPU/base environment"
        execute([sys.executable, "scripts/phase3/preflight.py", "--report-only", "--output",
                 args.output / "base-environment.json"], "base-environment.txt", stage)
        execute(["git", "log", "-12", "--oneline"], "git-log.txt", "Record source history")
        stage = "2/7: bootstrap isolated venvs and resolve dependencies"
        setup = [sys.executable, "scripts/phase3/setup_colab.py", "--env-root", args.env_root,
                 "--output", args.output / "setup"]
        execute(setup, "setup-plan.txt", stage)
        verify_resolution(args.output / "setup")
        stage = "3/7: install isolated stacks and verify FP16 CUDA"
        execute([*setup, "--install"], "setup-install.txt", stage)
        execute([vllm_python, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                "vllm-server-help.txt", "Check vLLM CLI")
        if args.through != "setup":
            stage = "4/7: CPU regression tests and compile checks"
            execute([runtime_python, "-m", "pytest", "-q"], "python-tests.txt", stage)
            execute([runtime_python, "-m", "compileall", "-q", "src", "scripts", "tests"],
                    "compileall.txt", "Compile source/tests")
            execute(["git", "diff", "--check"], "git-diff-check.txt", "Check source whitespace")
            stage = "5/7: real Qwen FP16 GPU correctness and HTTP/SSE"
            code = ("from huggingface_hub import HfApi; import json,sys; "
                    "d={'model':'Qwen/Qwen2.5-0.5B','revision':HfApi().model_info('Qwen/Qwen2.5-0.5B').sha}; "
                    "f=open(sys.argv[1],'x'); json.dump(d,f,indent=2); f.close(); print(json.dumps(d))")
            execute([runtime_python, "-c", code, args.output / "model.json"], "model-resolution.txt", "Pin model revision")
            model = json.loads((args.output / "model.json").read_text())
            execute([runtime_python, "scripts/phase3/correctness.py", "--model", model["model"],
                     "--revision", model["revision"], "--concurrencies", "1", "2", "4",
                     "--output", args.output / "correctness.json"], "correctness.txt", stage)
            if args.through != "correctness":
                command = [runtime_python, "scripts/phase3/benchmark.py", "--correctness", args.output / "correctness.json",
                           "--runtime-python", runtime_python, "--vllm-python", vllm_python]
                stage = "6/7: real two-engine smoke (not final measurements)"
                execute([*command, "--smoke", "--repetitions", "2", "--concurrencies", "1",
                         "--output", args.output / "smoke"], "smoke.txt", stage)
                if args.through == "all":
                    stage = "7/7: full benchmark and raw-artifact validation"
                    execute([*command, "--repetitions", "10", "--concurrencies", "1", "2", "4",
                             "--output", args.output / "comparison"], "benchmark.txt", stage)
                    execute([runtime_python, "scripts/phase3/validate.py", args.output / "comparison/gpu_final_comparison.json",
                             "--correctness", args.output / "correctness.json"], "validation.txt", "Validate measured evidence")
        status["status"] = "pass"
        print(f"\nRequested stages passed: {args.through}. Remote evidence still needs review; project NOT marked complete.", flush=True)
    except BaseException as error:
        status["status"] = "fail"
        status["failed_stage"] = stage
        status["error"] = f"{type(error).__name__}: {error}"
        print(f"\nSTOPPED at {stage}: {error}\nLater stages were not run.", flush=True)
        raise
    finally:
        status["finished_at_utc"] = utc_now()
        save_json(args.output / "run-status.json", status)
        export_evidence(args.output)


if __name__ == "__main__":
    main()
