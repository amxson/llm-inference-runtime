# Phase 3: one-cell Colab launcher

This is a **fresh-environment reproduction**, not a recovery cell for an
already-running benchmark. Do not run it while a comparison is in progress.
For the already installed environments use the [guarded continuation](phase3-fp32-comparison.md).
Both engines use the user-approved native FP32 profile after the retained FP16
failures; internal exact token parity is still mandatory. vLLM uses unmodified
TRITON_ATTN with enforce-eager, not a best-tuned production configuration.

Use this instead of the seven manual cells, in the existing authorized T4
notebook. It is self-contained: do not rerun the old Cell 1 or restart the
runtime first. Existing clean clones are updated without discarding tracked
changes; failed environments and evidence are left untouched. Every attempt
uses fresh output/environment paths. Set `STOP_AFTER` to `setup`, `correctness`,
`smoke` or `all`. Default `all` includes the full benchmark, only after all gates.

The same guarded scripts perform numbered stages, with live output and a
30-second heartbeat while a stage is quiet. Base Colab packages and driver
library paths are preserved. No apt install, base pip upgrade, premium GPU,
public tunnel or paid infrastructure is requested. The new venv bootstrap
does not use the failing `ensurepip` step.

For an exact code snapshot set `COMMIT` to the full commit supplied in the chat.
The default takes one recorded snapshot of `origin/main` before execution.
Once execution starts, do not edit or update the checkout. If a stage fails,
later stages stop and the launcher exports the available JSON/text evidence.
Review and return that ZIP; it is not automatically committed or published.
Stopping the entire runtime can prevent export, so download available evidence
before disconnecting it. GPU compatibility/results are not assumed from local
tests: real Colab execution remains required.

```python
from pathlib import Path
from datetime import datetime, timezone
import os, subprocess, sys, uuid

COMMIT = None  # Optional: full correction commit supplied in the chat.
STOP_AFTER = "all"  # setup / correctness / smoke / all
ROOT = Path("/content/llm-inference-runtime-phase3")
ENV = os.environ.copy()  # Keep Colab's working NVIDIA driver-library paths.
ENV.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
           VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")

if ROOT.exists():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, env=ENV, text=True)
    if dirty.strip():
        raise RuntimeError("Tracked changes exist; stop rather than overwrite them.")
    subprocess.run(["git", "fetch", "origin"], cwd=ROOT, env=ENV, check=True)
else:
    subprocess.run(["git", "clone",
                    "https://github.com/amxson/llm-inference-runtime.git", str(ROOT)],
                   env=ENV, check=True)

subprocess.run(["git", "checkout", "--detach", COMMIT or "origin/main"],
               cwd=ROOT, env=ENV, check=True)
os.chdir(ROOT)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
RUN = ROOT / "benchmark_results" / "phase3-colab" / RUN_ID
ENVS = Path("/content/phase3-envs") / RUN_ID
print("Evidence:", RUN, flush=True)

command = [sys.executable, "-u", "scripts/phase3/run_colab.py",
           "--output", str(RUN), "--env-root", str(ENVS), "--through", STOP_AFTER,
           "--execution-policy", "fp32-math",
           "--model-revision", "060db6499f32faf8b98477b0a26969ef7d8b9987"]
with subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    result = process.wait()

archive = RUN.parent / f"phase3-colab-{RUN_ID}.zip"
if archive.exists():
    print("Download this evidence ZIP and return it in the chat:", archive)
    from google.colab import files
    files.download(str(archive))
if result:
    raise RuntimeError("Workflow stopped at its first failure. Return the ZIP/logs; do not run later stages manually.")
print("Requested stages passed. Return the ZIP for review; project signoff is still pending.")
```
