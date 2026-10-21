# FP16 batch-parity diagnostic — not a correctness pass

The returned `20261021T080713847703Z-2ec61482` evidence bundle passed installation,
both FP16 CUDA matmul preflights and 203 tests. Real Qwen2.5-0.5B execution on
the T4 matched all 24 independent cached outputs at capacity 1. Capacity 2
differed for request-01 at generated token 36 and request-18 at token 29.
Capacity 4, real GPU HTTP/SSE and production-engine inference were not reached.
All 35 archive-manifest hashes and all 37 recorded source hashes were checked
locally against the submitted run; no artifact has been promoted to a result.

Run `scripts/phase3/diagnose_parity.py` using the **existing runtime venv**, with
`--prior` pointing to that run's `correctness.json` and `--output` a fresh ignored
directory. Both `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are required:
weights/tokenizer are reused from the existing cache, with no package install
or model download. The chat supplies a single pinned recovery/diagnostic cell.
Do not restart the Colab runtime first. Missing environments/cache produce an
error rather than silently reinstalling anything.

The diagnostic adds observations around an ordinary `CachedContinuousBatcher`.
It does not replace selected tokens, alter the scheduler, change precision or
attention settings, relax exact parity, or run a benchmark. Original runtime
file hashes and installed Torch/Transformers versions must match the failed
run. Inference remains FP16; FP32 copies are used only to measure logit/cache
differences. No tensor/model files are exported.

At the first observed disagreement (or the prior divergence step if it does
not reproduce), it records the cohort, prefix, masks, positions and recent
admission/decode trace. For the same actual token prefix it compares:

- The captured batch with an exact-input batch replay.
- Independent cached reference forwards, repeated twice.
- Single-row and batched full-prefix uncached forwards.
- Single-row execution with the actual observed request cache.
- The same batch with only the target row's cache replaced by the independent
  reference cache.

Per-layer K/V comparisons distinguish collation/splitting from differences
already present in a request's cache. Top token scores, argmax IDs, exact ties
and top-two margins are retained. Numeric deltas are evidence to investigate,
not a tolerance-based replacement for the required token equality. A failed
reproduction is explicitly labelled, never presented as a fix.

The JSON/text evidence ZIP is created automatically, including partial error
reports when possible. Return it for analysis; do not resume the full benchmark
or edit the old correctness artifact. GPU execution of this diagnostic remains
pending user execution; local tests use tiny randomly initialized CPU models
and clearly synthetic old mismatches.

## One-cell execution in the existing Colab session

Set `COMMIT` to the full diagnostic commit supplied in the chat. The default
fetches one recorded snapshot of main. This cell requires the prior run and
its installed interpreter still to exist; it does not recreate either.

```python
from pathlib import Path
from datetime import datetime, timezone
import os, subprocess, uuid

COMMIT = None  # Full diagnostic commit supplied in the chat.
ROOT = Path("/content/llm-inference-runtime-phase3")
PRIOR_ID = "20261021T080713847703Z-2ec61482"
PRIOR = ROOT / "benchmark_results/phase3-colab" / PRIOR_ID / "correctness.json"
PYTHON = Path("/content/phase3-envs") / PRIOR_ID / "runtime/bin/python"
if not PRIOR.is_file() or not PYTHON.is_file():
    raise RuntimeError("The original run/environment is unavailable. Stop and tell me; do not reinstall.")

ENV = os.environ.copy()
ENV.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
dirty = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, env=ENV, text=True)
if dirty.strip():
    raise RuntimeError("Tracked changes exist; stop rather than overwrite them.")
subprocess.run(["git", "fetch", "origin"], cwd=ROOT, env=ENV, check=True)
subprocess.run(["git", "checkout", "--detach", COMMIT or "origin/main"],
               cwd=ROOT, env=ENV, check=True)
DIAG_ID = "diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
OUT = ROOT / "benchmark_results/phase3-colab" / DIAG_ID
command = [str(PYTHON), "-u", "scripts/phase3/diagnose_parity.py",
           "--prior", str(PRIOR), "--output", str(OUT)]
with subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    result = process.wait()
archive = OUT.parent / f"phase3-colab-{DIAG_ID}.zip"
if archive.exists():
    from google.colab import files
    files.download(str(archive))
if result:
    raise RuntimeError("Diagnostic stopped. Return the downloaded ZIP or console output.")
print("Diagnostic captured. Return the ZIP; do not resume the benchmark yet.")
```
