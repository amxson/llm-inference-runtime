# Controlled FP16 correctness experiments

The returned diagnostic from `eb5dec7` reproduced both capacity-2 disagreements
on the T4, with the same model revision and installed environment. All five
recorded mask/input/position invariants passed for both frames. K/V collation
and splitting preserved all 24 layers exactly. Repeated independent and batch
forwards reproduced their respective choices. Replacing the target row's cache
with its independent reference cache did not restore the independent token.

At request-01 token 36, the independent result tied ` a` and ` the`, choosing
token 264; the actual batch preferred token 279 by 0.015625. At request-18 token
29, the independent result preferred `8` (23) by 0.015625, while the batch tied
its top two scores and selected `4` (19). The largest logit differences versus
the independent vector were 0.0234375 and 0.033203125 respectively. These findings
support a numerical execution-shape hypothesis, not a claim that every cache
or attention implementation is proven correct or that a specific kernel is
the root cause. The strict token parity requirement remains unchanged.

`probe_numerics.py` runs three separate processes, each using the existing
`correctness.py` checks with all 24 prompts, budgets 32/64/128, capacities 1/2/4,
and real HTTP/SSE if internal checks pass:

1. SDPA with FP16 GEMM reduced-precision reductions disabled.
2. The same setting plus math-only SDPA, with low-precision math-SDPA reduction
   explicitly disabled. Process-wide flags also cover the HTTP server thread.
3. The same GEMM setting with Transformers eager attention instead of SDPA.

Model weights remain FP16. Full-precision internal accumulation is not FP32
model execution; math SDPA can use FP32 intermediate buffers, costing memory
and time. This is not a performance comparison. Each profile creates fresh
independent references under its own policy: it must match its own batched
outputs exactly, not force the previous profile's tokens.

There is no promise these profiles will pass. Expected parity failures are
retained and the next profile runs; unexpected environment/OOM/HTTP failures
stop the experiment. All reports/logs, including partial failures, are exported
with SHA256 hashes. No packages are installed, no weights are downloaded, no
scheduler/runtime source is changed, and no benchmark or vLLM server is run.
The original source comparison explicitly permits only the changed correctness
harness and benchmark rejection guard; other previously recorded source files
must still match. The new reports identify their new source snapshot.

Even a passing experiment is **not eligible for benchmarking**. The benchmark
gate explicitly rejects it because the ordinary serving process has not yet
adopted that policy. Any policy selected after evidence review must be applied
consistently to the reference, serving and measurement paths, recorded, and
reverified before performance measurements. No tolerance, tie override,
per-request inference fallback or custom production-engine patch is introduced.

Primary references: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html),
[CUDA backend controls](https://docs.pytorch.org/docs/2.10/backends.html), and
[Transformers attention backends](https://huggingface.co/docs/transformers/v5.17.0/attention_interface).
These explain the experiment choices; they do not establish this failure's
root cause or guarantee batch-independent greedy tokens.

## Returned FP16 results and next precision control

The returned `numerics-20261021T121338807622Z-1e364bfe` run from `eab17d0`
completed all three experiments. All seven manifest hashes matched; all 39
source hashes in each profile matched that checkout. The model, revision,
Torch/Transformers versions, FP16 dtype, GPU and recorded flags were checked.
All profiles matched 24/24 independent references at capacity 1, but none
passed capacity 2:

| Policy | Exact requests at capacity 2 | First differing generated tokens |
| --- | --- | --- |
| SDPA, full GEMM reduction | 21/24 | request-01: 36; request-08: 2; request-18: 29 |
| Math-only SDPA, full GEMM reduction | 23/24 | request-02: 111 |
| Eager, full GEMM reduction | 21/24 | request-01: 36; request-02: 2; request-20: 49 |

These are complete-output equality counts, not next-token accuracy estimates.
Capacity 4, HTTP/SSE, vLLM inference and performance measurements were not
reached. The first eight short uncached/manual checks passed under each
profile. None of the tested FP16 policies is a verified fix; no policy is
adopted or result promoted.

The next bounded control uses `probe_numerics.py --fp32-control`. It runs only
the math-only policy with **FP32 weights/activations** rather than FP16, and
compares fresh independent FP32 references with FP32 batching. It retains all
24 prompts, budgets 32/64/128, capacities 1/2/4 and subsequent HTTP/SSE checks.
It changes precision, not the token selection rule or workload. This is solely
a sensitivity control, never the primary FP16 run or a performance result.
It first loads the pinned weights as FP16, then promotes those same rounded
parameter values to FP32; it does not recover extra checkpoint precision.
Even if it passes, the FP16 requirement remains unmet. If it fails, FP32
roundoff is still possible; failure alone does not establish a cache bug.

The control runs in a separate process using existing installed packages and
cached weights. It can use more memory/time and may fail; partial reports are
exported. The ordinary correctness command still defaults to FP16. Explicit
FP32 is permitted only alongside the diagnostic math policy, recorded in its
report, and rejected by the benchmark gate. No automatic policy selection or
precision conversion in the production runtime is introduced. Real T4
execution of this FP32 control is pending user execution.

## One cell in the existing T4 session

Set `COMMIT` to the full commit supplied in chat. Do not restart the session or
rerun setup. Return the ZIP even when every profile fails.

```python
from pathlib import Path
from datetime import datetime, timezone
import os, subprocess, uuid

COMMIT = None  # Full FP32-control commit supplied in chat.
ROOT = Path("/content/llm-inference-runtime-phase3")
PRIOR_ID = "20261021T080713847703Z-2ec61482"
PRIOR = ROOT / "benchmark_results/phase3-colab" / PRIOR_ID / "correctness.json"
PYTHON = Path("/content/phase3-envs") / PRIOR_ID / "runtime/bin/python"
if not PRIOR.is_file() or not PYTHON.is_file():
    raise RuntimeError("Original environment/run unavailable. Stop and return this message; do not reinstall.")
ENV = os.environ.copy()
ENV.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
dirty = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, env=ENV, text=True)
if dirty.strip():
    raise RuntimeError("Tracked changes exist. Stop rather than overwrite them.")
subprocess.run(["git", "fetch", "origin"], cwd=ROOT, env=ENV, check=True)
subprocess.run(["git", "checkout", "--detach", COMMIT or "origin/main"],
               cwd=ROOT, env=ENV, check=True)
RUN_ID = "fp32-control-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
OUT = ROOT / "benchmark_results/phase3-colab" / RUN_ID
command = [str(PYTHON), "-u", "scripts/phase3/probe_numerics.py", "--fp32-control",
           "--prior", str(PRIOR), "--output", str(OUT)]
with subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    result = process.wait()
archive = OUT.parent / f"phase3-colab-{RUN_ID}.zip"
if archive.exists():
    print("Return this ZIP:", archive)
    from google.colab import files
    files.download(str(archive))
if result:
    raise RuntimeError("Experiment stopped. Return the ZIP or console output.")
print("Experiments captured, not a benchmark pass. Return the ZIP; stop here.")
```
