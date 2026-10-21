# Phase 3 preparation evidence

Status: **INCOMPLETE — real T4 inference ran; strict internal parity fails**.

Qwen/GPT-2 forward-policy support was pushed in `21ccf4f`. This preparation
adds the reproducible notebook, isolated installation plan, correctness gate,
loopback SSE benchmark, raw-output validation and export workflow. It does not
contain a GPU benchmark result or claim that vLLM has executed successfully.

## Local verification

Environment used: Windows, Python 3.11.9, PyTorch 2.14.0+cpu,
Transformers 5.17.0. This is not the proposed Colab software environment.

- `python -m pytest -q`: 199 passed; six existing deprecation warnings.
- `python -m compileall -q src scripts tests`: passed.
- `npm test`: 38 passed in eight files.
- `npm run typecheck`: passed.
- `npm run audit:design`: passed.
- `npm run build`: passed; nine historical benchmark artifacts derived.
- Notebook validation: seven code cells, syntax valid, no outputs, matching
  the documented cells.
- Protocol checks include a real TCP/SSE exchange with randomly initialized
  tiny Qwen2 weights on CPU, not real-checkpoint GPU inference.
- Unit-only synthetic validator inputs remain in tests and are never exported
  as benchmark measurements.

## Run and return

Use [the seven exact cells](../../phase3-colab.md), also available in
[`notebooks/phase3_colab.ipynb`](../../../notebooks/phase3_colab.ipynb).
The user executes the authorized T4 environment; no notebook credentials or
direct remote access were supplied to the coding agent.

The run exports actual base/isolated environment inventories, package
resolution/install reports and freezes, hardware observations, real-model
correctness outputs, streaming event timestamps/IDs, raw warmups/repetitions,
server logs, allocator statistics and sampled external process memory.
Return the evidence ZIP even after a failure. No artifact is uploaded,
committed, published in the UI or labelled final automatically.

Required GPU correctness, production-engine execution, measured comparison,
results UI/docs, final public-demo review and project signoff remain pending.

## Colab preflight correction

The user returned diagnostics from the T4: base PyTorch 2.11.0+cu130 could
initialize CUDA normally, but removing `LD_LIBRARY_PATH` made initialization
fail with a missing-driver error. This was a workflow bug, not evidence of a
missing GPU. Both notebook and nested child processes now preserve inherited
driver-library paths. The regression test checks preservation and an originally
absent variable. No GPU correctness or benchmark success follows from this fix;
the corrected Colab execution still needs to run.

The next user run failed in Python 3.13's venv `ensurepip` bootstrap, before
Torch/vLLM installation. The underlying ensurepip output was not exposed in
that traceback, so no specific OS-package cause is asserted. Preparation now
avoids ensurepip and targets only new venvs using pip's documented `--python`
option. Bootstrap failure logs are surfaced and preserved. The failed partial
environments and earlier evidence must not be deleted or reused silently.

The pip-less bootstrap was exercised successfully in a fresh local Windows
Python 3.11 venv using host pip's `--python` option. This is not a Colab 3.13 or
GPU result. The single-cell launcher now sequences the existing guarded
scripts, verifies core resolution pins before installation, prints stage/live
progress with a quiet-stage heartbeat, and exports a per-run evidence ZIP after
success or a caught failure. Unit tests verify fail-fast behavior, pin rejection,
and notebook syntax/documentation equivalence. Real T4 execution is pending.
After the bootstrap/launcher correction, the local Python suite passed all 203
tests; compilation and diff whitespace checks also passed.

## Returned GPU correctness run

The user supplied `phase3-colab-20261021T080713847703Z-2ec61482.zip` from source
`e699cd8`. Its 35 manifest digests and 37 source-file digests matched. The
isolated stacks and FP16 CUDA checks passed, as did 203 tests on Colab. Actual
Qwen FP16 GPU outputs matched all 24 references at capacity 1, but differed for
two requests at capacity 2. The gate stopped before GPU HTTP/SSE, vLLM inference
or benchmark measurement. No performance artifact is available to publish.

[The targeted parity diagnostic](../../phase3-parity-diagnostic.md) reuses the
original installed runtime and cached checkpoint offline, records actual
cohort/logit/cache comparisons, and exports JSON only. It makes no runtime
change or relaxed correctness policy.

## Returned parity diagnostic

The user supplied `phase3-colab-diagnostic-20261021T115412030140Z-0cd5ea86.zip`
from `eb5dec7`. The manifest digest and all 38 recorded source digests matched.
ZIP SHA256: `3f123c5c9a6f4eb37325d5f2ef7df59dbd41f1e4c50040e5a4470a2b5f02fd01`.
Contained `diagnostic.json` SHA256:
`2291812d633793bfb7d982b2c5aaa824959238f671e464e7e7d8538b70178c21`.
It reused Torch 2.10.0+cu129, Transformers 5.17.0, the T4, the pinned Qwen
revision and FP16/SDPA. Both mismatches reproduced at the same token offsets.
Both frames passed all recorded input/mask/position checks and all 24 layers
of collation/splitting were bitwise exact. Repeated forwards were stable within
each execution shape. Independent-vs-batch top token choices differed at tied
or near-tied FP16 scores. Reference-cache substitution alone did not restore
the independent token. This supports, but does not fully isolate, a numerical
execution-shape explanation; it is not a passed correctness gate.

[The bounded numerical probe](../../phase3-numerics-probe.md) keeps FP16 weights,
the fixed workload and exact parity checks. It records three explicitly changed
numeric/attention policies in separate processes. No runtime default is changed,
and experiment reports cannot authorize benchmarks. The returned GPU execution
is documented below; no performance result or project completion is claimed.
Local verification after adding the controlled probes: 214 Python tests passed
with six existing warnings; compilation and diff whitespace checks passed.
These local checks use CPU/synthetic fixtures, not the new T4 profiles.

## Returned numerical experiments

The user supplied `phase3-colab-numerics-20261021T121338807622Z-1e364bfe.zip`
from `eab17d0`. All seven manifest hashes passed; every profile's 39 recorded
source hashes matched the submitted checkout. ZIP SHA256:
`05709396c28b0f72a22eaf7fc9740ff06cabf59212c216ff3bd544a4a6399918`.
The original model revision, T4, installed runtime versions and FP16 dtype
were retained, and each requested numerical setting was recorded as applied.

All three profiles passed all 24 requests at capacity 1. Capacity-2 exact
complete-output parity was 21/24 with SDPA/full reduction, 23/24 with
math-only/full reduction, and 21/24 with eager/full reduction. Each profile
stopped on its first failed parity assertion after retaining its full capacity-2
cohort. Capacity 4, HTTP/SSE, vLLM inference and benchmarks were not reached.
None of these policies is adopted as a fix, and no failed request is excluded.

[The same probe](../../phase3-numerics-probe.md) now provides a single FP32
math-policy sensitivity control using the existing environment. This is not
an FP16 correctness pass or permission to benchmark; the primary FP16 target
and exact parity requirement remain unchanged. The returned T4 control is
documented below. No additional environment installation is required.
After adding this control, all 217 local Python tests passed with six existing
warnings; compilation and whitespace checks passed. These checks do not
substitute for real GPU execution. The primary FP16 parity gate remains failed.

## Returned FP32 sensitivity control

The user supplied `phase3-colab-fp32-control-20261021T123720551114Z-2af3282f.zip`
from `c4b18e2`. All three manifest hashes and all 39 source hashes matched.
ZIP SHA256: `a09a9d90b40f40fd2817207800c7998182d271d04da5fd37ddb99f4597135380`.
Contained `math-fp32-control.json` SHA256:
`3520e981be436b00c5e13f7d9fb2cf47c56c59e859e2f2adbd4db4ec2262b136`.
The pinned Qwen/T4/software environment was retained. FP16-rounded parameter
values were promoted to FP32 for this diagnostic only, with math-only SDPA.

- Exact independent-vs-batched parity: 24/24 at each of capacities 1, 2 and 4.
- Eight short manual-vs-cached checks passed.
- Ordinary loopback HTTP passed.
- All 24 loopback SSE requests succeeded with exact reference IDs; all 1,760
  token IDs matched their recorded arrival-event concatenations and every
  request's prompt IDs were verified.
- Zero-budget generation made zero model executions; forced EOS ended after
  one selected token; the dynamic cohort/cache trace assertions passed.

This validates those paths under the diagnostic FP32 policy on the submitted
workload, not pure FP16 or arbitrary inputs. It is not a public-site request,
vLLM execution or measured comparison. The allocator observations are retained
in the diagnostic bundle, not promoted to benchmark memory measurements.
The required primary FP16 run remains failed; the project is incomplete.

The next bounded diagnostic retains the FP16 model and recomputes only its
vocabulary projection in FP32. This mixed-compute experiment is explicitly
labelled and rejected by the benchmark guard. Production source/defaults are
unchanged; no stability policy is adopted without real evidence and review.
Local verification after adding the projection control: 222 Python tests
passed with six existing warnings; compilation and whitespace checks passed.
The tied-weight, unchanged-parameter and diagnostic cleanup checks are local
CPU/unit tests, not evidence that this control passes on the T4.

## Returned projection control and approved precision change

The user returned `phase3-colab-fp32-head-control-20261021T125129948472Z-29f615d4.zip`
from `a78bbb1`. All three manifest hashes and 39 source hashes matched.
The control kept every parameter FP16 and produced finite FP32 projection
logits, but failed short uncached/manual parity on request-02. It did not
reach continuous batching, HTTP/SSE, vLLM or performance measurement. Changing
the final projection alone is therefore not a verified fix.

Amar explicitly approved an FP32 comparison for both engines on 2026-10-21,
with exact internal parity still mandatory and failed FP16 evidence preserved.
[The guarded continuation](../../phase3-fp32-comparison.md) loads the original
checkpoint natively as FP32 for both engines, checks shared serving/correctness
policy, runs a smoke gate and only then attempts the full comparison. This
replaces neither the failed evidence nor the previous diagnostic with a fake
pass. Actual native-FP32 correctness/vLLM inference/measurements remain pending.
Local preparation checks passed: 228 Python tests with six existing warnings,
compilation, and diff whitespace checks. The new policy tests use CPU/synthetic
fixtures only. The existing notebook was reached in the authorized Chrome
profile; its save-conflict warning was not overridden and a four-cell notebook
backup was downloaded before adding any new execution cell.
