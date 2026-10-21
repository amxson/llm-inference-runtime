# Phase 3 preparation evidence

Status: **INCOMPLETE — real Colab T4 execution pending**.

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
change or relaxed correctness policy; GPU diagnosis remains pending execution.
