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
