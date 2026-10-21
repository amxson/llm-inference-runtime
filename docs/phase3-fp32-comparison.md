# Approved FP32 comparison on the existing Colab T4

On 2026-10-21 Amar approved changing **both** compared engines to clearly
labelled FP32 while retaining exact internal token parity and all failed FP16
evidence. This is a documented precision exception, not a claim that FP16 was
fixed or that failed outputs can be excluded.

The final FP16/head-only control from `a78bbb1` failed the short uncached/manual
check for request-02, before batching or HTTP checks. All three archive hashes
and 39 source hashes matched. ZIP SHA256:
`3d42e264bb58a89b123a5c063141152840b9a6f75f3bb87c2c26ba586f08ef27`.
The earlier full-FP32 control passed the fixed workload, but loaded FP16 and
promoted its rounded values; that diagnostic cannot authorize this benchmark.

## Shared policy

`scripts/phase3/policies.py` defines `fp32-math`. Correctness and our measurement
server both load the original pinned checkpoint **directly as FP32**, with
math-only SDPA, no FP16 promotion, no projection hook and no TF32 matmul.
The benchmark validates the actual serving policy against the passing gate.
The ordinary public CPU/distilgpt2 demo and runtime implementation are unchanged.

Unmodified vLLM 0.17.1 loads the same checkpoint/revision directly as FP32, uses
`TRITON_ATTN`, disables prefix caching, caps context at 2048 and uses
`--enforce-eager`. CUDA graphs/compile optimizations are therefore disabled in
this comparison; do not describe this configuration as best-tuned vLLM.
Its pinned source declares [FP32 support in Triton attention](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/attention/backends/triton_attn.py)
and [FP32 support on Turing GPUs](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/platforms/cuda.py).
Declarations are preflight checks, not proof that actual T4 inference works.
No vLLM source/kernel is patched, no quantization is used and no packages are
reinstalled. Inference must succeed through the native HTTP/SSE server.

## One guarded continuation

`scripts/phase3/resume_colab.py` uses the already installed isolated runtime and
vLLM interpreters. It verifies their actual versions/CUDA execution, GPU
inventory, original runtime source hashes and cached model revision, then runs:

1. CPU regression/compile checks.
2. Fresh native-FP32 correctness at capacities 1/2/4, including HTTP/SSE.
3. Real two-engine streaming smoke, which is not final performance evidence.
4. The full 24-prompt mixed 32/64/128-token workload, capacities 1/2/4,
   two warmup cohorts per session/configuration, ten measured repetitions per
   engine/configuration, balanced ABBA sessions and raw event collection.
5. Strict raw-artifact validation and a SHA256-manifest ZIP.

Any error stops later stages and exports the available logs/reports. Previous
runs are never overwritten or deleted. Successful stages still require evidence
review and do not automatically publish results or mark the project complete.
`--through smoke` or `--through correctness` can bound troubleshooting runs.

The chat supplies the tested pinned launcher for the existing notebook. Its
environment preserves inherited NVIDIA driver-library paths and enables
offline Hugging Face loading. Use a fresh output directory and do not change
the checkout or run competing GPU work while measurements are in progress.
No premium GPU, paid resource, account, credential, tunnel or public backend
hosting change is needed.

## Recovery and reproduction

The approved run executed source `14700d46569f2889d9b3ed110cbccf1b293c6c38`
in the existing environments. Its native FP32 correctness gate passed all 24
requests at capacities 1/2/4, eight manual checks and 24 real SSE requests
(1,760 token IDs), plus the dynamic-admission/zero/EOS checks. The two-engine
smoke artifact passed consistency/provenance review: four measured tiny cohorts,
16 successful requests and no failures. It remains `final_eligible: false`.
The full comparison subsequently passed: 60 measured cohorts, 1,440 successful
requests and no failures. [Original artifacts and review](evidence/phase3/INDEX.md).

After the local battery shutdown, the notebook reopened an older saved version,
but the original run directory and installed environments remained available.
The benchmark log continued advancing into the second vLLM session; no cell was
rerun and no runtime was restarted to recover the evidence. A notebook save
conflict is therefore not itself proof that remote computation stopped.

The [fresh-run one-cell notebook](phase3-colab-one-cell.md) and seven-cell guide
now explicitly select this approved native FP32 profile and the same immutable
checkpoint. They create new environments; they are **not** recovery commands for
an active run. The fresh-run setup changes are locally tested command planning,
not a claim that another complete Colab installation/benchmark was executed.
`node scripts/sync_phase3_notebooks.mjs --write` mechanically mirrors the guides
into output-free notebooks; omit `--write` to check for drift.

To review published raw evidence after later UI/docs commits, use:

```powershell
python scripts/phase3/validate.py benchmarks/results/gpu_final_comparison.json --correctness docs/evidence/phase3/correctness/correctness.json --source-commit 14700d46569f2889d9b3ed110cbccf1b293c6c38
```

This requires the measured commit in local Git history and hashes its immutable
source blobs. It does not substitute today's working files or relax collection's
clean-checkout requirement. The raw final artifact path is usable only once the
completed evidence is actually published. Validation checks internal consistency
and source correspondence, not independent attestation of remote GPU execution.
