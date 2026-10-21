# LLM Inference Runtime

I built a small inference runtime on PyTorch and Hugging Face to understand what happens below `model.generate()`. Its own explicit token loops handle decoding, request state, KV caching, scheduling, batching and SSE delivery. GPT-2 and Qwen share the same scheduler through a small model-input adapter. This is an educational engineering project, not a replacement for vLLM, TGI or llama.cpp.

## Demo

[Open the profiler](https://llm-inference-runtime.vercel.app/#/runtime). Playground runs real **distilgpt2 / CPU** inference; health reports the responding engine and cache status. Replay and committed benchmark exploration work without a backend. The recorded Qwen GPU comparison is a separate Colab experiment, not Vercel GPU hosting. Public cold starts can be slow; telemetry is per server instance. Use non-sensitive prompts.

## What it does

- Explicit greedy decoding and separate temperature/top-k/top-p sampling.
- Request-local KV reuse, FIFO admission and cached/uncached continuous batching.
- Ragged prompt/cache collation, one-token decode and dynamic admission/removal.
- HTTP/SSE serving and a React profiler with live traces and independent replay.
- CPU cache-length, INT8 and draft/verify experiments, including negative results.

## GPU comparison

Same **Qwen2.5-0.5B**, Tesla T4 and **FP32**; 24 varied prompts with 32/64/128-token budgets, two excluded warmup cohorts per session/configuration, ten measured repetitions per engine/capacity and ABBA engine sessions. Latency is request median; throughput is median whole-cohort generated tokens/sec over loopback HTTP/SSE.

| Concurrency | Our latency ms | vLLM latency ms | Our tok/s | vLLM tok/s | vLLM/ours tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,171.24 | 1,679.99 | 29.36 | 38.03 | 1.30× |
| 2 | 2,494.51 | 1,723.22 | 49.06 | 70.94 | 1.45× |
| 4 | 3,021.65 | 1,741.34 | 76.78 | 126.97 | 1.65× |

All 1,440 measured requests succeeded; internal reference IDs and all 720 paired cross-engine outputs matched. FP32 was explicitly approved after strict FP16 parity failures, which remain preserved. Unmodified vLLM 0.17.1 uses TRITON_ATTN with enforce-eager and prefix caching disabled; this is **not best-tuned vLLM** or an FP16 result. Production engines have integrated executor/kernel/cache optimizations this project does not.

[Raw GPU data](benchmarks/results/gpu_final_comparison.json), [method and evidence](docs/evidence/phase3/INDEX.md), [reproducible Colab workflow](docs/phase3-colab-one-cell.md). The earlier [CPU length sweep](benchmarks/results/cache_length_sweep_cpu.json) and [cached continuous comparison](benchmarks/results/cached_continuous_batching_cpu.json) remain separate experiments, not GPU baselines.

## Architecture

```text
HTTP → FIFO admission → prompt prefill → request-local KV
Retained KV → padded collation → model adapter → [B,1] decode → split/update → SSE
Completion releases KV; waiting requests fill vacancies on the next tick
```

[Execution paths and API reference](docs/architecture.md).

## Run locally

Python 3.11+; Node 22.12+ (24 recommended). From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[server,test]"
$env:LLM_RUNTIME_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"
python -m uvicorn llm_runtime.server:app --host 127.0.0.1 --port 8765
```

In another terminal: `cd frontend`, `npm ci`, then `npm run dev`. Open [the local profiler](http://127.0.0.1:5173/#/runtime). First load downloads distilgpt2 into the normal Hugging Face cache. On Linux/macOS use `source .venv/bin/activate` and `export LLM_RUNTIME_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"`.

## Tests

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
cd frontend
npm test
npm run typecheck
npm run audit:design
npm run build
```

## Limits

- Compact cache collation copies K/V; no physical PagedAttention or custom CUDA kernels.
- No tensor parallelism, multi-GPU executor or memory-aware eviction/swapping.
- One small model/GPU/workload, exploratory tails and non-identical preprocessing; not production-capacity certification.
- Public Vercel demo is CPU, with cold starts and instance-local state; no authentication or coordinated multi-worker scheduler.
- INT8 changed output; the tested speculative draft accepted no proposals and did not accelerate generation.

## More detail

[Architecture](docs/architecture.md) · [Postmortem](docs/postmortem.md) · [UI/deployment guide](docs/portfolio-ui.md) · [GPU evidence](docs/evidence/phase3/INDEX.md) · [CPU methodology](docs/finalization.md)
