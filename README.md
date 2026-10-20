# LLM Inference Runtime

An LLM inference runtime with a token-generation loop written directly in Python and PyTorch. It uses Hugging Face models, but handles decoding, request state, scheduling and batching itself. I built it to learn how LLM serving works beyond calling `model.generate()`.

## Demo

[Open the profiler](https://llm-inference-runtime.vercel.app/#/runtime).

Playground runs real SSE inference. HTTP serving supports **cached continuous greedy batching**, the local default; health/snapshot identify the selected backend. Replay, Benchmarks and Architecture also work without the model server.

The public CPU demo can be slow on a cold start. Runtime counters belong to the responding server instance.

## What it implements

- Model/tokenizer loading and single-forward logits inspection.
- Manual greedy decoding and temperature/top-k/top-p sampling.
- Single-request KV-cache reuse with exact token-ID comparisons.
- Request lifecycle, FIFO admission and round-robin cached scheduling.
- Static uncached and continuous cached/uncached multi-row greedy batching.
- FastAPI endpoints and streamed token events over SSE.
- Logical block allocation, ownership and reuse.
- CPU INT8 and greedy draft/verify experiments, including negative results.
- A React profiler for live traces, recorded replay and benchmark results.

## KV cache result

The [cache-length sweep](benchmarks/results/cache_length_sweep_cpu.json) uses distilgpt2 on CPU with the prompt `The capital of France is`. Each length has two excluded warmups per mode and ten measured cached/uncached pairs, alternating execution order.

| Budget / generated | Median uncached (ms) | Median cached (ms) | Median paired U/C | Uncached positions | Cached positions | Position U/C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 / 32 | 2,237.90 | 706.37 | 3.10× | 656 | 36 | 18.22× |
| 128 / 128 | 19,198.77 | 2,843.37 | 6.44× | 8,768 | 132 | 66.42× |
| 512 / 512 | 169,055.99 | 10,144.94 | 18.09× | 133,376 | 516 | 258.48× |

The latency ratio is the median of the ten **matched-pair** ratios, not a ratio of aggregate means. Cached and uncached token IDs match exactly in every measured pair. Forwarded positions count input work, not FLOPs or elapsed time.

Both paths still make one model call per generated token and do prompt prefill. Cached attention still reads earlier K/V. Python/model-call costs matter for short sequences on this small CPU model; their individual contributions were not measured. See [methodology and spread](docs/finalization.md#cache-length-sweep).

The [old ten-token comparison](benchmarks/results/m7_cached_vs_uncached_cpu.json) remains available as historical evidence.

The separate [cached continuous batching comparison](benchmarks/results/cached_continuous_batching_cpu.json) uses eight varied prompts, budgets 16/24/32 and capacities 1/2/4. It compares whole cohorts, not the single-request timing above. See the [cache design and invariants](docs/cached-continuous-batching.md).

## Architecture

```text
HTTP -> FIFO admission -> prompt prefill -> request-local KV
Later ticks: retained KV -> padded collation -> [B,1] decode -> split/update -> SSE
Completion -> release KV; the next tick admits waiting requests into vacancies
```

The uncached HTTP engine and independent cached decoder remain references. [Architecture and API details](docs/architecture.md).

## Run it

Python 3.11+; Node 22.12+ (24 recommended). From the repository root, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[server,test]"
$env:LLM_RUNTIME_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"
python -m uvicorn llm_runtime.server:app --host 127.0.0.1 --port 8765
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open [the local profiler](http://127.0.0.1:5173/#/runtime). The backend loads distilgpt2 once at startup; weights use the normal Hugging Face cache. On Linux/macOS, activate with `source .venv/bin/activate` and use `export LLM_RUNTIME_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"`.

To repeat the length sweep, plug into AC power, pause heavy work, and choose a new output path:

```powershell
python -m llm_runtime.benchmark_cache_sweep --confirm-ac-quiet --output benchmark_results/cache-sweep-new.json
```

See the [UI guide](docs/portfolio-ui.md) for API-base configuration and deployment.

## Tests

```powershell
# Repository root: download-free tests
python -m pytest -q
cd frontend
npm test
npm run typecheck
npm run audit:design
npm run build
```

## Limits

- Cache collation copies padded K/V every tick; GPT-2 full-attention DynamicCache is the tested format. Sampling is a separate CLI path.
- The logical allocator tracks capacity and ownership; it is not PagedAttention.
- CPU timings from one small model and fixed workloads are not GPU performance or production capacity claims.
- INT8 changed the recorded output. Only selected attention/MLP projections were quantized; embeddings, layer norms and the tied output head stayed FP32. Weight-payload reduction is not total RSS.
- The tested speculative draft produced zero accepted proposals on the recorded workload. The experiment checks draft/verify and parity behavior; it did not demonstrate acceleration.
- Public serving has cold starts, usage limits and instance-local telemetry. Authentication, shared multi-worker scheduling and production backpressure remain future work.

## More detail

- [Architecture and CLI/API reference](docs/architecture.md)
- [Postmortem and experiment tradeoffs](docs/postmortem.md)
- [Profiler UI guide](docs/portfolio-ui.md)
- [Benchmark methods and limitations](docs/finalization.md)
- [Raw benchmark artifacts](benchmarks/results)
