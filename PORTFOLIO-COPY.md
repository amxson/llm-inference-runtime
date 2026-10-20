# LLM Inference Runtime

## Short portfolio description

[Public static demo](https://llm-inference-runtime.vercel.app/#/runtime).

An educational AI systems project that makes token generation inspectable below `model.generate()`: explicit decoding, request lifecycle, cache-aware FIFO continuous batching, real SSE delivery and a restrained token-level profiler. Replay and source-derived benchmarks work independently of the backend; public Live uses a CPU Python function, subject to cold starts and compute quotas. The original uncached engine remains a reference.

## CV bullets

- Built an inspectable PyTorch/Transformers inference runtime with manual decoding, request-local KV reuse, FIFO continuous batching, cancellation and genuine token-level HTTP/SSE streaming.
- Developed a React/TypeScript profiler with linked request/cache views, engine-generated replay and provenance-linked timing analysis. Historical 128-token HTTP checks and new concurrent cached SSE checks are recorded separately.
- Benchmarked cached/uncached single-request CPU decoding with alternating pairs and excluded warmups; observed 11,080.8 ms versus 2,032.9 ms mean latency for one 128-token workload with exact token-ID parity. Documented residual host load, worker-time tails, changed INT8 output and slower speculative decoding rather than generalizing speedup claims.

## Concise interview explanation

I built the token loop and request execution machinery instead of delegating generation to another engine. Scheduling chooses which requests run; batching determines which rows share a forward. Cached HTTP serving collates different K/V histories, forwards one newest token per continuing row, then splits compact request-local caches. New prompts prefill separately and join incumbents on later ticks. The original full-context engine is retained for comparison. Logical block ownership is bookkeeping, not physical PagedAttention. Measurements separate local cohort time from HTTP arrivals, retain negative experiments and expose power/background limitations. This is inference/observability engineering, not a production engine or an ML-research claim.

Numbers and limitations: [finalization report](docs/finalization.md), [raw artifact](benchmarks/results/final_cpu.json).
