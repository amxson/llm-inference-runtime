# LLM Inference Runtime

## Short portfolio description

[Public static demo](https://llm-inference-runtime.vercel.app/#/runtime).

An educational AI systems project that makes token generation inspectable below `model.generate()`: explicit decoding, request lifecycle, FIFO continuous batching, real SSE delivery, a logical ownership model and a restrained token-level profiler. Replay and source-derived benchmarks work independently of the backend; public Live uses a CPU Python function, subject to cold starts and compute quotas. Cached single-request decoding is a separate path from uncached HTTP serving.

## CV bullets

- Built an inspectable PyTorch/Transformers inference runtime with manual decoding, request-local KV reuse, FIFO continuous batching, cancellation and genuine token-level HTTP/SSE streaming.
- Developed a React/TypeScript profiler with linked request/allocator views, deterministic engine replay and provenance-linked timing analysis; validated five real SSE requests with 128 contiguous token events, one completion and zero errors each.
- Benchmarked cached/uncached single-request CPU decoding with alternating pairs and excluded warmups; observed 11,080.8 ms versus 2,032.9 ms mean latency for one 128-token workload with exact token-ID parity. Documented residual host load, worker-time tails, changed INT8 output and slower speculative decoding rather than generalizing speedup claims.

## Concise interview explanation

I built the token loop and request execution machinery instead of delegating generation to another engine. Scheduling chooses which requests run; batching determines which rows share a forward. The HTTP path recomputes contexts and streams actual token events. A separate cached path retains Hugging Face K/V and forwards only the newest token after prefill; logical block ownership is bookkeeping, not physical PagedAttention. The UI helps distinguish those boundaries and shows raw benchmark provenance. My final measurements separate local selection timing from HTTP arrival timing, preserve negative results, and label host-contention explanations as hypotheses. This is inference/observability engineering, not a production engine or an ML-research claim.

Numbers and limitations: [finalization report](docs/finalization.md), [raw artifact](benchmarks/results/final_cpu.json).
