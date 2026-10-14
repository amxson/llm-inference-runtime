# Engineering postmortem

## What was built

M0–M17 developed an inspectable inference runtime: model/tokenizer loading, logits inspection, explicit greedy/sampled loops, request-local KV reuse, paired measurements, lifecycle/scheduling, static/continuous multi-row batching, HTTP/SSE, logical block ownership, native CPU INT8 and greedy draft/verify experiments. Separate milestone commits preserve the progression instead of replacing reference paths with an opaque final engine.

The service is a bounded, single-owner, **uncached continuous greedy batcher**. Cached CLI/scheduler execution, sampling, quantization and speculation are separate paths. This is a scope boundary, not a claim that all optimizations run together.

## What improved performance

The clearest controlled result is [M7's paired cache experiment](../benchmarks/results/m7_cached_vs_uncached_cpu.json). On the five-token France prompt with ten new tokens, one warmup/mode and three alternating measured pairs, all IDs matched. Forwarded positions fell **95 -> 14** (85.26% reduction); mean latency fell **699.031 -> 381.035 ms**, a **1.835x** ratio. Mean ITL fell 72.084 -> 36.816 ms. TTFT stayed approximately unchanged at 50.271 -> 49.693 ms: prompt prefill still happens.

Positions are not FLOPs; this is one CPU workload, not a universal speedup. Historical M5 had a different measured baseline and was not M7's denominator. Cache memory grows and attention still reads prior states.

[M13](../benchmarks/results/m13_serving_cpu.json) established actual multi-client network serving: all 48 requests succeeded with matching IDs. Maximum observed batches were 1/2/4, with 64/40/21 shared forwards in the measured levels. This demonstrates sharing forwards, not a clean causal estimate for the whole throughput difference.

## What did not improve, or traded away something

[M15 INT8](../benchmarks/results/m15_quantization_cpu.json) reduced estimated unique weight payload **327,650,304 -> 200,248,320 bytes** (38.88%). Mean latency was 317.228 versus FP32 362.930 ms, but greedy outputs **differed**: FP32 discussed the French Republic; INT8 continued about a major crisis. One repeated prompt is not perplexity/quality evaluation. Small logit perturbations can change the first argmax and subsequent context, amplifying output drift.

It quantized 24 attention/MLP projections, not the whole model. Embeddings, layer norms and tied output head stayed FP32. Payload excludes backend packing/qparams/activations/cache and is not RSS; both models stayed resident. Load and copy/adapter/probe/quantization setup were separate. Legacy PyTorch worked on installed `onednn` but emitted deprecation warnings. This is an experiment, not a durable universal quantization backend.

[M16 speculation](../benchmarks/results/m16_speculative_cpu.json) retained exact ordinary greedy IDs but mean latency **increased 375.235 -> 517.977 ms** (38.04%). Tiny-gpt2 proposed 27 tokens/request and accepted **zero**; ten target verifications remained plus 27 draft forwards. A compatible small draft need not be a useful predictor. Drafting/extra verification positions had costs without saved target calls. Synthetic tests demonstrate acceptance/bonus correctness; they are not real-model speed evidence.

M14 demonstrates capacity/ownership/reuse, not physical savings. Calling it paged attention or treating its utilization as GPU memory utilization would be misleading.

## Benchmark surprises

M13 concurrency one took 43.963 s versus 3.501/2.476 s for levels two/four, anomalously slow relative to later phases. Raw measurements remain unchanged; this sweep cannot establish why or attribute the whole gap to batching. Transient machine conditions are a possible confound, not an established diagnosis. Repeated randomized-order controlled sweeps would be needed before explaining causality.

Concurrency four increased throughput relative to two but also mean latency (588.227 versus 427.234 ms). Throughput and per-client latency differ. Sixteen samples/level yield exploratory percentiles, not trustworthy tail SLAs.

Draft/target had matching vocabulary/core BPE but different postprocessor/added-token metadata. Compatibility was checked, not inferred from the GPT-2 name. Shared target tokenization plus mapping/rules/probes avoids relying on differences. The old draft checkpoint included obsolete `attn.masked_bias` keys reported as unexpected; current model forwards and real parity checks succeeded.

## Design compromises

- Explicit loops preserve reference behavior instead of relying on generation helpers.
- Separate cached execution from uncached batching instead of pretending independent variable-length HF caches automatically form one ragged batch.
- Masks/positions make padded GPT-2 rows match independent decoding without tokenizer mutation.
- One serving owner processes disconnect commands between forwards, favoring correctness/responsive HTTP over parallel engine complexity.
- A logical pool models allocation without inventing a kernel/address translation. Exhaustion is failure, not an unimplemented admission policy.
- Greedy uncached speculation isolates acceptance correctness from rollback/stochastic acceptance.
- Raw artifacts retain failures, changed outputs, negative results and definitions. The summary copies source values/content hashes, not a cross-milestone leaderboard.

Download-free tests and real model/network checks complement each other. Version bounds are simple, not a lockfile. Recorded Python/PyTorch/Transformers/thread counts matter. Deprecation warnings remain visible; passing now does not guarantee all future dependencies behave identically. No hosted CI result is claimed.

## Why this is not vLLM or a production replacement

No physical KV paging, custom/fused attention kernels, distributed/tensor-parallel executor, CUDA graphs, memory-aware admission/eviction/swapping or optimized cached continuous batching exists here. HTTP has no authentication, OpenAI API compatibility, durable queue, delivery acknowledgments, multi-process coordination or validated backpressure policy. GPU performance was not measured; serving samples are small/CPU-specific.

Production engines additionally need integrated cache layouts/kernels, high-utilization prompt/decode scheduling, overload controls, cancellation/backpressure guarantees, observability, security, fault handling, broad model compatibility, and representative hardware/performance/quality validation. Those require engineering and measurement, not renaming bookkeeping classes.

## What to build next with more time

First design physically cached variable-length continuous batching with ownership/parity tests. Then memory-aware admission, repeated controlled load/multi-prompt sweeps and actual memory measurements. Evaluate quantization quality/migrate deprecated APIs; choose a better draft and implement cache rollback before expecting speculative acceleration. Add serving authentication, delivery/backpressure policy, observability and shutdown/disconnect stress tests.

These are proposals only. M17 finalizes documentation/results; no stretch mechanisms are started.
