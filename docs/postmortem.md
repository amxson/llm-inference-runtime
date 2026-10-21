# Engineering postmortem

## What was built

M0–M17 developed an inspectable inference runtime: model/tokenizer loading, logits inspection, explicit greedy/sampled loops, request-local KV reuse, paired measurements, lifecycle/scheduling, static/continuous multi-row batching, HTTP/SSE, logical block ownership, native CPU INT8 and greedy draft/verify experiments. Separate milestone commits preserve the progression instead of replacing reference paths with an opaque final engine.

The service is a bounded, single-owner, **cached continuous greedy batcher** by default after Phase 2, with the original uncached engine retained for comparison. Sampling, quantization and speculation remain separate paths. This is not a claim that all optimizations run together.

## What improved performance

The historical [M7 paired cache experiment](../benchmarks/results/m7_cached_vs_uncached_cpu.json) established exact token parity and less repeated work. On the five-token France prompt with ten new tokens, one warmup/mode and three alternating measured pairs, all IDs matched. Forwarded positions fell **95 -> 14** (85.26% reduction); mean latency fell **699.031 -> 381.035 ms**, a **1.835x** ratio. Mean ITL fell 72.084 -> 36.816 ms. TTFT stayed approximately unchanged at 50.271 -> 49.693 ms: prompt prefill still happens. These historical timings may be host-contention affected; this is a hypothesis, not proven causality. They are not the preferred final headline measurement. See [finalization measurements](finalization.md) for the quiet-window rerun and explicit measurement scopes.

Positions are not FLOPs; this is one CPU workload, not a universal speedup. Historical M5 had a different measured baseline and was not M7's denominator. Cache memory grows and attention still reads prior states.

The 95 → 14 count is not a prediction of a 6.79× latency ratio. M7's five-token prompt and ten-token continuation are tiny, and distilgpt2 is small. Both paths still execute ten Python/model calls and do prompt prefill. Cached attention reads the earlier K/V on every decode step even though projection inputs shrink to one position. Input positions do not count the different operations inside a forward, including the growing attention read. Fixed call costs and those retained operations need not scale with the position count. Their individual contributions were not measured; CPU timing also says nothing about GPU kernel behavior. The [32/128/512 length sweep](finalization.md#cache-length-sweep) tests the longer-workload question directly rather than assigning an unmeasured cause to M7's timing.

[M13](../benchmarks/results/m13_serving_cpu.json) established actual multi-client network serving: all 48 requests succeeded with matching IDs. Maximum observed batches were 1/2/4, with 64/40/21 shared forwards in the measured levels. This demonstrates sharing forwards, not a clean causal estimate for the whole throughput difference.

## What did not improve, or traded away something

[M15 INT8](../benchmarks/results/m15_quantization_cpu.json) reduced estimated unique weight payload **327,650,304 -> 200,248,320 bytes** (38.88%). Mean latency was 317.228 versus FP32 362.930 ms, but greedy outputs **differed**: FP32 discussed the French Republic; INT8 continued about a major crisis. One repeated prompt is not perplexity/quality evaluation. Small logit perturbations can change the first argmax and subsequent context, amplifying output drift.

It quantized 24 attention/MLP projections, not the whole model. Embeddings, layer norms and tied output head stayed FP32. Payload excludes backend packing/qparams/activations/cache and is not RSS; both models stayed resident. Load and copy/adapter/probe/quantization setup were separate. Legacy PyTorch worked on installed `onednn` but emitted deprecation warnings. This is an experiment, not a durable universal quantization backend.

[M16 speculation](../benchmarks/results/m16_speculative_cpu.json) retained exact ordinary greedy IDs but mean latency **increased 375.235 -> 517.977 ms** (38.04%). Tiny-gpt2 proposed 27 tokens/request and accepted **zero**; ten target verifications remained plus 27 draft forwards. A compatible small draft need not be a useful predictor. Drafting/extra verification positions had costs without saved target calls. Synthetic tests demonstrate acceptance/bonus correctness; they are not real-model speed evidence.

The recorded draft/verify experiment checks the implementation and parity behavior; it did not demonstrate acceleration. Zero acceptance is observed on this workload. The artifact does not isolate why that draft produced those proposals, so no claim about model alignment is inferred from it. The original setup used a shared target tokenizer after checking the complete token mapping and core tokenization rules; compatibility alone cannot predict acceptance.

M14 demonstrates capacity/ownership/reuse, not physical savings. Calling it paged attention or treating its utilization as GPU memory utilization would be misleading.

## Benchmark surprises

M13 concurrency one took 43.963 s versus 3.501/2.476 s for levels two/four, anomalously slow relative to later phases. Raw measurements remain unchanged; this sweep cannot establish why or attribute the whole gap to batching. Transient machine conditions are a possible confound, not an established diagnosis. Repeated randomized-order controlled sweeps would be needed before explaining causality.

Concurrency four increased throughput relative to two but also mean latency (588.227 versus 427.234 ms). Throughput and per-client latency differ. Sixteen samples/level yield exploratory percentiles, not trustworthy tail SLAs.

Draft/target had matching vocabulary/core BPE but different postprocessor/added-token metadata. Compatibility was checked, not inferred from the GPT-2 name. Shared target tokenization plus mapping/rules/probes avoids relying on differences. The old draft checkpoint included obsolete `attn.masked_bias` keys reported as unexpected; current model forwards and real parity checks succeeded.

## Design compromises

- Explicit loops preserve reference behavior instead of relying on generation helpers.
- Preserve the uncached reference; explicitly collate/split variable-length KV histories in an isolated adapter for cached continuous execution.
- Masks/positions make padded GPT-2 rows match independent decoding without tokenizer mutation.
- One serving owner processes disconnect commands between forwards, favoring correctness/responsive HTTP over parallel engine complexity.
- A logical pool models allocation without inventing a kernel/address translation. Exhaustion is failure, not an unimplemented admission policy.
- Greedy uncached speculation isolates acceptance correctness from rollback/stochastic acceptance.
- Raw artifacts retain failures, changed outputs, negative results and definitions. The summary copies source values/content hashes, not a cross-milestone leaderboard.

Download-free tests and real model/network checks complement each other. Version bounds are simple, not a lockfile. Recorded Python/PyTorch/Transformers/thread counts matter. Deprecation warnings remain visible; passing now does not guarantee all future dependencies behave identically. No hosted CI result is claimed.

## Phase 2: connecting caches to continuous admission

Variable-length cache histories were the missing tensor problem: a shared GPT-2
forward has one physical cache width, but each row needs its own logical next
position. The adapter left-pads K/V, masks synthetic history, passes per-row
position IDs, and clones trimmed rows after the forward. Compact storage is
simple to inspect but copies history every tick; it is not an optimized layout.

The off-by-one test mattered: prefill T selects an output but leaves cache T,
not T+1. Other checks cover distinct prompt lengths, row order/aliasing, padding,
EOS/removal, cancellation and A+B → B+C → C+D sharing actual one-token decode
forwards. Separately captured real TCP SSE checks verify concurrent clients,
decoded prefixes and disconnect cleanup; direct-engine tests alone would not
prove serving behavior. A telemetry regression ensures a mixed tick reports
the final actual batch size, not the combined row count.

Cached execution was faster in this eight-prompt CPU run: median paired cohort
ratios 3.67×/3.68×/3.62× at capacities 1/2/4. This includes collation overhead and
more model calls at capacities two/four. However, host power telemetry changed
despite AC confirmation, charge fell 52% → 46%, and timings varied. This is a
qualified measurement, not a rigorously power-controlled causal result. The
[design/results note](cached-continuous-batching.md#cpu-comparison) and raw runs
retain that caveat. No memory savings, GPU behavior or broad model compatibility
were measured. There is still no paging, memory-aware eviction or distributed
scheduler. The final GPU result below is separate evidence.

## Why this is not vLLM or a production replacement

No physical KV paging, custom/fused attention kernels, distributed/tensor-parallel executor, CUDA graphs or memory-aware admission/eviction/swapping exists here. Cached continuous batching uses explicit padded copies, not a production cache layout. HTTP has no authentication, OpenAI API compatibility, durable queue, delivery acknowledgments, multi-process coordination or validated backpressure policy. GPU measurements cover one small model/workload, not production capacity.

Production engines additionally need integrated cache layouts/kernels, high-utilization prompt/decode scheduling, overload controls, cancellation/backpressure guarantees, observability, security, fault handling, broad model compatibility, and representative hardware/performance/quality validation. Those require engineering and measurement, not renaming bookkeeping classes.

## Historical scope limits

The earlier CPU-only review left GPU memory measurement open; Phase 3 now records allocator statistics and sampled process memory below. Memory-aware admission/eviction, quantization quality/API migration, a better speculative draft/cache rollback, authentication, delivery/backpressure policy and multi-process coordination remain outside the completed scope.

These are limitations, not a new roadmap. No stretch mechanisms are started.

## GPU / production-engine comparison

Qwen2.5-0.5B ran on the existing cached engine without redesigning its scheduler.
The model-input boundary removes GPT-2 segment IDs, supplies logical per-row
rotary positions, respects generation EOS lists and requests final-position
logits for token loops. Transformers 5.17 Qwen2 no longer takes `cache_position`.
Full-attention DynamicCache is supported; sliding-window caches are rejected.
Masked physical padding is not logical history.

The hard failure was exact FP16 greedy parity across batch shapes. The fixed
workload passed capacity one but not two. Diagnostics verified input/mask/position
invariants and bitwise-exact cache collation/splitting at the mismatch frames;
different execution shapes changed near-tied token rankings. Math-only attention,
full reductions and a FP32 vocabulary head did not establish FP16 correctness.
That supports numerical sensitivity, not proof of one faulty kernel or universal
stability. Failed outputs remain in the evidence, not excluded from the workload.

Amar explicitly approved FP32 for both engines. A fresh native-FP32 load of the
original checkpoint—not promoted rounded FP16 weights—passed all 24 prompts at
capacities 1/2/4, eight manual checks, dynamic admission, zero/EOS and HTTP/SSE.
The final run completed 60 measured cohorts and 1,440 requests without failure.
All internal reference IDs and 720 paired cross-engine outputs matched. This
establishes this workload, not a general equivalence theorem.

Unmodified vLLM 0.17.1 was faster: median generated-token throughput was
38.03/70.94/126.97 tok/s versus our 29.36/49.06/76.78 at concurrency 1/2/4,
ratios 1.30×/1.45×/1.65×. Our median request latency increased with concurrency;
greater throughput did not mean lower client latency. At capacity one our median
TTFT was lower (37.19 vs 54.74 ms), despite lower overall throughput.

Known differences: our Python scheduler observes forwards and copies compact K/V
into padded batches every tick; vLLM has an integrated executor and kernel/cache
layout, including its supported Triton attention path. These plausibly contribute
to the gap, but timing alone does not isolate their costs. CUDA graphs/compilation
and prefix caching were disabled in vLLM here, so cannot explain its measured
advantage. FP32 on a T4 is not a best-performance production profile. Both engines
use the same Torch/CUDA wheels but different Transformers versions. vLLM receives
pretokenized IDs; ours tokenizes HTTP text. Request tails pool correlated samples;
external VRAM sampling is not an exact comparable peak. Only our allocator peaks
are reported as allocated/reserved, not physical VRAM savings.

Closing the remaining gap would require increasingly specialized layouts,
memory/kernel work, profiling and possibly distributed execution. This project
stops here: it demonstrates core runtime mechanisms and measures them against a
production engine. There is no Phase 4. Public CPU serving stays separate from
the recorded T4 experiment. [Raw data and methods](evidence/phase3/INDEX.md).
