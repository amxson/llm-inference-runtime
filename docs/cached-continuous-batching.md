# Cached continuous batching

## Design before implementation

Inspected locally: Transformers 5.17.0 `cache_utils.py`, GPT-2
`modeling_gpt2.py`, and the existing request/scheduler/batching/server code.
`DynamicCache` iterates layer `(keys, values, optional_sliding_window)` tuples
and accepts layer tensors through its documented `ddp_cache_data` constructor.
Tensors have shape `[batch, heads, history, head_dim]`. `update` appends along
history. `batch_select_indices` supports row selection, but cannot concatenate
ragged histories or remove *leading* padding; `crop` removes trailing history.
There is no need for legacy-cache conversion or private layer mutation.

Each active request owns a compact, single-row, full-attention DynamicCache.
The adapter validates that protocol and rejects sliding/static/empty caches.
It left-pads shorter histories with zero K/V and concatenates rows into a fresh
cache. The decode attention mask is `[synthetic zeros, real history ones, 1]`.
Explicit position IDs are each row's logical history length, not the padded
physical length. GPT-2 in this version defaults positions to the shared physical
cache length, so that default is wrong for ragged rows. It appends K/V via
`cache.update`; it does not require a separate `cache_position` argument.

Prefill also left-pads prompts, using a mask and cumulative logical positions.
After either forward, split each returned layer by row and keep only the final
logical-history positions. Clone slices so no request retains the batch storage
or aliases a neighbor. Padding exists only during a shared forward, never in
stored request history. This copies K/V every tick; it is not paging, a custom
attention kernel, or a memory-efficiency claim.

FIFO admission fills vacancies at tick boundaries. A tick partitions newcomers
and continuing rows *before* forwarding: at most one prefill and one decode
forward, at most one selected token per request. Newcomers join continuing rows
in subsequent decode ticks. Completion/cancellation/error releases local cache.
The uncached ContinuousBatcher remains a separate reference implementation.

Invariants to test:

- Stored cache length equals tokens actually forwarded, not tokens selected:
  prompt length T selects output 1 with cache T; the next forward makes T+1.
- Decode inputs are `[B, 1]`; logical positions and masks are row-specific.
- Collation/splitting preserve row order, contents, independent ownership, and
  lengths, including after admission/removal/reordering.
- Varied prompts/budgets and changing cohorts exactly match independent cached
  greedy token IDs; stops and errors do not leave live cache ownership behind.

This adapter deliberately targets GPT-2 full-attention DynamicCache as inspected
above, not every Transformers cache/model architecture. A future API change must
pass the adapter tests before being supported.

## Verified behavior and reproduction

The normal server now defaults to `continuous_cached_greedy`; select
`continuous_uncached_greedy` with `create_app(engine=...)` or
`LLM_RUNTIME_ENGINE` for the reference. Core dependencies constrain Transformers
to the tested 5.17 family; the public function already pins 5.17.0. No dependency
was added for cache collation. The adapter uses the public iterator/constructor,
not private cache layer writes. Tests also exercise documented
`batch_select_indices` for row reordering.

Traces retain existing `events`/`completed` fields for SSE compatibility and add
`serving_cache`, `prefill_request_ids`, `decode_request_ids`, their input shapes,
`logical_cache_lengths_before/after`, and `model_forward_count_this_tick`.
Lengths after forwarding describe work just done even if completion immediately
releases that row's cache. Published request summaries separately show current
retained lengths. Last-batch health telemetry reports the last actual forward,
not the sum or maximum of two groups in a tick.

Download-free tests instantiate random tiny GPT-2 locally. Explicit real-weight
checks are separate:

```powershell
python scripts/check_cached_continuous.py
python scripts/check_cached_http.py
python scripts/record_portfolio_trace.py
python -m llm_runtime.benchmark_continuous_cache --confirm-ac-quiet --output benchmark_results/continuous-new.json
```

The HTTP check starts a temporary loopback TCP server and verifies concurrent
SSE IDs, indices, decoded prefixes, dynamic admission, shared `[2,1]` decode,
zero tokens, and actual client disconnect cleanup. This is distinct from an
in-process HTTP test. Both complement direct engine parity, EOS, context-limit,
error, cancellation and ownership tests.

## CPU comparison

[Raw measurement](../benchmarks/results/cached_continuous_batching_cpu.json):
eight different prompts, budgets 16/24/32 (184 selected tokens/cohort), same
loaded distilgpt2 CPU model/tokenizer, four threads, capacities 1/2/4. Two excluded
warmup cohorts/mode/capacity, then ten alternating adjacent pairs. Timing starts
before request construction and includes tokenization, queue waiting, cache
copies and engine tracing through final completion. Model loading and final
text export are excluded. This is not HTTP timing.

| Capacity | Median uncached ms | Median cached ms | Median paired U/C | Median tok/s U / C | Padded input positions U / C | Forwards U / C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,346.77 | 6,316.09 | 3.67× | 7.88 / 29.18 | 3,540 / 235 | 184 / 184 |
| 2 | 15,111.16 | 4,544.82 | 3.68× | 12.81 / 40.49 | 4,428 / 236 | 104 / 109 |
| 4 | 6,974.06 | 1,970.08 | 3.62× | 26.38 / 93.40 | 4,956 / 257 | 56 / 59 |

All 30 measured pairs matched exact IDs/lifecycle outcomes, with eight successes
and 184 output tokens per cohort. Each cached cohort forwards 176 decode input
positions; the rest are padded prefill inputs. Cached was faster in every pair
of this run, despite *more* calls at capacities two/four. The paired-ratio
min–max ranges were 2.43–5.07×, 2.54–4.18× and 3.23–4.08×. These are observations,
not a general speed guarantee or an estimate for a modern GPU model.

Important host caveat: the operator confirmed AC and paused heavy work, but
Win32_Battery reported status 1 before and 2 afterward, with charge 52% → 46%.
[Microsoft's documented meanings](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-battery#properties)
are discharge versus AC access (not necessarily charging). Continuous AC was
therefore **not verified**. CPU-load observations were 42%/17%; power plan stayed
HP Optimized, unchanged. Large timing variation remains in the raw runs. There
were no concurrent tests/builds/browser review, but power transitions,
background activity and thermal drift were not isolated. Fixed ascending
capacity order also limits causal cross-capacity comparisons. Small-sample
request p90/p95 values are exploratory, not tail SLAs. Input positions include
padding, not attention reads or FLOPs. No raw historical measurement was edited.
