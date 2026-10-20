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
