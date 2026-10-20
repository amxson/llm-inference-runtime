# Phase 2 verification

## Starting state

Local `main`, fetched `origin/main`, and GitHub main all started at
`fa8528c7bf0634b1b5a14b670b5d160ee374d121`. There were no tracked edits.
Phase 1 commits `96da844`, `116753c`, `8981ae2`, `44e07d2`, `4104516`,
`ad57089`, `06bb1e7`, and `fa8528c` were present. Source inspection verified
published-boundary health counters, exception logging, the concurrent health
test, the real length sweep, updated benchmark UI, shorter README and historical
caveats. The existing suite and recorded-artifact checks passed before advancing.

These original untracked items were not edited, staged or uploaded:
`abel_coursework_final/`, `alternative_rdbms_solution/`, `assignment_review/`,
`coursework_submission/`, `mysql_download_page.html`,
`rdbms_runtime_review_20261019/`, `rdbms_submission_review_20261019/`, and
`rdbms_trigger_revision_20261019/`.

## Cache/engine correctness

See the [design and results note](../../cached-continuous-batching.md).
Local Python is 3.11.9, torch 2.14.0+cpu and Transformers 5.17.0. Its installed
cache/GPT-2 source was inspected before the design note and implementation.
No private layer mutation is used: DynamicCache iteration and `ddp_cache_data`
construction wrap ordinary padded tensor collation. Row-selection tests exercise
`batch_select_indices`. Unsupported, empty or sliding cache formats fail clearly.
Normal dependencies now constrain Transformers to the inspected 5.17 family.

Download-free tests cover equal/ragged histories, order/reordering, padding,
logical positions, one-token extension, independent ownership, random tiny-GPT-2
logits and complete greedy parity. Engine tests include prompt lengths 3/8/5,
different budgets, A+B → B+C → C+D, zero tokens, EOS, active/waiting cancellation,
context boundary and errors. Finished rows release K/V/runtime inputs.

- [Real distilgpt2 engine check](engine-smoke.json): four changing requests,
  exact independent `cached_greedy_decode` parity, shared one-token forwards.
- [Real distilgpt2 loopback TCP HTTP check](http-smoke.json): four overlapping
  SSE clients, exact IDs/indices/decoded prefixes, admission during decoding,
  shared `[2,1]` decode, zero-token done and real disconnect cleanup.
- The final application revision `f432410` was also rechecked with both real
  scripts: [final engine check](final-engine-smoke.json) and
  [final TCP/SSE check](final-http-smoke.json), both passed.
- `tests/test_cached_server.py` additionally gates a random tiny-GPT-2 forward
  while four actual HTTP/SSE clients overlap, polls health/snapshot during it,
  checks cache identity/traces, and verifies cancellation and worker recovery.
- A mixed-tick telemetry regression checks prefill `[3,2]` plus decode `[1,1]`
  reports last actual batch 1, maximum actual batch 3, and three total forwards
  after the preceding single-row prefill. Row counts are not summed into a
  fictitious shared batch.

### Critical real trace

Extracted from the real engine check, not a hand-created replay:

| Tick | Admitted | Prefill | Decode | Decode input | Cache before → after forward | Completed |
| ---: | --- | --- | --- | --- | --- | --- |
| 2 | — | — | A, B | [2,1] | A 5→6; B 4→5 | A |
| 3 | C | C | B | [1,1] | C 0→7; B 5→6 | — |
| 4 | — | — | B, C | [2,1] | B 6→7; C 7→8 | — |
| 5 | — | — | B, C | [2,1] | B 7→8; C 8→9 | B |
| 6 | D | D | C | [1,1] | D 0→4; C 9→10 | — |
| 7 | — | — | C, D | [2,1] | C 10→11; D 4→5 | C |

New C is prefetched while B retains its history, then they share a real cached
decode call on tick 4. The same happens for C+D on tick 7. Tick 3 makes two
forwards but selects only one token per request. Reported after-forward lengths
precede immediate completion cleanup; a finished row does not retain that cache.

## Measurement

[New raw artifact](../../../benchmarks/results/cached_continuous_batching_cpu.json)
and [run log](benchmark-run.txt) are separate from every Phase 1/historical raw
measurement. Eight prompts, budgets 16/24/32, 184 tokens/cohort; capacities 1/2/4;
two warmup cohorts/mode/level and ten alternating measured pairs/level. One
loaded distilgpt2 CPU model/tokenizer, four threads. All 30 pairs match exact IDs,
all eight requests succeed in every cohort. Raw runs retain request latency,
forward groups/sizes, physical input-position counts, logical cache lengths,
admission/completion summaries and paired statistics.

| max_active | Median uncached ms | Median cached ms | Median paired U/C | Median tok/s U / C | Input positions U / C | Forwards U / C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,346.77 | 6,316.09 | 3.67× | 7.88 / 29.18 | 3,540 / 235 | 184 / 184 |
| 2 | 15,111.16 | 4,544.82 | 3.68× | 12.81 / 40.49 | 4,428 / 236 | 104 / 109 |
| 4 | 6,974.06 | 1,970.08 | 3.62× | 26.38 / 93.40 | 4,956 / 257 | 56 / 59 |

Cached was faster in this run, not assumed in the harness. Ratios are medians of
paired wall-time ratios, not ratios of aggregate medians. Counts include padding;
they are not FLOPs. Each cached cohort processes 176 newest-token decode inputs.
Extra prefill calls can increase total forwards even when input work decreases.

The user freshly confirmed AC/quiet before the run. No tests, builds or browser
review ran during measurement. However, before/after battery codes were 1/2,
charge 52%/46%, and CPU-load observations 42%/17%. Continuous AC was not verified;
the power plan remained HP Optimized without changes. Preserve this power/
background/thermal caveat and the substantial raw timing variation. Fixed level
order and ten pairs limit causal interpretation; p90/p95 are exploratory only.
Runtime baseline is `59a9c88`; the new harness is identified by file hashes and
committed with the artifact in `892827a`. This does not claim the harness already
existed in the baseline commit.

## Local browser/UI

The replay was regenerated by `scripts/record_portfolio_trace.py` using actual
cached distilgpt2 execution, not manually edited JSON. It has six requests,
changing admission, compact KV reuse and nine discrete ticks, not fabricated
elapsed times. The recording's engine baseline is `59a9c88`.

All four routes were captured against the API-configured production build at
1440/1024/768 CSS pixels. [Browser results](browser/results.json) record zero
console errors and failed requests, no document-wide overflow, and no API calls
on static routes. Dense tables/pipelines use local horizontal scrolling.
Screenshots were visually inspected; the palette, fonts and existing profiler
layout were retained. No new UI dependency or decorative design system was added.

- [Runtime 1440](browser/runtime-1440.png), [1024](browser/runtime-1024.png),
  [768](browser/runtime-768.png)
- [Playground 1440](browser/playground-1440.png), [1024](browser/playground-1024.png),
  [768](browser/playground-768.png)
- [Benchmarks 1440](browser/benchmarks-1440.png), [1024](browser/benchmarks-1024.png),
  [768](browser/benchmarks-768.png)
- [Architecture 1440](browser/architecture-1440.png), [1024](browser/architecture-1024.png),
  [768](browser/architecture-768.png)

[Local live-browser transcript](live-browser/results.json) records eight real
frontend SSE token events, one done, visible output while still running, cache
status on, actual request selection in Runtime Live, zero console errors/failed
requests, and replay selection/step/play/pause with the API blocked and zero API
requests. [Streaming capture](live-browser/playground-streaming.png),
[completion](live-browser/playground-done.png), [live trace](live-browser/runtime-live.png),
[backend-independent replay](live-browser/replay-api-blocked.png).

Windows Computer Use returned `native pipe is unavailable` on the tooling check.
The available bundled Playwright/preinstalled headless Edge path was used; no
browser permission or security setting was changed. One local capture attempt
used the deliberately static-only build and was stopped before retrying with
the explicit local API base. Only completed checks are reported as passed.

## Automated checks

| Check | Result | Raw output |
| --- | --- | --- |
| Python | 175 passed; six existing dependency warnings | [pytest](checks/python-tests.txt) |
| Compile/import | exit 0 | [compileall](checks/python-compile.txt) |
| Frontend | 38 passed / 8 files | [npm test](checks/npm-test.txt) |
| TypeScript | exit 0 | [typecheck](checks/npm-typecheck.txt) |
| Design audit | exit 0 | [audit](checks/npm-design-audit.txt) |
| Production build | exit 0 | [build](checks/npm-build.txt) |

[Check command/timestamp/exit metadata](checks/results.json) and
[exact npm-command metadata](checks/npm-results.json) distinguish static-only
build verification from the API-configured local build. Logs normalize only
user/repository paths and trailing whitespace; warnings/results are retained.
No hosted CI result is claimed.

## Public deployment — functionally verified

[Deployment metadata](public/deployment.json): `dpl_6JcM3G7XHySsJgYPM8dQmA7fowCE`,
application source `f432410202df01af66765972fe3ae57e8d902833`, READY and aliased
to <https://llm-inference-runtime.vercel.app>. The function bundle is 332.59 MB
compressed, using the already-enabled large-functions support. Hobby/active
was rechecked; no plan, host, domain, paid resource or environment setting was
changed. One CPU thread and capacity one remain the public limits. Local engine
tests and benchmark capacities 2/4 prove multi-row execution; the public demo is
not presented as a multi-row throughput benchmark.

Public backend: <https://llm-inference-runtime.vercel.app/api>.
[Actual health/CORS/SSE check](public/health-sse.json) requires JSON health and
reports:

```json
{"status":"ok","model":"distilgpt2","device":"cpu","engine":"continuous_cached_greedy","serving_cache":true,"max_active":1}
```

That block selects identity fields from the full recorded response; counters
are instance-local and time-dependent. The allowed-origin preflight passed;
an untrusted origin received no allow-origin header. Eight genuine token events
had contiguous indices and IDs matching one done event, with no error event.
Continuation: ` the capital of the French Republic. The`.

[Actual public browser transcript](public/browser/results.json) separately
verifies the frontend path, not just an HTTP client. It observes a cloned real
fetch-SSE response, returns the original unchanged to React, captures a rendered
token while UI status is still `running`, then records completion. No model,
network response, token or healthy status was mocked. Token IDs were
`[262,3139,286,262,4141,2066,13,383]`; one done; zero console errors or failed
requests. Playground showed serving cache on and the top status returned to
idle. Runtime Live selected that exact generated request and displayed prompt
prefill followed by cached one-token decode; final forwarded history 11→12 was
released on completion. [Streaming](public/browser/playground-streaming.png),
[done](public/browser/playground-done.png), [live trace](public/browser/runtime-live.png).

Replay selection/step/play/pause also passed in a fresh browser context with API
routes blocked, issuing zero API requests. [Replay capture](public/browser/replay-api-blocked.png).
[Public responsive results](public/responsive/results.json) cover all four routes
at 1440/1024/768, zero document overflow, zero console errors/failed requests,
and no API polling from static routes. Fresh screenshots show the new cached
trace and comparison below the unchanged Phase 1 sweep.

[Configuration inspection](public/config.json) confirms both production keys
exist. The platform API did not return their values, even through its individual
environment endpoint; no missing value was treated as a mismatched actual URL.
Effective production configuration was verified through real browser requests
to `https://llm-inference-runtime.vercel.app/api/runtime/snapshot` and
`/api/generate/stream`, plus the explicit origin checks above. These support the
expected API base and CORS origin without printing credentials or changing
settings. Historical raw evidence remains unchanged; application success is
not inferred from route HTTP 200 or artifact hashes alone.

## Git and scope

Normal commits pushed separately:

1. `bc6b019` — feat: add batched KV cache collation
2. `3169e2b` — feat: add cached continuous batching
3. `59a9c88` — feat: serve cached continuous batches
4. `892827a` — bench: compare cached continuous batching
5. `334fdee` — feat: visualize cached continuous serving
6. `f432410` — fix: expose cached batching benchmark artifact

The sixth commit corrects an actual final-review omission and adds a regression
assertion; it is not manufactured debugging history. This documentation/evidence
record follows as its own substantive commit. Every push was normal. No amend,
rebase, squash, force-push or old-date/history change was performed. Old raw
benchmark files and reference engines remain unchanged. No model weights,
caches, credentials, unrelated coursework or generated deployment auth are
tracked/uploaded. Deployment uses `git archive HEAD`, not the dirty workspace.

No Phase 3 work was started: no GPU/modern-model migration, external-engine
benchmark, CUDA/Triton kernel, physical paging or distributed execution.
