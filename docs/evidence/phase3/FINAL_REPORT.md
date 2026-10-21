# Final engineering report — Phase 3

## Starting state and scope

Phase 3's verified base is `345fe9443d98caafec78fced6a125c6bafb6ff84`.
All seven Phase 2 commits remain ancestors of main, in order: `bc6b019`,
`3169e2b`, `59a9c88`, `892827a`, `334fdee`, `f432410`, `345fe94`.
The cached backend remains default, the uncached reference remains available,
and Phase 2 tests and static artifacts remain present. No history was rewritten.

Re-reading the actual CPU artifact confirms exact parity across 30 measured
pairs. Median uncached/cached cohort seconds at capacities 1/2/4 are
23.3468/6.3161, 15.1112/4.5448, 6.9741/1.9701; median paired ratios are
3.6685/3.6779/3.6154. Those are CPU results, not GPU baselines.

This final resumption started at `da21983` with intentionally saved UI/docs
changes. Eight unrelated untracked coursework/review paths were left untouched.
The saved benchmark commit's normal push was retried successfully. The suspended
one-time continuation was paused after the user explicitly resumed this chat.

## Hardware, model and isolated environments

- Authorized Google Colab Linux T4; no paid resources or premium GPU requested.
- Tesla T4, compute capability 7.5, NVIDIA driver 580.82.07.
- nvidia-smi reports 15,360 MiB; Torch-visible total is 15,637,086,208 bytes
  (14.56 GiB). These are distinct reported quantities, not allocator peaks.
- Driver CUDA support 13.0; installed Torch CUDA runtime **12.9**. The toolkit
  reports nvcc 13.0.88. No claim that the Torch wheel is CUDA 13.0.
- Both isolated environments: Python 3.13.15, Torch 2.10.0+cu129.
- Our runtime: Transformers 5.17.0. Baseline: unmodified vLLM 0.17.1,
  Transformers 4.57.6. Both pip checks pass; complete freezes are preserved.
- Model: `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`; Qwen2Tokenizer,
  pad/eos 151643. Recorded model context 32768; shared serving benchmark limit 2048.
- **Native FP32 in both engines**, explicitly approved after strict FP16
  parity failures. Original checkpoint loaded directly at FP32, not rounded
  FP16 weights promoted back to FP32. No primary quantization.

The existing Colab base Torch 2.11.0+cu130 environment was preserved; compatible
Torch/vLLM packages were installed deliberately in separate environments.

## Modern execution and correctness

The focused adapter leaves scheduling/cache ownership unchanged. Qwen omits
GPT-2 token-type inputs, uses logical per-row RoPE position IDs, respects
generation EOS lists and keeps only final-position logits inside token loops.
The inspected Transformers 5.17 Qwen forward no longer takes `cache_position`.
Only full-attention DynamicCache is supported; sliding-window layouts are rejected.
Existing GPT-2 behavior remains covered by the full CPU regression suite.

Real T4 native-FP32 gate: finite full-forward logits `[1,5,151936]`, eight manual
greedy checks, 24 independent cached references, and exact full-output parity
for all 24 requests at each capacity 1/2/4. Same/different prompt lengths,
zero-token execution, forced EOS, token budgets, HTTP and 24 SSE requests pass.
The SSE correctness cohort contains 1,760 token IDs with zero failures.

The actual changing-cohort trace contains 11 frames. Tick 1 prefills `[2,5]`;
tick 2 decodes `[2,1]` and one request finishes. Tick 3 admits a new request,
prefills `[1,7]`, and retains the old request's cached decode. Tick 4 decodes
old/new rows together `[2,1]`. A later departure/admission repeats at ticks 6/7;
the final row completes at tick 11. Raw JSON records logical cache lengths,
memberships and output IDs. These are real GPU frames, not UI illustrations.

Earlier failed FP16 policies and diagnostics remain unmodified under `precision/`.
Numerical execution-shape sensitivity is supported by the near-tie/cache/input
diagnostics, but no single faulty kernel was isolated. FP16 is not declared fixed.

## Production baseline and benchmark

Both engines use the same pinned model/tokenizer revision on the same T4, one
resident server at a time. Our math-only SDPA is compared with ordinary vLLM
TRITON_ATTN, `--enforce-eager`, no prefix caching, no quantization,
`max_model_len=2048`, `max_num_seqs=4`, GPU utilization setting 0.70.
CUDA graphs/compilation are disabled: this is **not best-tuned vLLM**.

Fixed 24 varied prompts, mixed 32/64/128-token budgets, capacities 1/2/4.
Four ABBA engine-residency sessions rotate capacity order. Each session/config
excludes two warmup cohorts: 24 excluded warmups total. Ten measured repetitions
per engine/capacity yield 60 cohorts and 1,440 successful requests, zero failures.
Model startup/warmup is excluded; both serving paths use loopback HTTP/SSE.

Request latency and TTFT pool 240 request observations per engine/capacity;
throughput is the median whole-cohort rate. TTFT begins before HTTP send after
client semaphore admission, ending at the first nonempty token-ID event.

| Capacity | Engine | Median latency ms | p95 ms | Median TTFT ms | Median tok/s | Median req/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Ours | 2171.24 | 4445.09 | 37.19 | 29.36 | 0.400 |
| 1 | vLLM | 1679.99 | 3411.81 | 54.74 | 38.03 | 0.519 |
| 2 | Ours | 2494.51 | 5093.21 | 100.72 | 49.06 | 0.669 |
| 2 | vLLM | 1723.22 | 3460.31 | 76.81 | 70.94 | 0.967 |
| 4 | Ours | 3021.65 | 6105.12 | 118.08 | 76.78 | 1.047 |
| 4 | vLLM | 1741.34 | 3493.94 | 78.18 | 126.97 | 1.731 |

| Capacity | vLLM/ours throughput | Ours/vLLM median latency | Exact cross-engine outputs | Our peak allocated/reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2953× | 1.2924× | 100% | 1.873 / 1.988 |
| 2 | 1.4459× | 1.4476× | 100% | 1.878 / 1.994 |
| 4 | 1.6537× | 1.7352× | 100% | 1.894 / 2.033 |

All internal reference IDs and all 720 paired cross-engine outputs match.
Cross-engine equality was recorded, not made a universal validity requirement.
Our allocated/reserved peaks are allocator metrics, **not physical VRAM**.
Equivalent vLLM allocator statistics are unavailable (null, not zero).
External process memory is sampled approximately every 0.5 seconds plus query
cost, not an exact comparable peak. Raw samples, current allocated/reserved,
per-request ITL/rates, p90/max, forward work and cache traces remain in the artifact.

Known implementation differences include our Python scheduling/telemetry and
per-tick padded K/V copying versus vLLM's integrated executor/cache/kernel path.
They plausibly affect performance; timing alone does not identify their separate
costs. CUDA graphs cannot explain this measured gap because they were disabled.
Different Transformers versions and preprocessing (vLLM receives prompt IDs;
our HTTP server tokenizes text) are documented confounds. Correlated samples,
chunk-bundled arrival times, EOS lengths, one small model/workload/T4, and FP32
limit generalization. This is neither production capacity nor a tail SLA.

## Provenance

Measured engine/harness snapshot: `14700d46569f2889d9b3ed110cbccf1b293c6c38`.
No `src/` change follows that snapshot in this final publication.
Original run `approved-fp32-20261021T132004512269Z-fc8b7b66` finished
2026-10-21 14:14:03 UTC. All 118 original ZIP manifest entries were verified.
The original ZIP SHA256 is
`24522713a44c6d50fd93c4cca53a072fc97e212353c83dce2d03c521a80ff188`.

The unchanged 40,524,755-byte `gpu_final_comparison.json` SHA256 is
`e609163efa26fdaaafd87895b1f2cffe66d276c462dd631c9507a393efd72a6d`.
Immutable-source validation passes all 60 measured cohorts; this is a
consistency/provenance check, not independent remote-execution attestation.
The public index contains a compact derived summary, not duplicated raw events.
Raw JSON remains linked and downloadable; its final public bytes match the digest.

## Public demo and UI

https://llm-inference-runtime.vercel.app — existing Vercel project/plan.
Public backend/health are `/api` and `/api/health` on that domain.
Live health: `ok`, `distilgpt2`, `cpu`, `continuous_cached_greedy`, cache enabled,
capacity one. No Qwen/GPU inference was deployed to Vercel. Production API/CORS
configuration is present and effective; allowed-origin preflight passes, unrelated
origin is denied. Recorded GPU results are explicitly separate from this live CPU.

Actual public Playground emits real SSE tokens, transitions running → done,
and returns connected/idle. The 128-token request records multiple real network
chunks, matching terminal IDs, Runtime Live/request linking, empty console/page
errors and failed-request lists. Replay selection/step/play/pause works with API
access blocked and makes no API calls. Direct HTTP additionally verifies eight
tokens and one done event. See [public verification](public/public-demo-check.md).

Public telemetry remains instance-local: one post-deploy attempt completed
generation but did not find its request in the later worker snapshot. A subsequent
serial capture links the real request successfully; no global/sticky history
guarantee is claimed. Attempt logs are retained. Route-cleanup cancellation of an
already-pending snapshot poll is recorded separately from actual API/CORS errors.

All four routes pass at 1440/1024/768: twelve responsive captures, no document
overflow or console/network errors. Wide tables/path diagrams scroll locally.
Architecture gains only the adapter boundary. Benchmarks leads with GPU then
CPU length sweep/cached continuous/history. Actual final screenshots are
`docs/images/final-gpu-benchmarks.jpg` and `docs/images/final-runtime.jpg`.
The short README and postmortem state the loss, precision exception and limits.

## Final checks and publication

- Python 3.11.9 locally: **235 passed**, six existing warnings; compileall passes.
- Actual measured Colab snapshot: **228 Python tests passed**, one warning;
  separately the real Qwen GPU correctness/HTTP/SSE gate passed.
- Frontend: **50 passed**, nine files; typecheck, design audit and build pass.
- One UI test timed out while suites ran concurrently; isolated full reruns pass
  without changing tests/timeouts. A short public browser request completed too
  quickly to sample running state; the longer real request supplies the proof.
- Raw artifact validation and public provenance checks pass. Windows CRLF in a
  first static download was corrected without changing the measured artifact.
- Diff whitespace check passes; credential-pattern and tracked-weight/cache
  scans find no secrets or model weights. `.env.example` is public sample config,
  not a credential file. Original execution logs retain incidental environment paths.
- Normal commits/push only; no amend/rebase/squash/force-push. Final tracked
  state and local/origin/GitHub equality are checked after the final evidence push.
- Local ignored external-review ZIP includes this report, README/docs, raw GPU
  and CPU artifacts, hardware, screenshots, test outputs and final Git log.

Phase 3 commits, chronologically:

```text
21ccf4f feat: support modern causal language models
0e9b4e3 bench: prepare reproducible Colab GPU comparison
1c0f69a fix: preserve Colab GPU driver library paths
e699cd8 fix: bootstrap Colab venvs and add one-cell workflow
eb5dec7 test: add offline FP16 batching parity diagnostic
eab17d0 test: probe FP16 numerical policies with strict parity
c4b18e2 test: add diagnostic-only FP32 precision control
a78bbb1 test: isolate vocabulary projection precision
14700d4 bench: add approved native FP32 comparison workflow
af9ad91 bench: make approved FP32 workflow reproducible
da21983 bench: compare GPU runtime with vllm
9f35eb4 feat: publish final GPU benchmark results
dd7f5e8 fix: preserve canonical benchmark download hashes
```

The final `docs: finalize GPU runtime comparison` commit contains this signoff;
its exact hash is in the final chat handoff and local review bundle's Git log.
Deployment application snapshot is `dd7f5e8`; the final commit only adds
documentation/evidence and ignore/byte-preservation rules.

## Final Definition of Done

| # | Requirement | Result / evidence |
| ---: | --- | --- |
| 1 | Explicit autoregressive decoding | PASS — greedy_decode + regression tests |
| 2 | Greedy/sampling | PASS — separate paths and tests |
| 3 | Request-local KV | PASS — cached_decode tests + real GPU references |
| 4 | Cache-length benchmark | PASS — committed CPU sweep |
| 5 | Lifecycle/scheduling | PASS — request/scheduler tests |
| 6 | Static batching | PASS — batching tests |
| 7 | Continuous batching | PASS — continuous_batcher tests |
| 8 | Cached continuous batching | PASS — cached batcher tests + GPU |
| 9 | Variable cached prompt lengths | PASS — CPU/GPU parity |
| 10 | Admission/removal | PASS — 11-frame real GPU trace |
| 11 | Real SSE | PASS — GPU and public raw events |
| 12 | Public live demo | PASS — actual browser inference |
| 13 | Runtime profiler | PASS — Live, linking and Replay captures |
| 14 | CPU evidence | PASS — unchanged committed artifacts |
| 15 | Modern non-GPT2 model | PASS — pinned Qwen |
| 16 | Real CUDA | PASS — T4 hardware/execution evidence |
| 17 | GPU cached continuous | PASS — exact parity capacities 1/2/4 |
| 18 | Same-model/GPU production engine | PASS — unmodified vLLM |
| 19 | Real comparison artifact | PASS — 60 measured cohorts |
| 20 | Publish unfavorable result | PASS — vLLM faster, plainly displayed |
| 21 | Hardware/dtype/versions | PASS — inventories, freezes and policies |
| 22 | No fake PagedAttention | PASS — logical allocator explicitly distinct |
| 23 | Public CPU/GPU distinction | PASS — UI, README and health |
| 24 | Short/plain README | PASS — concise final README |
| 25 | Current architecture/postmortem | PASS — focused final updates |
| 26 | Python/frontend tests | PASS — 235 / 50 |
| 27 | Build/audit/typecheck | PASS — raw check outputs |
| 28 | Genuine history | PASS — normal additive commits |
| 29 | No secrets/weights | PASS — scans and staged review |
| 30 | Evidence pushed | PASS — final evidence commit; remote equality checked in handoff |

PROJECT STATUS: COMPLETE

The demonstrated mechanisms and production comparison are the stopping point.
There is no Phase 4 and no new runtime feature proposed.
