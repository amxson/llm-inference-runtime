# Phase 1 verification — 20 October 2026

This record covers serving safety, the CPU cache-length sweep, benchmark presentation and README cleanup. No cached continuous batching, GPU work or external-engine comparison was added.

## Starting state and history

Local `main`, fetched `origin/main` and the observed GitHub main started at `fa3bed2e348add9ff141b230a2826619eace7b37`. There were no tracked edits. Pre-existing untracked work was left untouched: `assignment_review/`, `abel_coursework_final/`, `alternative_rdbms_solution/`, `coursework_submission/`, `mysql_download_page.html`, `rdbms_runtime_review_20261019/`, `rdbms_submission_review_20261019/` and `rdbms_trigger_revision_20261019/`.

The first six normal commits are `96da844` (server), `116753c` (harness/tests), `8981ae2` (real artifact), `44e07d2` (UI), `4104516` (methods/provenance) and `ad57089` (README). This verification record is the seventh commit. No old commit, timestamp or history was rewritten.

## Server

`ServingRuntime._publish()` now captures the health counters between forwards. `/health` reads `snapshot()`, not the inference thread's live trace. The existing bounded engine-history scope for maximum batch size is retained. No lock or second synchronization scheme was added.

The regression test holds three injected fake-model forwards open and polls actual `/health` requests 60 times. A guarded trace rejects live reads while a forward is active, and each response must report the prior completed boundary. Events/semaphores provide synchronization; no model download or timing-based performance assertion is used.

The worker uses standard `logger.exception`, retaining traceback internally. Tests cover sanitized SSE error/done and HTTP failure, ERROR records with exception information, worker survival and a subsequent successful generation.

## Real cache experiment

- [Raw artifact](../../../benchmarks/results/cache_length_sweep_cpu.json): distilgpt2 CPU FP32, prompt `The capital of France is`, five prompt tokens, 32/128/512 generated tokens. Every requested length was reached.
- Two excluded warmups/mode/length; ten adjacent measured pairs/length, alternating U→C then C→U. Exact token IDs match in all 30 measured pairs.
- Median paired latency ratios: 3.10× / 6.44× / 18.09×. Position ratios: 18.22× / 66.42× / 258.48×. These are different quantities; positions are not FLOPs.
- [Methods, full result table, host observations and limitations](../../finalization.md#cache-length-sweep). AC confirmed, power plan unchanged, four PyTorch threads; point CPU observations 34%/33%. No tests/builds/browser review during measurement. Thermal/background isolation was not established.
- Runtime baseline `96da844`; harness `116753c` was committed after the run began. Actual source-file hashes are recorded and verified. This is not a claim that the harness existed in the baseline parent.
- Historical raw artifacts and timed decoders/batch algorithms were not changed. Existing model-call/Python costs, prefill and attention reads explain why work counts do not predict the same latency ratio; individual timing contributions were not measured.

## Automated checks

| Check | Result | Raw record |
| --- | --- | --- |
| Sweep helpers/artifact integrity | 20 passed | [output](checks/benchmark-helpers.txt) |
| Full Python suite | 151 passed; six existing warnings | [output](checks/python-tests.txt) |
| Frontend suite | 33 passed across seven files | [output](checks/frontend-tests.txt) |
| Typecheck | Exit 0 | [output](checks/typecheck.txt) |
| Trace design audit | Passed | [output](checks/design-audit.txt) |
| Production frontend build | Passed | [output](checks/production-build.txt) |
| Whitespace | Exit 0 | [output](checks/diff-check.txt) |
| Commands, timestamps and exit codes | All zero | [record](checks/results.json) |
| README links and displayed values | All links checked; all rows match raw JSON | [record](checks/readme-links-and-values.json) |
| Eight artifact hashes/source commits | Verified; Git-free manifest fallback also passed | [record](checks/provenance.json) |

These are actual local command results, not hosted CI. Paths are redacted and trailing whitespace normalized. Empty typecheck output apart from npm's heading is expected.

## Browser and README

Headless Edge reviewed the **production build**, not only route HTTP statuses. [Browser results](browser/results.json) record 1440/1024/768 CSS-pixel widths, no document-wide overflow, local table scrolling, expanded timing spread, retained historical concurrency-one anomaly, and zero console errors, failed requests or API calls on Benchmarks. Full-page captures were visually inspected, as were work/latency columns and timing spread. The canonical Trace palette/styles were retained; no chart library or UI dependency was added.

- [1440](browser/benchmarks-1440.png) · [1024](browser/benchmarks-1024.png) · [768](browser/benchmarks-768.png)
- [Work columns / 768](browser/work-columns-768.png) · [latency columns / 768](browser/latency-columns-768.png) · [spread / 768](browser/spread-768.png)
- [GitHub-rendered README table](browser/github-readme-table.png) and [verification](browser/github-readme.json).

README decreased from 267 to 108 lines. CLI/API reference moved into architecture; detailed methods and historical/negative results remain in docs and raw artifacts. Milestone-completion language and the unsupported speculative "poorly aligned" explanation were removed from the front page. INT8 scope and payload-versus-RSS limits remain explicit.

## Environment issues and deployment boundary

The desktop browser connection reported an unavailable native pipe. The bundled headless Edge runtime provided the fresh browser review instead; no permissions/settings were changed. An unrelated ancestor-folder Node 20 installation shadowed Node 24 inside npm scripts. Verification used Node 24.19.0/npm 11 with a local ignored executable link in `frontend/node_modules/.bin`, without changing the ancestor/system installation. An initial wrong-directory test attempt produced only a regenerable Vitest cache; cleanup was blocked by host policy, so that precise generated cache is ignored, not published. The missing default favicon produced a real browser 404; an empty favicon declaration removes that unnecessary request without changing the UI or masking API errors. Final records above are passing reruns, not those failed initial attempts.

The owner authorized updating the existing public Vercel project after the seventh push. The deployment uses a tracked-source archive and existing environment/project settings, with no paid infrastructure change. Public post-push validation is a separate step from these local records: local screenshots and successful builds do **not** by themselves certify a live public Playground. The task handoff must report the actual deployed health/browser/SSE result separately.
