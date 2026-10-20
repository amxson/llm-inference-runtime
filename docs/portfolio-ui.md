# Trace portfolio UI

The presentation layer follows the supplied **Trace** direction: a quiet graphite shell, IBM Plex Sans/Mono (400/500), discrete instruments, linked request selection, and one sodium-amber accent. It uses React, TypeScript, Vite, plain CSS, HTML and SVG. No component, icon or chart library is installed.

## Run locally

Use Python 3.11+, Node **22.12+** (Node 24 recommended), and a current npm. The project was verified with Node 24.19.0; an older system npm resolver failed during installation, so npm 11 was used locally without modifying the system installation. The lockfile records frontend dependencies.

Terminal 1, from the repository root:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[server,test]"
$env:LLM_RUNTIME_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"
python -m uvicorn llm_runtime.server:app --host 127.0.0.1 --port 8765
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open [the profiler](http://127.0.0.1:5173/#/runtime). Backend startup loads the real model once; its normal Hugging Face cache remains outside the repository. On Unix, use `source .venv/bin/activate` and `export LLM_RUNTIME_CORS_ORIGINS=...`.

`frontend/.env.example` documents `VITE_API_BASE_URL`; override it in an untracked `.env.local` if necessary. Local defaults do not hard-code a production hostname; the public deployment sets its API base explicitly as described below. CORS is disabled unless explicit origins are supplied, and wildcard origins are rejected. Bind local demos to loopback; this is not authenticated public hosting. For live requests from a preview server, also explicitly allow its origin.

## What is live and what is recorded?

| View | Source and scope |
| --- | --- |
| Playground | Real `POST /generate/stream` SSE. Only prompt/token budget are sent. Greedy, continuous **uncached** serving; no fake sampling/cache controls. |
| Runtime / Live | `GET /runtime/snapshot`, polled every 500 ms. Bounded completed-worker-boundary snapshots; pause freezes the captured frames, not engine execution. |
| Runtime / Replay | Committed recording from real distilgpt2 CPU forwards through `ContinuousBatcher`, with its optional logical allocator. **Not** an HTTP capture. |
| Benchmarks | Build-time mirrors of eight Git-tracked benchmark JSONs: the cache-length sweep, the earlier final rerun and six historical artifacts. Not a live benchmark or a cross-milestone leaderboard. |
| Architecture | An explicit rendering of the paths documented in `docs/architecture.md`, separating uncached HTTP from cached CLI/scheduler execution. |

The top status line identifies replay versus live snapshot. The footer's engine hash identifies the M17 engine baseline, not the current presentation commit. Replay cadence (1×/0.25×) is an exploration control, not measured elapsed time. Keyboard comma/period steps iterations.

The recording has no invented timestamps. Its allocator snapshots are **before forward** and after completion; the grid/table display the before-forward ownership. State/token updates reflect that tick's completed work. Blocks are logical pool IDs, not tensor addresses; no shared-prefix reference counts, copy-on-write, preemption or physical paging is implied. HTTP leaves the allocator disabled and shows “—”.

## Observability additions

- `GET /runtime/snapshot`: active/waiting IDs, queue depth, current batch/capacity, completed tick/forward counts, small request summaries, and the latest 128 trace records. No prompts, runtime inputs, model tensors or DynamicCache contents are serialized.
- SSE token fields additionally include the real engine tick and prefill/decode phase.
- Worker observations include monotonic relative timestamps and worker-step wall duration. This includes dispatch/forward/observation overhead—not isolated kernel latency. Playground shows it only when a matching telemetry tick is available.
- `GET /benchmarks` and `GET /benchmarks/{name}` expose an exact allowlist of committed representative filenames. Arbitrary local paths are never served.

Snapshots are published by the serving owner **between** forwards. Readers never traverse mutating inference-thread state. Polling may skip transient states; missing fields and unobserved transitions stay absent/“—”, rather than becoming synthetic forward events. Client disconnects retain the existing queued-cancellation behavior.

Playground output uses actual decoded SSE prefixes. When individual fragments cannot concatenate safely (e.g. split Unicode), it preserves the correct prefix rather than corrupting text for hover wrappers. Browser-observed TTFT/done arrival include HTTP overhead and are separate from worker timing. Zero-token requests have no TTFT.

## Provenance and regeneration

For a static-only build, leave `VITE_API_BASE_URL` unset. Replay/Benchmarks/Architecture are standalone experiences, and disconnected Live does not attempt localhost. The public production configuration instead sets `VITE_API_BASE_URL=https://llm-inference-runtime.vercel.app/api` and deploys the real runtime with `api/index.py`. Set `LLM_RUNTIME_CORS_ORIGINS=https://llm-inference-runtime.vercel.app` on that API. Secrets must never be placed in a `VITE_` variable (they are public build-time values). See [deployment details](public-runtime.md).

`vercel.json` builds Vite from the repository root (`npm --prefix frontend ci`, `npm --prefix frontend run build`, output `frontend/dist`). API rewrites precede the SPA fallback; API failures cannot turn into HTML-200 health responses. A Python function hosts actual uncached greedy inference; static assets are not served through the model. Local `.vercel/` and generated auth environments are ignored. `benchmarks/provenance.json` is a generated checksum/source-commit manifest for environments without Git metadata; stale/missing artifact hashes fail the build. Regenerate it after committing changed artifacts using `node frontend/scripts/sync-data.mjs --write-manifest`. The CLI deployment does not claim a working GitHub auto-deploy integration.

Review fixes and final measured distributions: [finalization report](finalization.md). The central accent remains the supplied `#EFA23A`; aliases retain full IDs in details, scheduler rows preserve grouped counts, and unmeasured throughput is omitted.

The primary KV-cache section now uses `cache_length_sweep_cpu.json`: 32/128/512-token budgets, two excluded warmups/mode/length, ten alternating measured pairs/length. The compact table separates forwarded-position counts/ratios from median request times and median **paired** latency ratios. Expand the spread table for min/quartiles/p95/max. Data comes from the same checksum/provenance synchronization, not copied React constants. Exact token parity is required before ratios are displayed. The earlier five-pair 128-token rerun is retained under an expandable section; M7 remains historical and M13 is labeled an exploratory serving/concurrency result with its anomalous concurrency-one timing visible.

`frontend/scripts/sync-data.mjs` reads only Git-tracked benchmark files and derives ignored `frontend/public/data/` files, including source commit and content hash. It runs before dev/build/tests. The UI never manually duplicates benchmark values. Raw JSON links point to the derived source copies. Quantized-output drift and slower speculative decoding remain prominent.

To intentionally replace the recorded runtime fixture (requires real model loading):

```powershell
python scripts/record_portfolio_trace.py
python -m pytest tests/test_portfolio_recording.py -q
```

The provenance test verifies FIFO admission, token counts, request stopping and allocator ownership/reuse using real runtime classes, without downloading models. The generator and fixture are not a performance benchmark.

## Verification

```powershell
# Repository root
python -m pytest -q
python -m compileall -q src scripts tests
# frontend/
npm test
npm run typecheck
npm run audit:design
npm run build
npm run preview
```

Frontend tests cover split-byte Unicode/CRLF SSE, terminal/error handling, token updates, cancellation, trace validation/stepping, linked selection, benchmark derivation, missing values, labels and major views. The static audit guards palette, heavy weights, excessive radii, gradients, blur, shadows and forbidden dependencies. It complements—not replaces—visual inspection.

The original UI was browser-reviewed at **1440, 1280, 1024 and 768 CSS pixels**, including full-page captures. Document-wide overflow was checked; dense instruments/paths retain local horizontal scrolling. The narrow layout uses text navigation, not stacked KPI cards. Live generation, cancellation, zero-token completion, live/pause/step, and replay/benchmark exploration with the backend stopped were checked at that revision. These historical captures do not verify the new sweep section. Its requested 1440/1024/768 review must be recorded separately; automated view tests and design audit are not screenshots.

Phase 1's new Benchmarks page was reviewed on 20 October 2026 using a fresh headless Edge browser against the production build at **1440, 1024 and 768 CSS pixels**. [Current captures and browser results](evidence/phase1/browser/results.json) include the full page, work columns, horizontally scrolled latency columns and expanded spread table. The document stayed within each viewport; the dense table uses local scrolling at 1024/768. All three reviews had zero console errors, failed requests or API calls. This is a local production-build review, not public Playground signoff. The desktop browser connection was unavailable; no browser settings were changed.

Accessibility checks include associated form labels, semantic tables/headings, keyboard controls and skip link, visible focus, text alongside status colors, a non-token-chattering status announcement, focusable scrollers, and reduced-motion rules. The exact supplied low-emphasis palette is retained; this is not a claim of a comprehensive WCAG certification.

## Screenshots

[Runtime recording](images/trace-runtime.jpg) · [Benchmark report](images/trace-benchmarks.jpg). Both are actual browser captures, not mockups. Additional local responsive-review captures are ignored under `benchmark_results/portfolio-review/`.

## Deliberate limits

This layer does not add cached batched serving, physical paging, sampling endpoints or production backpressure/security. Finalization adds a scoped rerun using existing decoding mechanisms; historical measurements remain intact. Static data supports backend-free exploration; live generation still requires the Python model server.
