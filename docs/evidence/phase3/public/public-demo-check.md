# Final public demo review

Public URL: https://llm-inference-runtime.vercel.app
Backend: https://llm-inference-runtime.vercel.app/api
Health: https://llm-inference-runtime.vercel.app/api/health

The existing Vercel project/plan is retained. No GPU, paid resource, premium
Colab tier, tunnel or plan change was provisioned. The CPU Python function keeps
the existing large-function setting. Qwen weights are not deployed to Vercel.

Production configuration names were checked with `vercel env ls production`:
`VITE_API_BASE_URL`, `LLM_RUNTIME_CORS_ORIGINS`, and
`VERCEL_SUPPORT_LARGE_FUNCTIONS` are present. Secret values were not exported.
Observed browser requests use the public same-origin `/api` base. Allowed-origin
preflight passes; an unrelated origin is denied. The successful frontend stream
also verifies the effective API base, not just the existence of an environment name.

## Actual runtime checks

- `public-health.json`: JSON health is `ok`, model `distilgpt2`, device `cpu`,
  engine `continuous_cached_greedy`, serving cache `true`, capacity one.
- Direct public HTTP/SSE: eight token events, one terminal completion, no error;
  indices and IDs match the terminal result. Output starts
  ` the capital of the French Republic. The`.
- `live-browser/results.json`: a real public Playground form submission requests
  128 tokens. The browser sees actual output while state is `running`, then
  `done`, and the top live status returns to `idle` (connected, not hard-coded).
  The observed response is cloned only for recording; its original is returned
  unchanged to the frontend. No tokens, health or status are mocked.
- Raw SSE transcript has 128 token events and one done event. Multiple real
  network chunks arrive over time; chunk timestamps are not fabricated
  per-token timings or a serving benchmark.
- Runtime Live displays the same request ID, cached one-token decode and output.
- Replay request/cache selection, next/play/pause work in a separate context
  with API routes blocked; Replay makes zero API requests.
- Console errors, browser page errors, failed network requests and HTTP errors
  are recorded and asserted empty before deliberate context cleanup.

One post-redeployment attempt completed SSE but could not find its request in
the subsequently polled Live snapshot. Vercel snapshots are instance-local,
not globally coordinated or sticky; the successful linking capture does not
promise cross-instance request history. This limitation is retained, not hidden.
Another harness attempt counted React's intentional route-change cancellation
of a pending snapshot poll as a failure. The final observer separately records
only those `ERR_ABORTED` requests that were already pending when it deliberately
changed routes. HTTP/API/CORS errors and unrelated aborts still fail the check.
Original attempt failures are preserved in `attempts/`.

The initial eight-token browser attempt finished before the automation could
sample its running state. It was not claimed as streaming proof. The longer
real request supplies that proof without slowing or mocking the backend.

## Responsive review

All four routes were checked at 1440, 1024 and 768 pixels in isolated headless
Edge contexts. The built-in browser tool reported disabled access; the user's
earlier authorization to use any available browser was followed. No personal
Chrome profile, Colab window, desktop mouse or keyboard was used.

`responsive/results.json` records all 12 checks: document width never exceeds
viewport width, no console/network errors, and static/Replay pages do not poll
the inference backend. Real JPEG captures are in `responsive/`. Wide instrument
tables/architecture paths intentionally scroll within their regions. The GPU
table was captured at both horizontal ends at every width. Screenshots were
visually inspected; these are not generated mockups.

GPU results lead the Benchmarks page, followed by the CPU cache-length sweep,
cached continuous experiment, historical and exploratory results. Architecture
adds only the focused GPT-2/Qwen input-adapter boundary. The live CPU and recorded
FP32 T4 environments are explicitly distinct.

## Download provenance correction

Windows source archiving initially introduced CRLF into the static raw JSON
download. Its canonical JSON was unchanged, but physical bytes did not match
the displayed canonical-LF digest. `dd7f5e8` normalizes mirror line endings to
the same canonical LF bytes used by provenance. A regression checks the actual
download file's SHA256; `public-artifacts.json` records the final public check.
The original measured JSON and benchmark Git blob were never changed.

The deployed application source is `dd7f5e8cf5a96c809c3b195d578ab52cc1664d58`.
Final production deployment: `dpl_J5a9PLp4nNq6UzrBptvf5BsbAXHa`, ready and
aliased to the canonical URL. Final browser/live and responsive records are dated
2026-10-21 15:04 UTC; health/SSE and raw-download checks also follow this deployment.
Subsequent final documentation/evidence changes do not change the deployed
runtime, UI or measurements. See [the final report](../FINAL_REPORT.md).
