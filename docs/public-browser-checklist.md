# Public browser verification checklist

This checklist is for the owner to run. It is **not** a record of completed browser verification. The agent's browser tool remained unavailable in this Work session; the owner requested continued HTTP validation and a manual checklist instead of repeated permission requests.

## 1. Load the actual public Playground

1. Open <https://llm-inference-runtime.vercel.app/#/playground> in a fresh browser tab.
2. Open Developer Tools. In Network, enable **Preserve log** and **Disable cache**, then hard-refresh. Open Console and clear old messages.
3. Confirm the application renders. In Network, filter `/api/`: `GET /api/runtime/snapshot` should return **200 JSON**, not the HTML application shell.
4. Wait for model initialization on a cold instance. The header should show `distilgpt2`, `cpu`, `max_batch=1`, and **idle** (or **running** during work), not a permanently hard-coded connected badge. Initial offline status while a request is pending is not proof of a failed backend; a failed response or persistent error is.
5. Open <https://llm-inference-runtime.vercel.app/api/health> directly. Expect JSON with `status: "ok"`, `model: "distilgpt2"`, `device: "cpu"`, `engine: "continuous_uncached_greedy"`, `max_active: 1`.

## 2. Submit real inference from the UI

1. Use only this non-sensitive prompt: `The capital of France is`. Set **max_new_tokens** to **8**, then click **Run**.
2. Confirm output appears incrementally and the stream status ends at **done**, with **Generated = 8**. The public HTTP check produced the continuation ` the capital of the French Republic. The`; this is model output, not a high-quality assistant answer.
3. In Network, inspect **POST /api/generate/stream**: status **200**, response content type **text/event-stream**. In the EventStream/Response view, expect eight `token` events with `token_index` 1 through 8, followed by exactly one `done` event, and no `error` event. Token IDs must match `done.generated_token_ids`.
4. Confirm events arrive over time, rather than only after completion. The UI must not simulate timing with a typewriter animation. Token selection should show the real ID and event details.
5. Inspect response headers: `access-control-allow-origin` should be `https://llm-inference-runtime.vercel.app`. Since frontend and API are same-origin, the browser may not send a preflight; that is normal. Separate HTTP checks exercise allowed and untrusted-origin preflights.
6. Clear Network filters and review Console. There should be no failed API requests, CORS errors, unhandled exceptions, or localhost API requests. A successful stream does not excuse a failing snapshot poll.

## 3. Review live mode honestly

1. Open **Runtime**, select **Live**, and confirm the header uses the live model/device and an idle/running state based on successful snapshots.
2. Inspect any available iteration/request and its full request ID in details. Do not mistake a presentation alias for the backend ID.
3. Live telemetry is **instance-local**. A snapshot can reach a different serverless replica than the stream; absence of that request in another instance is not evidence of a shared global scheduler. Do not mark cross-replica request linking as verified without observing it.

## 4. Verify Replay without the backend

1. Navigate to <https://llm-inference-runtime.vercel.app/#/runtime> and select **Replay**. Allow any in-flight live poll to cancel.
2. In DevTools Network request blocking, block `*llm-inference-runtime.vercel.app/api/*` only. Do not block static assets or put the entire browser offline.
3. Hard-refresh on Runtime (default mode is Replay). Verify recorded iterations render; **Play**, **Pause**, previous/next iteration, speed, and request selection work. The header should say **replay**, not claim a live connection. There should be no new API polling while in Replay.
4. Visit **Benchmarks** and **Architecture**. Static content should load without calling the backend. At desktop width and a 390-pixel mobile viewport, confirm navigation, controls, and details remain usable.
5. With API blocking still enabled, choose **Live** and then **Playground**. Confirm they truthfully report unavailable/failed live requests, do not fabricate output, and allow returning to Replay. Failed API requests are **expected only during this deliberate failure test**.
6. Disable request blocking, clear Console/Network, return to Playground, and repeat the eight-token request. Confirm recovery with no unexpected API/CORS failures.

## Evidence to retain

- A screenshot of connected Playground (idle/done plus model/device), one of Runtime Live, and one of Replay with API blocking enabled.
- A short screen recording of clicking Run and incremental tokens arriving, plus a screenshot of the SSE event sequence and response headers.
- A fresh Console/Network review after removing request blocking. If exporting a HAR, remove cookies, authorization/bypass headers, identifiers and sensitive prompts before sharing it.
- Record pass/fail for each section and any exact error. Do not mark public browser signoff complete based only on the separate HTTP evidence.
