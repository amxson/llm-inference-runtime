# Public runtime deployment

## Routing and configuration

Canonical frontend: <https://llm-inference-runtime.vercel.app>.

Production API base: <https://llm-inference-runtime.vercel.app/api>.

Health: <https://llm-inference-runtime.vercel.app/api/health>.

The former static-only deployment had no production environment variables or public backend. Its `/health` URL returned SPA HTML with HTTP 200, which was not API health. The current configuration deploys `api/index.py` as a real Python function alongside static Vite assets, with `/api/*` routed before the SPA fallback. No status value or token event is hard-coded to simulate connectivity.

Vercel production variables:

```text
VITE_API_BASE_URL=https://llm-inference-runtime.vercel.app/api
LLM_RUNTIME_CORS_ORIGINS=https://llm-inference-runtime.vercel.app
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

Preview uses `/api` as its same-origin base. The production hostname is an explicit build-time configuration value, not a secret. Changing it requires a frontend rebuild. CORS is an exact allowlist, not authentication; untrusted-origin preflights must not receive an allow-origin header.

## Actual compute, not a proxy to another engine

`api/index.py` uses the existing model loader and `create_app` runtime. It loads real distilgpt2 FP32 weights on CPU and uses the existing manual uncached greedy token loop—no provider API, fake stream or `model.generate()`. The deployment package pins CPU-only torch 2.14.0 and Transformers 5.17.0, with Python 3.12 for the platform runtime. Local core development still supports Python 3.11+.

The Vercel Hobby account was inspected before deployment. No paid host, plan upgrade or local-machine tunnel was provisioned. The runtime uses one CPU thread and one active request row per instance to bound the small public demo; other requests wait under the existing request-capacity policy. This does not remove the engine's separate continuous-batching implementation or imply a throughput claim for public hosting.

Model downloads use `/tmp/llm-runtime-hf`, not the read-only bundle. Cold starts download/load weights and can be slow; Hub rate limits, network failures, platform duration/memory/usage quotas and instance recycling remain limitations. Function duration is configured to 300 seconds, not infinite. This is a portfolio demo, not always-on production capacity.

Each function instance owns its own worker/history. Autoscaling does **not** create a coordinated shared scheduler. Snapshot/batch/queue/forward metrics describe the responding instance, not all replicas; a poll is not guaranteed to reach the instance serving a different connection. Replay and historical benchmark exploration do not poll or wake the model runtime. More reliable global profiling would require a single persistent runtime or explicit instance affinity/aggregation, not a UI label change.

The API is public. CORS does not stop non-browser clients; use only non-sensitive prompts. Recent generated token events are inspectable through runtime telemetry. Production authentication, global rate limits and distributed observability are not claimed.

## Validation

Download-free tests verify the deployment prefix, actual app health/snapshot contracts, SSE event order and exact CORS policy. The real HTTP smoke check is separate:

```powershell
python scripts/check_public_runtime.py --base-url https://llm-inference-runtime.vercel.app/api --origin https://llm-inference-runtime.vercel.app --output benchmark_results/public-runtime/production.json
```

It requires JSON health, correct uncached backend identity, allowed-origin preflight, rejection of an untrusted origin, eight contiguous generated token events, exactly one done, no errors, and matching token/completion IDs. It records actual arrival timestamps and output. An HTML 200 or static artifact checksum cannot satisfy it.

Both the authenticated preview and canonical production domain passed real health and eight-token SSE with the exact production origin. [Production evidence](evidence/final/public-runtime.json) records token IDs, event arrival timestamps, output, preflight results and backend identity. The [final deployment record](evidence/final/deployment-public.json) identifies the deployed application commit and function. These are functional checks, not new benchmark claims. **Browser Playground/console/network signoff** must not be inferred from a command-line test. Browser tooling remained unavailable in this Work session; the owner requested continued non-browser validation instead of repeated permission requests. Follow the [exact manual browser checklist](public-browser-checklist.md) for the remaining interaction checks. No browser signoff is claimed.

Platform references: [Python runtime and streaming](https://vercel.com/docs/functions/runtimes/python), [large Python functions](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size). Support for these features is not proof this application is healthy; the actual inference checks are required.
