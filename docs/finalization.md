# Final measurements and review boundaries

## Facts: final quiet-window rerun

Source: [raw final artifact](../benchmarks/results/final_cpu.json), generated with [the finalization harness](../scripts/benchmark_portfolio.py). The recorded source revision identifies the unchanged runtime baseline at measurement time; the new harness/artifact is committed in the subsequent finalization commit. Historical JSON and the earlier evidence ZIP were retained unchanged.

The owner confirmed AC power and paused heavy work. The existing HP Optimized power plan was retained, not silently changed. Intel i7-1165G7, CPU FP32 distilgpt2, Python 3.11.9, torch 2.14.0+cpu, Transformers 5.17.0. CLI execution used four PyTorch threads. No tests, builds or browser captures ran during measurement. Read-only host CPU samples were **21% before / 32% after**: residual background activity existed. This is a quiet-window rerun, **not an isolated/idle-host certification**. HTTP server thread count was not captured and must not be assumed equal to the CLI setting.

The existing M7 harness ran two excluded warmups per mode and five alternating measured pairs. Same model, prompt `The capital of France is`, 128 generated tokens; loading/final text decode excluded, timing begins before tokenization and ends at selection. All ten runs had identical generated IDs. Mean local request latency: **11,080.8 ms uncached / 2,032.9 ms cached**; observed ratio **5.45x**, specific to this workload. Forwarded positions: **8,768 / 132**. These are work counts, not FLOPs. The longer workload differs from historical M7's ten-token budget; do not attribute cross-run differences to a machine fix.

Network measurement was separate: two excluded warmups, then five serial real HTTP/SSE requests, each with **128 contiguous token events, exactly one done, zero errors**. Worker traces were read only after completion; no high-frequency snapshot observer was added during measured requests. Token IDs must match completion IDs or the harness refuses the result. Network arrival timings include transport/delivery; they are not CLI selection timings or pure model-forward latency.

| Pooled distribution / ms | Count | Median | p90 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Worker steps, including prefill | 640 | 127.34 | 186.96 | 215.34 | 1,404.60 |
| Decode worker steps only | 635 | 127.76 | 187.14 | 215.50 | 1,404.60 |
| HTTP inter-token gaps | 635 | 128.15 | 187.51 | 216.10 | 1,405.20 |

Worker-step scope is dispatch + forward + observation overhead. Quantiles use linear interpolation over pooled observations, not independent-trial confidence intervals. Keep the large maximum visible; do not discard it as an outlier merely to improve presentation. Five requests/one prompt are exploratory, not tail-SLA validation.

![Actual mean per-position gaps, separately scoped](images/final-token-gaps.svg)

Regenerate this chart with `python scripts/render_timing_evidence.py`; the UI independently derives the same means from source JSON. Both retain all measured positions; straight segments are not a smoothed fit. Cached local selection is a supported comparison, while **cached HTTP serving is not implemented**. Attention still reads prior K/V; neither perfectly flat latency nor universal speedup follows from architecture.

## Hypothesis, not diagnosis

The earlier evidence's abrupt mid-request latency change and M13 concurrency-one anomaly are consistent with changing host conditions/contention. The evidence does not establish causality. Historical M5/M7/M13 timings remain preserved and labeled historical/potentially contention affected, not the preferred headline. The new run also retains residual-load and worker-tail limitations. No optimization percentage is inferred by dividing unrelated historical baselines.

## UI review resolutions

- Live aliases (`r1`, `r2`, …) are stable within a browser session. Selection/data keys remain full request IDs; tooltips and request details retain them. Replay retains its original IDs. Aliases are presentation, not server identity.
- Removed the unmeasured `decode — tok/s` field rather than imply a valid throughput estimate.
- Public static mode has no default backend URL and does not poll localhost. Live explains the disconnected deployment and points to Replay; Playground's Run is disabled until an explicit API URL is configured. In local development the loopback default remains available.
- Missing/empty live data clears selection and dependent panels, including paused captures. No stale `r2` header remains.
- Scheduler rows collapse only consecutive identical membership/queue/phase states without admission/completion, preserving tick range, order and total count. Distinct admission/completion or tick gaps never merge. The server's retained-window limit remains explicit.
- Small-model greedy repetition is explained, not concealed or replaced with different model output.
- The supplied central Trace token `#EFA23A` is canonical; no second design surface/accent was introduced.
- The giant blank band was traced to **real newline-heavy generated output**, not virtualization or a minimum-height bug: the Runtime token panel used `white-space: pre-wrap`, so each emitted newline reserved another blank line. Trace inspection now visibly escapes control characters (`\n`, `\r`, `\t`) and explicitly labels this representation; raw decoded prefixes and IDs remain in details. Playground keeps the actual streamed text and the small-model repetition caption. No model output is changed and no arbitrary height cap hides tokens. Focus/scroll positioning can separately affect fixed headers in full-page captures; final evidence records geometry and viewport captures.

## Reproduce

Start the local server with intended thread settings explicitly set **before** process startup. To retain comparable future network environments, record/confirm the server setting separately; the final artifact does not retrospectively invent it.

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
python -m uvicorn llm_runtime.server:app --host 127.0.0.1 --port 8765
# In another terminal, with the host on AC and heavy work paused:
python scripts/benchmark_portfolio.py --confirm-ac-quiet --output benchmark_results/new-final-run.json
```

Use a new output path to preserve the committed run. Do not run tests/builds/captures concurrently with measurement. Do not enable cached controls on the uncached HTTP server.
