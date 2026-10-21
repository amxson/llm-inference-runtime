const repo = "https://github.com/amxson/llm-inference-runtime";
function Pipeline({
  label,
  nodes,
}: {
  label: string;
  nodes: { title: string; detail: string }[];
}) {
  return (
    <div
      className="pipeline-scroll"
      tabIndex={0}
      role="region"
      aria-label={`${label}; scroll horizontally to inspect the full path`}
    >
      <ol className="pipeline" aria-label={label}>
        {nodes.map((node) => (
          <li key={node.title}>
            <span>{node.title}</span>
            <span className="mono">{node.detail}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
export function ArchitectureView() {
  return (
    <>
      <div className="page-heading">
        <p className="eyebrow">Architecture / implemented data paths</p>
        <h1>One engine. Distinct execution paths.</h1>
        <p>
          Transformers supplies model operations. This repository owns the token
          loops, request lifecycle, scheduling, batching, and stream delivery
          below model.generate().
        </p>
      </div>
      <p className="caption">
        Live requires a connected Python runtime and observes worker-boundary
        snapshots/SSE arrivals. Replay is an engine-generated cached execution
        recording. Replay and benchmark assets are static; Live uses the
        connected Python API. No fresh benchmarks run in the browser; the
        harness separately measures local selection time and network arrival
        time.
      </p>
      <section className="instrument-section">
        <h2>Model-specific execution, shared scheduler</h2>
        <p>
          GPT-2/distilgpt2 and full-attention Qwen2 use a small forward-input
          adapter. Qwen drops GPT-2 token-type inputs, derives rotary position
          IDs from each row’s logical mask, and keeps only the final-position
          logits during token loops. Cache ownership, collation/splitting,
          admission and SSE remain in the same engine—not separate model schedulers.
        </p>
        <p className="caption">
          The recorded Qwen/T4 experiment is separate from the public CPU
          Playground and its CPU replay recording. The adapter targets the
          inspected Transformers API and full-attention DynamicCache; it does
          not establish compatibility with arbitrary model/cache families.
        </p>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Live HTTP / cached continuous greedy</h2>
          <span className="caption">
            Default local path · Live status reports the configured engine
          </span>
        </div>
        <Pipeline
          label="Live cached HTTP data path"
          nodes={[
            { title: "HTTP request", detail: "prompt + budget" },
            { title: "FIFO admission", detail: "CachedContinuousBatcher" },
            { title: "Prompt prefill", detail: "masks + logical positions" },
            { title: "Request-local KV", detail: "compact DynamicCache" },
            { title: "Cache collation", detail: "left pad + mask" },
            {
              title: "Batched one-token decode",
              detail: "[B, 1] · cache=True",
            },
            { title: "Cache split / update", detail: "trim padding + select" },
            { title: "SSE / completion", detail: "retain or release KV" },
          ]}
        />
        <p className="caption">
          Repeat a shared model forward for unfinished rows; admit waiting
          requests when slots become free. Each tick has at most one prefill
          batch and one cached decode batch; one selected token per request.
          Completion/cancellation releases the row and client history.
        </p>
        <div className="architecture-notes">
          <div>
            <h3>Prefill vs decode</h3>
            <p>
              Prefill processes a newly admitted prompt. Decode predicts
              continuation tokens using only the newest input plus prior K/V.
              Position IDs follow logical history, not synthetic cache padding.
            </p>
          </div>
          <div>
            <h3>Scheduling vs batching</h3>
            <p>
              Admission chooses which requests run. Batching chooses which rows
              share a forward. Continuous batching refills vacancies instead of
              waiting for the whole cohort.
            </p>
          </div>
          <div>
            <h3>Single serving owner</h3>
            <p>
              Commands are handled between forwards. Model work runs off the
              event loop; disconnects queue cancellation rather than mutate an
              in-flight batch.
            </p>
          </div>
        </div>
      </section>
      <section className="instrument-section">
        <h2>Uncached reference / explicit alternative</h2>
        <Pipeline
          label="Live uncached HTTP data path"
          nodes={[
            { title: "HTTP request", detail: "continuous_uncached_greedy" },
            { title: "FIFO admission", detail: "ContinuousBatcher" },
            {
              title: "Full-context collation",
              detail: "batch_step · cache=False",
            },
            { title: "Select / complete", detail: "argmax + SSE" },
          ]}
        />
        <p className="caption">
          Retained as the correctness reference and benchmark baseline. Backend
          health/snapshot report the actual selected engine; the public demo
          keeps its own capacity limit.
        </p>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Separate path / request-local KV reuse</h2>
          <span className="caption">
            CLI decoder and round-robin scheduler · not the HTTP path
          </span>
        </div>
        <Pipeline
          label="Separate cached decoding data path"
          nodes={[
            { title: "Prompt prefill", detail: "use_cache=True" },
            { title: "Cache + logits", detail: "HF DynamicCache" },
            { title: "Token selection", detail: "argmax + mask update" },
            { title: "Newest-token decode", detail: "past_key_values" },
            { title: "Next / complete", detail: "reuse or release" },
          ]}
        />
        <p className="caption">
          After prefill, only the newest token is forwarded with prior K/V.
          Attention still reads the cached prefix. DecodeScheduler advances
          independently cached requests round-robin; it does not execute a
          multi-row cached batch.
        </p>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Request lifecycle</h2>
          <span className="caption">Actual RequestStatus transitions</span>
        </div>
        <div className="state-machine">
          <span>waiting</span>
          <span>→ prefill</span>
          <span>→ decoding</span>
          <span>→ finished</span>
        </div>
        <p className="caption architecture-caption">
          EOS, token limit and zero-token requests finish explicitly. Any
          unfinished request can finish through cancellation/error. Runtime
          tensors/cache are cleared. No preemption, shared-prefix refcounts, or
          copy-on-write is implemented.
        </p>
      </section>
      <section className="instrument-section">
        <h2>Supporting mechanisms, not one fused pipeline</h2>
        <table className="architecture-table">
          <thead>
            <tr>
              <th>Mechanism</th>
              <th>What exists</th>
              <th>Boundary</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Manual decoding / sampling</td>
              <td>
                Explicit forwards, temperature/top-k/top-p and local seeds
              </td>
              <td>Sampling is separate from greedy-only HTTP</td>
            </tr>
            <tr>
              <td>Static / continuous batching</td>
              <td>Left padding, masks/positions and independent stopping</td>
              <td>
                Cached path copies padded K/V then splits compact rows; not
                physical paging
              </td>
            </tr>
            <tr>
              <td>Logical block allocator</td>
              <td>Fixed pool, request tables, growth, free/reuse</td>
              <td>Capacity modeling—not physical tensors or PagedAttention</td>
            </tr>
            <tr>
              <td>Quantization experiment</td>
              <td>Dynamic INT8 attention/MLP projections</td>
              <td>Changed output; weight payload is not process RSS</td>
            </tr>
            <tr>
              <td>Speculative decoder</td>
              <td>Draft → verify → matching prefix → correction</td>
              <td>Uncached greedy; slower in the recorded workload</td>
            </tr>
            <tr>
              <td>Benchmark harnesses</td>
              <td>Raw runs, paired comparisons, network concurrency</td>
              <td>Small CPU workloads—not production capacity claims</td>
            </tr>
          </tbody>
        </table>
      </section>
      <section className="instrument-section">
        <h2>Read the engineering record</h2>
        <div className="document-links">
          <a href={`${repo}#readme`}>README</a>
          <a href={`${repo}/blob/main/docs/architecture.md`}>
            Architecture source
          </a>
          <a href={`${repo}/blob/main/docs/postmortem.md`}>Postmortem</a>
          <a href={repo}>GitHub repository</a>
        </div>
        <p className="caption">
          Derived from docs/architecture.md and the implemented source. No
          physical paging, optimized cached serving, distributed execution,
          custom attention kernel, or production hardening is implied.
        </p>
      </section>
    </>
  );
}
