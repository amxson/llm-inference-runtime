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
      <section className="instrument-section">
        <div className="section-title">
          <h2>Live HTTP / uncached continuous greedy</h2>
          <span className="caption">The Playground path</span>
        </div>
        <Pipeline
          label="Live uncached HTTP data path"
          nodes={[
            { title: "HTTP request", detail: "prompt + budget" },
            { title: "Request state", detail: "RequestState" },
            { title: "FIFO admission", detail: "ContinuousBatcher" },
            { title: "Collate / forward", detail: "batch_step · cache=False" },
            { title: "Select token", detail: "logits → argmax" },
            { title: "State update", detail: "record_token" },
            { title: "Stream token", detail: "SSE token / done" },
          ]}
        />
        <p className="caption">
          Repeat a shared model forward for unfinished rows; admit waiting
          requests when slots become free. Full contexts are recollated each
          tick. Completion/cancellation releases the row and client history.
        </p>
        <div className="architecture-notes">
          <div>
            <h3>Prefill vs decode</h3>
            <p>
              Prefill processes a newly admitted prompt. Decode predicts
              continuation tokens. Here, both recompute full contexts—the
              serving path does not retain K/V.
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
              <td>No combined variable-length HF cache</td>
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
