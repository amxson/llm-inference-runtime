import type { Artifact } from "../data/benchmarkAdapter";
import { adaptGpuComparison } from "../data/gpuComparisonAdapter";
import { metric } from "../runtime/traceAdapter";

export function GpuComparison({ artifact }: { artifact: Artifact }) {
  let data: ReturnType<typeof adaptGpuComparison>;
  try { data = adaptGpuComparison(artifact); }
  catch (error) { return <p role="alert">{String(error)}</p>; }
  const gpu = data.hardware.gpus[0];
  const production = data.sessions.find((s) => s.engine === "vllm")?.environment.versions;
  return (
    <section className="instrument-section" aria-label="Recorded modern GPU comparison">
      <div className="section-title">
        <h2>Same model, same GPU, different engine design.</h2>
        <span className="caption">Recorded Colab experiment · not the public backend</span>
      </div>
      <p>
        {data.model} · {gpu.name} · FP32 in both engines. Our cached continuous
        runtime and unmodified vLLM serve the same {data.workload.length} prompts
        over loopback HTTP/SSE, with mixed 32/64/128-token budgets. This page does
        not run benchmarks. The public Playground remains distilgpt2 on CPU;
        its live health is reported separately.
      </p>
      <p className="caption caveat">
        FP32 was explicitly approved after strict FP16 parity failed. This is not
        an FP16 result or a claim that FP16 was fixed. vLLM uses TRITON_ATTN and
        enforce-eager: CUDA graphs/compilation are disabled, so this is not a
        best-tuned vLLM claim. Internal runtime/reference token parity is mandatory;
        cross-engine differences are retained.
      </p>
      <div className="table-scroll" tabIndex={0} role="region"
        aria-label="GPU engine measurements; scroll horizontally when needed">
        <table>
          <thead><tr>
            <th>Concurrency</th><th>Engine</th>
            <th className="numeric">Median latency / ms</th>
            <th className="numeric">p95 / ms</th>
            <th className="numeric">Median TTFT / ms</th>
            <th className="numeric">Median tok/s</th>
            <th className="numeric">Median req/s</th>
          </tr></thead>
          <tbody>{data.aggregates.flatMap((row) => (["runtime", "vllm"] as const).map((engine) => {
            const measurements = row[engine];
            return <tr key={`${row.concurrency}-${engine}`}>
              <td>{row.concurrency}</td><td>{engine === "runtime" ? "Our runtime" : "vLLM"}</td>
              <td className="numeric">{metric(measurements.request_latency_ms.median, 1)}</td>
              <td className="numeric">{metric(measurements.request_latency_ms.p95, 1)}</td>
              <td className="numeric">{metric(measurements.ttft_ms.median, 1)}</td>
              <td className="numeric">{metric(measurements.generated_token_throughput.median, 2)}</td>
              <td className="numeric">{metric(measurements.request_throughput.median, 3)}</td>
            </tr>;
          }))}</tbody>
        </table>
      </div>
      <p className="caption">
        Latency/TTFT pool request observations; throughput is the median whole-cohort
        rate, not the reciprocal of median request latency. Each engine/configuration
        has {data.aggregates[0].runtime.measured_repetitions} measured repetitions.
        ABBA residency sessions rotate concurrency order and exclude two warmup
        cohorts/session/configuration. Small, correlated samples are not tail SLAs.
      </p>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="GPU ratios and allocator observations">
        <table><thead><tr>
          <th>Concurrency</th><th className="numeric">vLLM / ours tok/s</th>
          <th className="numeric">Ours / vLLM latency</th>
          <th className="numeric">Cross-engine exact outputs</th>
          <th className="numeric">Our peak allocated / reserved GiB</th>
        </tr></thead><tbody>{data.aggregates.map((row) => <tr key={row.concurrency}>
          <td>{row.concurrency}</td>
          <td className="numeric">{metric(row.comparison.vllm_over_runtime_token_throughput, 2)}×</td>
          <td className="numeric">{metric(row.comparison.runtime_over_vllm_latency, 2)}×</td>
          <td className="numeric">{metric(row.comparison.exact_output_parity_rate * 100, 1)}%</td>
          <td className="numeric">
            {metric(row.runtime.torch_peak_allocated_bytes == null ? null : row.runtime.torch_peak_allocated_bytes / 2 ** 30, 3)} / {metric(row.runtime.torch_peak_reserved_bytes == null ? null : row.runtime.torch_peak_reserved_bytes / 2 ** 30, 3)}
          </td>
        </tr>)}</tbody></table>
      </div>
      <p className="caption">
        Ratios above 1 favor vLLM in their stated statistic; below 1 favor our runtime.
        Allocated/reserved figures describe our PyTorch allocator only, not physical
        VRAM or vLLM memory. Raw nvidia-smi process samples are available, but no
        exact comparable peak-memory claim is made. Different EOS lengths and
        cross-engine outputs affect throughput denominators.
      </p>
      <details><summary>Environment, workload and limitations</summary>
        <p className="mono">
          Python {data.hardware.python} · Torch {data.hardware.versions.torch} ·
          CUDA package runtime {data.hardware.cuda_runtime} · runtime Transformers {data.hardware.versions.transformers} ·
          vLLM {production?.vllm ?? "—"} / Transformers {production?.transformers ?? "—"}
        </p>
        <p className="mono">Torch-visible device memory {metric(gpu.total_vram_bytes / 2 ** 30, 2)} GiB · compute capability {gpu.compute_capability.join(".")}<br />Recorded {data.started_at_utc} → {data.finished_at_utc}</p>
        <p className="mono">Model revision {data.revision}<br />Measured source {data.source_commit}<br />Raw artifact SHA256 {artifact.source_sha256}</p>
        <p className="caption">Measured source is the executed engine/harness snapshot; the publication commit below may be later. The SHA256 covers the full raw source, not this build-time summary.</p>
        <ul>{data.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
        <ol>{data.workload.map((item) => <li key={item.id}>{item.prompt} · budget {item.max_new_tokens}</li>)}</ol>
      </details>
      <p className="source-line">
        <a href={`${import.meta.env.BASE_URL}data/${artifact.name}`} target="_blank" rel="noreferrer">Raw GPU JSON</a>
        <a href="https://github.com/amxson/llm-inference-runtime/blob/main/docs/evidence/phase3/INDEX.md">Correctness and evidence</a>
        <span className="mono">{artifact.source}</span>
        <a href={`https://github.com/amxson/llm-inference-runtime/blob/${artifact.source_commit}/${artifact.source}`}>publication {artifact.source_commit.slice(0, 7)}</a>
      </p>
    </section>
  );
}
