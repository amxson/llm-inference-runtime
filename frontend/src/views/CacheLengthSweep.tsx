import type { Artifact } from "../data/benchmarkAdapter";
import { metric } from "../runtime/traceAdapter";

interface Distribution {
  count: number;
  median: number;
  p25: number;
  p75: number;
  p95: number;
  min: number;
  max: number;
}
export interface SweepData {
  schema_version: number;
  recorded_at_utc: string;
  exact_token_id_parity: boolean;
  environment: { model_name: string; torch_thread_count: number };
  config: {
    prompt: string;
    warmup_runs_per_mode_per_length: number;
    measured_pairs_per_length: number;
  };
  levels: {
    requested_tokens: number;
    observed_generated_tokens: number[];
    measured_pair_count: number;
    exact_token_id_parity: boolean;
    all_runs_reached_budget: boolean;
    work_positions: {
      uncached_positions: number;
      cached_positions: number;
      position_ratio_uncached_over_cached: number;
    } | null;
    aggregates: Record<
      "uncached" | "cached",
      { total_latency_ms: Distribution }
    >;
    paired_latency_ratios: { summary: Distribution };
  }[];
}

export function CacheLengthSweep({ artifact }: { artifact: Artifact }) {
  const data = artifact.data as SweepData;
  if (
    data.schema_version !== 1 ||
    !data.exact_token_id_parity ||
    !data.levels?.length ||
    data.levels.some((level) => !level.exact_token_id_parity)
  ) {
    return (
      <p role="alert">
        Cache sweep invalid: exact token parity is required before showing
        latency ratios.
      </p>
    );
  }
  return (
    <section
      className="instrument-section"
      aria-label="Controlled KV cache length sweep"
    >
      <div className="section-title">
        <h2>KV cache / generation-length sweep</h2>
        <span className="caption">
          Single-request CPU · median paired timings
        </span>
      </div>
      <p>
        How does reuse change with longer output? Each length has{" "}
        {data.config.measured_pairs_per_length} measured cached/uncached pairs
        and {data.config.warmup_runs_per_mode_per_length} excluded warmups per
        mode. Exact generated token IDs match in every pair. Work counts and
        elapsed time are different quantities.
      </p>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Cache sweep work and latency; scroll horizontally when needed"
      >
        <table>
          <thead>
            <tr>
              <th rowSpan={2}>Budget / generated</th>
              <th colSpan={3}>Work / forwarded input positions</th>
              <th colSpan={3}>Time / local request latency</th>
            </tr>
            <tr>
              <th className="numeric">Uncached positions</th>
              <th className="numeric">Cached positions</th>
              <th className="numeric">Position ratio U/C</th>
              <th className="numeric">Median uncached / ms</th>
              <th className="numeric">Median cached / ms</th>
              <th className="numeric">Median paired ratio U/C</th>
            </tr>
          </thead>
          <tbody>
            {data.levels.map((level) => (
              <tr key={level.requested_tokens}>
                <td className="mono">
                  {level.requested_tokens} /{" "}
                  {level.observed_generated_tokens.join(", ")}
                </td>
                <td className="numeric">
                  {metric(level.work_positions?.uncached_positions)}
                </td>
                <td className="numeric">
                  {metric(level.work_positions?.cached_positions)}
                </td>
                <td className="numeric">
                  {metric(
                    level.work_positions?.position_ratio_uncached_over_cached,
                    2,
                  )}
                  ×
                </td>
                <td className="numeric">
                  {metric(level.aggregates.uncached.total_latency_ms.median, 2)}
                </td>
                <td className="numeric">
                  {metric(level.aggregates.cached.total_latency_ms.median, 2)}
                </td>
                <td className="numeric">
                  {metric(level.paired_latency_ratios.summary.median, 2)}×
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        U/C = uncached divided by cached. The final column is the median of
        matched-pair latency ratios, not a ratio of aggregate means. Positions
        are not FLOPs or a latency prediction. Budget and observed generated
        length are shown separately; EOS is honored. If work counts vary across
        pairs, the single count is omitted; inspect the raw per-pair counts.
      </p>
      <details className="result-details">
        <summary>Timing spread / all measured pairs</summary>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Cache sweep timing distributions"
        >
          <table>
            <thead>
              <tr>
                <th>Budget / statistic</th>
                <th className="numeric">Count</th>
                <th className="numeric">Min</th>
                <th className="numeric">p25</th>
                <th className="numeric">Median</th>
                <th className="numeric">p75</th>
                <th className="numeric">p95</th>
                <th className="numeric">Max</th>
              </tr>
            </thead>
            <tbody>
              {data.levels.flatMap((level) =>
                [
                  ["Uncached / ms", level.aggregates.uncached.total_latency_ms],
                  ["Cached / ms", level.aggregates.cached.total_latency_ms],
                  ["Paired ratio / ×", level.paired_latency_ratios.summary],
                ].map(([name, raw]) => {
                  const d = raw as Distribution;
                  return (
                    <tr key={`${level.requested_tokens}-${name}`}>
                      <td>
                        {level.requested_tokens} · {String(name)}
                      </td>
                      <td className="numeric">{d.count}</td>
                      {[d.min, d.p25, d.median, d.p75, d.p95, d.max].map(
                        (value, index) => (
                          <td className="numeric" key={index}>
                            {metric(value, 2)}
                          </td>
                        ),
                      )}
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        </div>
      </details>
      <p className="caption">
        Same loaded {data.environment.model_name}, CPU FP32,{" "}
        {data.environment.torch_thread_count} PyTorch threads. Alternating
        adjacent pairs; ascending length order. Timing starts before
        tokenization and ends at last selection; model loading and final text
        decoding are excluded. Empirical quantiles use linear interpolation. One
        prompt/host; thermal or background drift is possible. Ten pairs do not
        establish tail SLAs or GPU behavior.
      </p>
      <p className="caption">
        Why did historical M7 reduce positions 95 → 14 but latency only about
        1.83×? Its tiny sequences and small distilgpt2 model still pay
        Python/model-call costs for every generated step. Both modes perform
        prompt prefill; cached attention still reads earlier K/V. These
        mechanisms explain why the two ratios need not match; their individual
        timing contributions were not measured. The length sweep above measures
        the response rather than assuming a monotonic speedup.
      </p>
      <div className="method-block mono">
        prompt “{data.config.prompt}”<br />
        Recorded {data.recorded_at_utc}
      </div>
      <p className="source-line">
        <a href={`${import.meta.env.BASE_URL}data/${artifact.name}`}>
          Sweep raw JSON
        </a>
        <a
          className="mono"
          href={`https://github.com/amxson/llm-inference-runtime/blob/${artifact.source_commit}/${artifact.source}`}
        >
          commit {artifact.source_commit.slice(0, 7)}
        </a>
        <span className="mono">
          sha256 {artifact.source_sha256.slice(0, 12)}
        </span>
      </p>
    </section>
  );
}
