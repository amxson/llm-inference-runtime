import type { Artifact } from "../data/benchmarkAdapter";
import { metric } from "../runtime/traceAdapter";

type Spread = {
  median: number;
  min: number;
  max: number;
  p25: number;
  p75: number;
  p95: number;
};
type Metrics = Record<
  | "wall_ms"
  | "generated_tokens_per_second"
  | "total_input_positions"
  | "model_forward_calls",
  Spread
>;
export interface ContinuousData {
  schema_version: number;
  exact_token_id_parity: boolean;
  host?: {
    before: { battery_status: number };
    after: { battery_status: number };
  };
  config: {
    measured_pairs_per_level: number;
    warmup_cohorts_per_mode_per_level: number;
    workload: { prompt: string; max_new_tokens: number }[];
  };
  levels: {
    max_active: number;
    exact_token_id_parity: boolean;
    aggregates: Record<"uncached" | "cached", Metrics>;
    paired_wall_time_ratios: { summary: Spread };
  }[];
}

export function CachedContinuousBenchmark({
  artifact,
}: {
  artifact: Artifact;
}) {
  const data = artifact.data as ContinuousData;
  if (
    data.schema_version !== 1 ||
    !data.exact_token_id_parity ||
    !data.levels?.length ||
    data.levels.some((l) => !l.exact_token_id_parity)
  )
    return (
      <p role="alert">
        Cached continuous comparison invalid: exact token parity is required.
      </p>
    );
  return (
    <section
      className="instrument-section"
      aria-label="Cached continuous batching comparison"
    >
      <div className="section-title">
        <h2>Continuous batching / cached vs uncached</h2>
        <span className="caption">
          distilgpt2 CPU · whole-cohort timing, not HTTP
        </span>
      </div>
      <p>
        {data.config.workload.length} varied prompts and output budgets;
        capacities 1, 2, 4. Each capacity uses{" "}
        {data.config.warmup_cohorts_per_mode_per_level} excluded warmup cohorts
        per mode and {data.config.measured_pairs_per_level} alternating measured
        pairs. Exact token IDs match. Cache collation copies K/V; lower
        input-position counts do not guarantee lower latency.
      </p>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Continuous batching results; scroll horizontally when needed"
      >
        <table>
          <thead>
            <tr>
              <th>max_active</th>
              <th className="numeric">Median uncached / ms</th>
              <th className="numeric">Median cached / ms</th>
              <th className="numeric">Median paired U/C</th>
              <th className="numeric">Median tok/s U / C</th>
              <th className="numeric">Input positions U / C</th>
              <th className="numeric">Forwards U / C</th>
            </tr>
          </thead>
          <tbody>
            {data.levels.map((l) => (
              <tr key={l.max_active}>
                <td>{l.max_active}</td>
                <td className="numeric">
                  {metric(l.aggregates.uncached.wall_ms.median, 1)}
                </td>
                <td className="numeric">
                  {metric(l.aggregates.cached.wall_ms.median, 1)}
                </td>
                <td className="numeric">
                  {metric(l.paired_wall_time_ratios.summary.median, 2)}×
                </td>
                <td className="numeric">
                  {metric(
                    l.aggregates.uncached.generated_tokens_per_second.median,
                    1,
                  )}{" "}
                  /{" "}
                  {metric(
                    l.aggregates.cached.generated_tokens_per_second.median,
                    1,
                  )}
                </td>
                <td className="numeric">
                  {metric(l.aggregates.uncached.total_input_positions.median)} /{" "}
                  {metric(l.aggregates.cached.total_input_positions.median)}
                </td>
                <td className="numeric">
                  {metric(l.aggregates.uncached.model_forward_calls.median)} /{" "}
                  {metric(l.aggregates.cached.model_forward_calls.median)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        U/C above 1 means cached was faster in that paired statistic; below 1
        means slower. Physical padded input positions are a work proxy, not
        FLOPs. Request latency includes queue waiting; small-sample tails are
        exploratory, not SLAs. No GPU or production capacity claim.
      </p>
      {data.host &&
        data.host.before.battery_status !== data.host.after.battery_status && (
          <p className="caption caveat">
            Host power telemetry changed during this run despite the operator’s
            AC confirmation. Continuous AC power was not verified; timing
            variation and background/thermal conditions are limitations. See raw
            host observations.
          </p>
        )}
      <details>
        <summary>Paired ratio spread and workload</summary>
        <ul>
          {data.levels.map((l) => (
            <li key={l.max_active}>
              max_active={l.max_active}: U/C min{" "}
              {metric(l.paired_wall_time_ratios.summary.min, 2)}×, p25–p75{" "}
              {metric(l.paired_wall_time_ratios.summary.p25, 2)}–
              {metric(l.paired_wall_time_ratios.summary.p75, 2)}×, max{" "}
              {metric(l.paired_wall_time_ratios.summary.max, 2)}×.
            </li>
          ))}
        </ul>
        <ol>
          {data.config.workload.map((r, i) => (
            <li key={i}>
              {r.prompt} · budget {r.max_new_tokens}
            </li>
          ))}
        </ol>
      </details>
      <p className="source-line">
        <a
          href={`${import.meta.env.BASE_URL}data/${artifact.name}`}
          target="_blank"
          rel="noreferrer"
        >
          Raw JSON
        </a>
        <span className="mono">{artifact.source}</span>
        <a
          href={`https://github.com/amxson/llm-inference-runtime/blob/${artifact.source_commit}/${artifact.source}`}
        >
          commit {artifact.source_commit.slice(0, 7)}
        </a>
      </p>
    </section>
  );
}
