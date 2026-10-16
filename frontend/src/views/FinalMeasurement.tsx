import type { Artifact } from "../data/benchmarkAdapter";
import { metric } from "../runtime/traceAdapter";

interface Distribution {
  count: number;
  median: number;
  p90: number;
  p95: number;
  max: number;
}
interface Run {
  inter_token_latencies_ms: number[];
  total_latency_ms: number;
  generated_token_count: number;
}
export interface FinalData {
  recorded_at_utc: string;
  environment: {
    model_name: string;
    torch_thread_count: number;
    host_before: { cpu: string; power_plan: string };
  };
  config: {
    max_new_tokens: number;
    measured_runs_per_mode: number;
    warmup_runs_per_mode: number;
  };
  paired: {
    uncached: { runs: Run[]; summary: { mean_total_latency_ms: number } };
    cached: { runs: Run[]; summary: { mean_total_latency_ms: number } };
    comparison: { exact_token_id_parity: boolean };
  };
  network_runs: {
    token_gaps_ms: number[];
    token_events: number;
    done_events: number;
    error_events: number;
  }[];
  worker_step_ms: Distribution;
  decode_worker_step_ms: Distribution;
  network_token_gap_ms: Distribution;
}
export function meanByPosition(rows: number[][]): number[] {
  if (!rows.length) return [];
  const length = Math.min(...rows.map((row) => row.length));
  return Array.from(
    { length },
    (_, i) => rows.reduce((sum, row) => sum + row[i], 0) / rows.length,
  );
}
export function TimingFigure({
  series,
  label,
}: {
  series: { name: string; values: number[] }[];
  label: string;
}) {
  const max = Math.max(1, ...series.flatMap((s) => s.values));
  const length = Math.max(2, ...series.map((s) => s.values.length));
  const x = (i: number) => 64 + (i / (length - 1)) * 688;
  const y = (v: number) => 236 - (v / max) * 180;
  return (
    <svg
      className="timing-figure"
      viewBox="0 0 820 288"
      role="img"
      aria-label={label}
    >
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <path d={`M64 ${y(max * f)}H752`} stroke="var(--axis)" />
          <text
            x="56"
            y={y(max * f) + 4}
            textAnchor="end"
            className="mono svg-axis"
          >
            {metric(max * f, 1)}
          </text>
        </g>
      ))}
      <text x="4" y="24" className="svg-axis">
        ms
      </text>
      {series.map((s, j) => (
        <g key={s.name}>
          <text x={64 + j * 300} y="24" className="svg-label">
            {s.name}
          </text>
          <polyline
            fill="none"
            stroke={j === 1 ? "var(--accent)" : "var(--data-computed)"}
            strokeWidth="1.5"
            points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
          />
          {s.values.map((v, i) => (
            <circle
              key={i}
              cx={x(i)}
              cy={y(v)}
              r="1.5"
              fill={j === 1 ? "var(--accent)" : "var(--data-computed)"}
            >
              <title>
                {s.name}, token {i + 2}: {v.toFixed(3)} ms
              </title>
            </circle>
          ))}
        </g>
      ))}
      <text x="64" y="264" className="mono svg-axis">
        2
      </text>
      <text x="752" y="264" textAnchor="end" className="mono svg-axis">
        {length + 1}
      </text>
      <text x="410" y="280" textAnchor="middle" className="svg-axis">
        Output token position (first token / TTFT excluded)
      </text>
    </svg>
  );
}
export function FinalMeasurement({ artifact }: { artifact: Artifact }) {
  const data = artifact.data as FinalData;
  const paired = data.paired;
  return (
    <section
      className="instrument-section"
      aria-label="Final controlled measurement"
    >
      <div className="section-title">
        <h2>Final rerun / token-level timing</h2>
        <span className="caption">
          Single-request CPU · separate CLI and HTTP scopes
        </span>
      </div>
      <p>
        Fact: {data.config.measured_runs_per_mode} alternating measured pairs,{" "}
        {data.config.warmup_runs_per_mode} excluded warmups per mode,{" "}
        {data.config.max_new_tokens} generated tokens. Exact cached/uncached
        token parity:{" "}
        {paired.comparison.exact_token_id_parity ? "verified" : "not verified"}.
        Mean local request latency:{" "}
        <span className="mono">
          {metric(paired.uncached.summary.mean_total_latency_ms, 1)} ms
        </span>{" "}
        uncached;{" "}
        <span className="mono">
          {metric(paired.cached.summary.mean_total_latency_ms, 1)} ms
        </span>{" "}
        cached.
      </p>
      <TimingFigure
        label="Mean local token selection gaps by output position: uncached versus cached"
        series={[
          {
            name: "Uncached · local selection",
            values: meanByPosition(
              paired.uncached.runs.map((r) => r.inter_token_latencies_ms),
            ),
          },
          {
            name: "Cached · local selection",
            values: meanByPosition(
              paired.cached.runs.map((r) => r.inter_token_latencies_ms),
            ),
          },
        ]}
      />
      <p className="caption">
        Mean at each position across measured runs; straight segments connect
        observed samples, not smoothing. Same loaded model/prompt, four PyTorch
        threads, alternating paired order. Loading, warmups and final text
        decode excluded. Caching reduces repeated projection work but still
        reads growing K/V; flat latency is not guaranteed.
      </p>
      <TimingFigure
        label="Mean uncached HTTP SSE arrival gaps by output position"
        series={[
          {
            name: "Uncached · real HTTP arrival",
            values: meanByPosition(
              data.network_runs.map((r) => r.token_gaps_ms),
            ),
          },
        ]}
      />
      <p className="caption">
        Separate serial HTTP/SSE runs. These are client arrival gaps, not cached
        CLI timings or pure kernel time. Snapshots are read after completion,
        not polled during each measured request.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Distribution / ms</th>
              <th className="numeric">Count</th>
              <th className="numeric">Median</th>
              <th className="numeric">p90</th>
              <th className="numeric">p95</th>
              <th className="numeric">Max</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Worker steps / all", data.worker_step_ms],
              ["Worker steps / decode only", data.decode_worker_step_ms],
              ["HTTP token gaps", data.network_token_gap_ms],
            ].map(([name, raw]) => {
              const d = raw as Distribution;
              return (
                <tr key={String(name)}>
                  <td>{String(name)}</td>
                  <td className="numeric">{d.count}</td>
                  {[d.median, d.p90, d.p95, d.max].map((v, i) => (
                    <td key={i} className="numeric">
                      {metric(v, 2)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Worker step = dispatch + forward + observation overhead. Pooled
        empirical quantiles use linear interpolation, not confidence bounds. One
        machine/model/prompt, not production capacity. Host contention is a
        hypothesis for historical anomalies, not proven causality.
      </p>
      <div className="method-block mono">
        {data.environment.model_name} · cpu · FP32 · CLI threads{" "}
        {data.environment.torch_thread_count}
        <br />
        User-confirmed AC/quiet window · existing HP power plan retained · no
        concurrent tests/builds
        <br />
        Residual background CPU load was observed; this is not an isolated-host
        baseline. HTTP server thread count was not recorded.
        <br />
        Recorded {data.recorded_at_utc}
      </div>
      <p className="source-line">
        <a href={`${import.meta.env.BASE_URL}data/${artifact.name}`}>
          Final raw JSON
        </a>
        <span className="mono">{artifact.source}</span>
        <span className="mono">
          sha256 {artifact.source_sha256.slice(0, 12)}
        </span>
      </p>
    </section>
  );
}
