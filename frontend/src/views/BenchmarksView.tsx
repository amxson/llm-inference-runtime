import { useEffect, useState } from "react";
import {
  adaptBenchmarks,
  artifactMethod,
  loadBenchmarks,
  type Artifact,
  type ModeMetrics,
} from "../data/benchmarkAdapter";
import { metric } from "../runtime/traceAdapter";
import { FinalMeasurement } from "./FinalMeasurement";
import { CacheLengthSweep } from "./CacheLengthSweep";
import { CachedContinuousBenchmark } from "./CachedContinuousBenchmark";

function Source({ artifact }: { artifact: Artifact }) {
  return (
    <>
      {!artifact.name.startsWith("m7_") && <Method artifact={artifact} />}
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
          className="mono"
          href={`https://github.com/amxson/llm-inference-runtime/blob/${artifact.source_commit}/${artifact.source}`}
        >
          commit {artifact.source_commit.slice(0, 7)}
        </a>
      </p>
    </>
  );
}
function Method({ artifact }: { artifact: Artifact }) {
  const method = artifactMethod(artifact);
  return (
    <div className="method-block mono">
      device {method.device} · model {method.model}
      {method.draft !== "—" ? ` · draft ${method.draft}` : ""}
      {method.backend !== "—" ? ` · backend ${method.backend}` : ""}
      <br />
      prompt “{method.prompt}” · generated {metric(method.generated)} tok ·
      excluded warmups {metric(method.warmups)}/mode or level · measured
      requests {metric(method.runs)}/mode or level
    </div>
  );
}
export function PositionFigure({
  uncached,
  cached,
}: {
  uncached: number[];
  cached: number[];
}) {
  const max = Math.max(1, ...uncached),
    cell = 14,
    gap = 3,
    offset = max * (cell + gap) + 64;
  const panels = [
    { name: "Uncached", lengths: uncached, x: 0, cached: false },
    { name: "KV-cached", lengths: cached, x: offset, cached: true },
  ];
  return (
    <svg
      className="position-figure"
      viewBox={`0 0 ${offset * 2} ${uncached.length * 22 + 56}`}
      role="img"
      aria-label="Forwarded positions: uncached staircase versus KV-cached prefill and single-position decoding"
    >
      <defs>
        <pattern
          id="kv-hatch"
          width="5"
          height="5"
          patternUnits="userSpaceOnUse"
        >
          <path
            d="M-1 1L1-1M0 5L5 0M4 6L6 4"
            stroke="var(--hatch-line)"
            strokeWidth="1"
          />
        </pattern>
      </defs>
      {panels.map((panel) => (
        <g
          key={panel.name}
          aria-label={`${panel.name}: ${panel.lengths.reduce((sum, n) => sum + n, 0)} forwarded positions`}
        >
          <text x={panel.x} y={16} className="svg-label">
            {panel.name}
          </text>
          <text
            x={panel.x + offset - 40}
            y={16}
            textAnchor="end"
            className="mono svg-number"
          >
            {panel.lengths.reduce((sum, n) => sum + n, 0)} pos
          </text>
          {panel.lengths.map((forwarded, row) => {
            const context = uncached[row] ?? forwarded;
            return (
              <g key={row}>
                <text x={panel.x} y={row * 22 + 48} className="mono svg-axis">
                  {row === 0 ? "P" : row}
                </text>
                {Array.from({ length: context }, (_, col) => {
                  const reused = panel.cached && col < context - forwarded;
                  return (
                    <rect
                      key={col}
                      data-kind={reused ? "reused" : "forwarded"}
                      x={panel.x + 24 + col * (cell + gap)}
                      y={row * 22 + 36}
                      width={cell}
                      height={cell}
                      rx="1"
                      fill={reused ? "url(#kv-hatch)" : "var(--data-computed)"}
                      stroke={reused ? "var(--border-strong)" : "none"}
                    >
                      <title>
                        {panel.name} step {row}: position {col},{" "}
                        {reused
                          ? "read from cached prefix"
                          : "forwarded through model"}
                      </title>
                    </rect>
                  );
                })}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}
function ResultsTable({
  rows,
  weights = false,
}: {
  rows: ModeMetrics[];
  weights?: boolean;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            {weights && <th className="numeric">Weight payload / bytes</th>}
            <th className="numeric">Mean latency / ms</th>
            <th className="numeric">TTFT / ms</th>
            <th className="numeric">Tokens/sec</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <td>{row.name}</td>
              {weights && (
                <td className="numeric">{metric(row.weightBytes)}</td>
              )}
              <td
                className={`numeric ${row.latency === Math.min(...rows.map((item) => item.latency ?? Infinity)) ? "best" : ""}`}
              >
                {metric(row.latency, 3)}
              </td>
              <td className="numeric">{metric(row.ttft, 3)}</td>
              <td className="numeric">{metric(row.tps, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function BenchmarksView() {
  const [artifacts, setArtifacts] = useState<Artifact[] | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    loadBenchmarks(controller.signal)
      .then(setArtifacts)
      .catch((error) => {
        if (!controller.signal.aborted) setError(String(error));
      });
    return () => controller.abort();
  }, []);
  if (error)
    return (
      <>
        <h1>Benchmarks</h1>
        <p role="alert" className="error-text">
          {error}
        </p>
      </>
    );
  if (!artifacts)
    return (
      <>
        <h1>Benchmarks</h1>
        <p role="status">Reading committed artifacts…</p>
      </>
    );
  let data: ReturnType<typeof adaptBenchmarks>;
  try {
    data = adaptBenchmarks(artifacts);
  } catch (error) {
    return <p role="alert">{String(error)}</p>;
  }
  const maxLatency = Math.max(
    data.uncached.latency ?? 0,
    data.cached.latency ?? 0,
    1,
  );
  const positionRatio = data.cached.positions
    ? (data.uncached.positions ?? 0) / data.cached.positions
    : null;
  const ordinary = data.specModes[0].latency,
    speculative = data.specModes[1].latency;
  const slower =
    ordinary && speculative != null
      ? ((speculative - ordinary) / ordinary) * 100
      : null;
  return (
    <>
      <div className="page-heading">
        <p className="eyebrow">Measurements / committed CPU experiments</p>
        <h1>Less repeated work. Measured, not assumed.</h1>
        <p>
          Build-time mirrors of committed JSON—not live benchmark measurements.
          Each figure links to its source and methodology.
        </p>
      </div>
      {artifacts.find((a) => a.name === "cache_length_sweep_cpu.json") ? (
        <CacheLengthSweep
          artifact={
            artifacts.find((a) => a.name === "cache_length_sweep_cpu.json")!
          }
        />
      ) : (
        <p role="alert">
          Controlled cache-length sweep artifact is unavailable.
        </p>
      )}
      {artifacts.find(
        (a) => a.name === "cached_continuous_batching_cpu.json",
      ) && (
        <CachedContinuousBenchmark
          artifact={
            artifacts.find(
              (a) => a.name === "cached_continuous_batching_cpu.json",
            )!
          }
        />
      )}
      {artifacts.find((a) => a.name === "final_cpu.json") && (
        <details className="result-details">
          <summary>Earlier 128-token rerun / CLI and network timing</summary>
          <FinalMeasurement
            artifact={artifacts.find((a) => a.name === "final_cpu.json")!}
          />
        </details>
      )}
      <h2>Historical experiments / preserved evidence</h2>
      <p className="caption caveat">
        The following M5–M17 artifacts are historical, not preferred headline
        timing. M7 and the M13 concurrency-one anomaly may be affected by host
        contention; this is a hypothesis, not a proven diagnosis. The length
        sweep above has different token budgets and is not a causal before/after
        machine comparison.
      </p>
      <p className="finding">
        KV caching forwards{" "}
        <span className="mono">{metric(data.cached.positions)}</span> positions
        instead of{" "}
        <span className="mono">{metric(data.uncached.positions)}</span>. The
        recorded uncached/cached mean latency ratio is{" "}
        <span className="mono accent-ratio">
          {metric(data.cacheSpeedup, 3)}×
        </span>
        .
      </p>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Fig. 1 / Positions forwarded</h2>
          <span className="caption">
            One solid cell = one forwarded input position · not FLOPs
          </span>
        </div>
        <PositionFigure
          uncached={data.lengths.uncached}
          cached={data.lengths.cached}
        />
        <p className="caption">
          P = prompt prefill. Solid cells were forwarded; hatched cells denote
          the logical prefix read from KV state, not additional forwarded
          positions. New output tokens are not counted as input until the
          following forward.
        </p>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Fig. 2 / Mean request latency</h2>
          <span className="caption">
            Historical M7 paired modes · milliseconds
          </span>
        </div>
        <svg
          className="latency-figure"
          viewBox="0 0 820 112"
          role="img"
          aria-label="Mean request latency comparison"
        >
          {[data.uncached, data.cached].map((row, i) => (
            <g key={row.name}>
              <text x="0" y={i * 48 + 24} className="svg-label">
                {row.name}
              </text>
              {row.latency != null && (
                <rect
                  x="120"
                  y={i * 48 + 8}
                  width={(row.latency / maxLatency) * 540}
                  height="24"
                  fill={i === 1 ? "var(--accent)" : "var(--data-computed)"}
                />
              )}
              <text
                x="800"
                y={i * 48 + 25}
                textAnchor="end"
                className="mono svg-number"
              >
                {metric(row.latency, 3)} ms
              </text>
            </g>
          ))}
        </svg>
        <p className="caption">
          {metric(positionRatio, 2)}× fewer forwarded positions does not imply
          the same latency ratio: cached steps still attend over prior K/V, and
          fixed execution/Python overhead remains.
        </p>
        <div className="method-block mono">
          device {data.method.device} · model {data.method.model} · dtype — ·
          threads {metric(data.method.threads)}
          <br />
          prompt {metric(data.method.promptTokens)} tok · generated{" "}
          {metric(data.method.generatedTokens)} tok · warmup{" "}
          {metric(data.method.warmups)}/mode · runs {metric(data.method.runs)}
          /mode
          <br />
          statistic: mean · alternating paired order · load/final text decode
          excluded · exact token parity
        </div>
        <Source artifact={data.cache} />
        <ResultsTable rows={[data.uncached, data.cached]} />
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Exploratory serving / concurrency result</h2>
          <span className="caption">M13 · actual network HTTP/SSE</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th className="numeric">Concurrency</th>
                <th className="numeric">Successful</th>
                <th className="numeric">Requests/sec</th>
                <th className="numeric">Tokens/sec</th>
                <th className="numeric">Mean / ms</th>
                <th className="numeric">p50 / ms</th>
                <th className="numeric">p95 / ms</th>
                <th className="numeric">p99 / ms</th>
              </tr>
            </thead>
            <tbody>
              {data.servingRows.map((row) => (
                <tr key={row.concurrency}>
                  <td className="numeric">{metric(row.concurrency)}</td>
                  <td className="numeric">{metric(row.requests)}</td>
                  <td className="numeric">{metric(row.rps, 3)}</td>
                  <td className="numeric">{metric(row.tps, 3)}</td>
                  <td className="numeric">{metric(row.latency, 3)}</td>
                  <td className="numeric">{metric(row.p50, 3)}</td>
                  <td className="numeric">{metric(row.p95, 3)}</td>
                  <td className="numeric">{metric(row.p99, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption caveat">
          The concurrency-one phase was anomalously slow; the cause is not
          established. A small sequential sweep cannot attribute the whole
          difference to batching. Throughput is not per-client latency; these
          are not capacity or tail-SLA claims.
        </p>
        <Source artifact={data.serving} />
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>INT8 / smaller weight payload, changed output</h2>
          <span className="caption">M15 · attention/MLP projections only</span>
        </div>
        <p className="caveat warning-note">
          Generated output changed. This is not an equivalent-output
          optimization or a quality evaluation.
        </p>
        <ResultsTable rows={data.quantModes} weights />
        <p className="caption">
          Unique parameter/unpacked weight payload—not process RSS. Embeddings,
          layer norms and tied output head stay FP32. Paired order, warmups and
          workload are recorded below. Setup is excluded.
        </p>
        <Source artifact={data.quant} />
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Speculation / {metric(slower, 2)}% slower in this workload</h2>
          <span className="caption">M16 · greedy draft/verify · uncached</span>
        </div>
        <p className="caveat">
          Accepted{" "}
          <span className="mono">
            {metric(data.specCounters.accepted)}/
            {metric(data.specCounters.proposed)}
          </span>{" "}
          draft proposals;{" "}
          <span className="mono">{metric(data.specCounters.calls)}</span> target
          verifications per request. Exact target-greedy output parity, but no
          saved target calls.
        </p>
        <ResultsTable rows={data.specModes} />
        <p className="caption">
          Target/draft identifiers and paired workload are recorded below. Draft
          overhead is included; loading/compatibility checks are excluded.
          Compatibility does not guarantee useful acceptance.
        </p>
        <Source artifact={data.spec} />
      </section>
      <section className="instrument-section">
        <h2>Artifact register</h2>
        <p className="caption">
          Experiments are separate runs, not a cross-milestone leaderboard.
          Historical M5 is not M7's denominator. M17 aggregates measurements; it
          does not create new ones.
        </p>
        <table>
          <thead>
            <tr>
              <th>Committed artifact</th>
              <th>Provenance</th>
              <th>Data</th>
            </tr>
          </thead>
          <tbody>
            {artifacts.map((artifact) => (
              <tr key={artifact.name}>
                <td className="mono">{artifact.name}</td>
                <td className="mono">{artifact.source_commit.slice(0, 7)}</td>
                <td>
                  <a
                    href={`${import.meta.env.BASE_URL}data/${artifact.name}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Raw JSON
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
