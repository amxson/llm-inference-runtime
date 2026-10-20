import { fireEvent, render, screen } from "@testing-library/react";
import type { Snapshot } from "../api/types";
import { PlaygroundView } from "../views/PlaygroundView";
import { RuntimeView } from "../views/RuntimeView";
import { CachedContinuousBenchmark } from "../views/CachedContinuousBenchmark";
import type { ContinuousData } from "../views/CachedContinuousBenchmark";
import type { Artifact } from "../data/benchmarkAdapter";
import { parseTraceFrames } from "../runtime/traceAdapter";
import fixture from "../data/recorded-trace.json";
import { readFileSync } from "node:fs";

test("serving cache status follows the responding backend including unavailable", () => {
  const { rerender } = render(<PlaygroundView snapshot={null} />);
  expect(screen.getByText(/serving cache: unknown/)).toBeInTheDocument();
  rerender(
    <PlaygroundView
      snapshot={
        { engine: "continuous_cached_greedy", serving_cache: true } as Snapshot
      }
    />,
  );
  expect(screen.getByText(/serving cache: on/)).toBeInTheDocument();
  rerender(
    <PlaygroundView
      snapshot={
        {
          engine: "continuous_uncached_greedy",
          serving_cache: false,
        } as Snapshot
      }
    />,
  );
  expect(screen.getByText(/serving cache: off/)).toBeInTheDocument();
});

test("cached replay exposes actual phase groups, one-token decode and selection without a physical block pool", () => {
  const frame = vi.fn();
  render(
    <RuntimeView
      snapshot={null}
      mode="replay"
      onMode={() => {}}
      onFrame={frame}
    />,
  );
  expect(screen.getAllByText(/CACHED DECODE/).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/PREFILL/).length).toBeGreaterThan(0);
  expect(
    screen.queryByRole("heading", { name: "Logical KV block allocator" }),
  ).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Inspect cache r4" }));
  expect(
    screen.getByRole("button", { name: "Select request r4" }),
  ).toHaveAttribute("aria-pressed", "true");
  fireEvent.click(screen.getByRole("button", { name: "Next iteration" }));
  expect(frame).toHaveBeenCalledWith(4);
});

test("cached trace validation rejects padding counted as history and full-context decode", () => {
  const frames = structuredClone(fixture.traces);
  frames[1].decode_input_shape![1] = 5;
  expect(() => parseTraceFrames(frames)).toThrow("cached batch shape");
  const history = structuredClone(fixture.traces);
  history[1].events[0].logical_cache_length_after += 1;
  expect(() => parseTraceFrames(history)).toThrow("logical cache length");
});

test("continuous benchmark keeps slower ratios visible and refuses invalid parity", () => {
  const spread = (median: number) => ({
    median,
    min: median,
    max: median,
    p25: median,
    p75: median,
    p95: median,
  });
  const metrics = {
    wall_ms: spread(100),
    generated_tokens_per_second: spread(10),
    total_input_positions: spread(99),
    model_forward_calls: spread(5),
  };
  const data: ContinuousData = {
    schema_version: 1,
    exact_token_id_parity: true,
    config: {
      measured_pairs_per_level: 10,
      warmup_cohorts_per_mode_per_level: 2,
      workload: [{ prompt: "unit fixture", max_new_tokens: 2 }],
    },
    levels: [
      {
        max_active: 2,
        exact_token_id_parity: true,
        aggregates: {
          uncached: metrics,
          cached: { ...metrics, wall_ms: spread(200) },
        },
        paired_wall_time_ratios: { summary: spread(0.5) },
      },
    ],
  };
  const artifact = {
    name: "cached_continuous_batching_cpu.json",
    source: "benchmarks/results/cached_continuous_batching_cpu.json",
    source_commit: "test",
    source_sha256: "test",
    data,
  } as Artifact;
  const { rerender } = render(
    <CachedContinuousBenchmark artifact={artifact} />,
  );
  expect(screen.getByText("0.50×")).toBeInTheDocument();
  expect(screen.getByText(/below 1 means slower/)).toBeInTheDocument();
  data.exact_token_id_parity = false;
  rerender(<CachedContinuousBenchmark artifact={artifact} />);
  expect(screen.getByRole("alert")).toHaveTextContent("exact token parity");
});

test("continuous benchmark displays the committed raw comparison below the sweep data", () => {
  const bundle = JSON.parse(
    readFileSync("public/data/benchmarks.json", "utf8"),
  );
  const artifact = bundle.artifacts.find(
    (a: Artifact) => a.name === "cached_continuous_batching_cpu.json",
  ) as Artifact;
  expect(artifact.source_commit).toMatch(/^[a-f0-9]{40}$/);
  const data = artifact.data as ContinuousData;
  expect(data.levels.map((l) => l.max_active)).toEqual([1, 2, 4]);
  render(<CachedContinuousBenchmark artifact={artifact} />);
  expect(screen.getByRole("table")).toBeInTheDocument();
  expect(screen.getByText(/Host power telemetry changed/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Raw JSON" })).toHaveAttribute(
    "href",
    "/data/cached_continuous_batching_cpu.json",
  );
  for (const level of data.levels)
    expect(
      screen.getByText(
        `${level.paired_wall_time_ratios.summary.median.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}×`,
      ),
    ).toBeInTheDocument();
});
