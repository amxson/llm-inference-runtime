import { render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { CacheLengthSweep, type SweepData } from "../views/CacheLengthSweep";
import { BenchmarksView } from "../views/BenchmarksView";
import type { Artifact } from "../data/benchmarkAdapter";
import { metric } from "../runtime/traceAdapter";
import { vi } from "vitest";

function artifact(): Artifact {
  return JSON.parse(
    readFileSync("public/data/benchmarks.json", "utf8"),
  ).artifacts.find((a: Artifact) => a.name === "cache_length_sweep_cpu.json");
}
afterEach(() => vi.unstubAllGlobals());

test("sweep table uses actual JSON medians, pair ratios and separate work counts", () => {
  const source = artifact();
  const data = source.data as SweepData;
  render(<CacheLengthSweep artifact={source} />);
  const region = screen.getByRole("region", { name: /work and latency/ });
  const rows = within(region).getAllByRole("row").slice(2);
  expect(rows).toHaveLength(3);
  data.levels.forEach((level, index) => {
    const cells = within(rows[index]).getAllByRole("cell");
    expect(cells.map((cell) => cell.textContent)).toEqual([
      `${level.requested_tokens} / ${level.observed_generated_tokens.join(", ")}`,
      metric(level.work_positions?.uncached_positions),
      metric(level.work_positions?.cached_positions),
      `${metric(level.work_positions?.position_ratio_uncached_over_cached, 2)}×`,
      metric(level.aggregates.uncached.total_latency_ms.median, 2),
      metric(level.aggregates.cached.total_latency_ms.median, 2),
      `${metric(level.paired_latency_ratios.summary.median, 2)}×`,
    ]);
  });
  expect(
    screen.getByText(/not FLOPs or a latency prediction/),
  ).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Sweep raw JSON" })).toHaveAttribute(
    "href",
    "/data/cache_length_sweep_cpu.json",
  );
  expect(screen.getByText(/Timing spread/)).toBeInTheDocument();
});

test("parity failure suppresses the entire sweep ratio table", () => {
  const source = artifact();
  (source.data as SweepData).levels[0].exact_token_id_parity = false;
  render(<CacheLengthSweep artifact={source} />);
  expect(screen.getByRole("alert")).toHaveTextContent(
    "exact token parity is required",
  );
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});

test("benchmark page leads with the sweep and retains historical anomalies", async () => {
  const bundle = JSON.parse(
    readFileSync("public/data/benchmarks.json", "utf8"),
  );
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, json: async () => bundle }),
  );
  render(<BenchmarksView />);
  await screen.findByRole("heading", {
    name: "KV cache / generation-length sweep",
  });
  const headings = screen.getAllByRole("heading", { level: 2 });
  expect(headings[0]).toHaveTextContent("KV cache / generation-length sweep");
  expect(
    screen.getByRole("heading", {
      name: "Exploratory serving / concurrency result",
    }),
  ).toBeInTheDocument();
  expect(
    screen.getByText(/concurrency-one phase was anomalously slow/),
  ).toBeInTheDocument();
  expect(screen.getByText("2,730.650")).toBeInTheDocument();
});
