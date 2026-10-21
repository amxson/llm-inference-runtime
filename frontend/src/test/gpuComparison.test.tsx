import { render, screen } from "@testing-library/react";
import type { Artifact } from "../data/benchmarkAdapter";
import { adaptGpuComparison, type GpuComparison } from "../data/gpuComparisonAdapter";
import { GpuComparison as GpuView } from "../views/GpuComparison";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

test("published GPU summary derives from the unchanged full artifact", () => {
  const rawText = readFileSync("../benchmarks/results/gpu_final_comparison.json", "utf8");
  const raw = JSON.parse(rawText);
  const bundle = JSON.parse(readFileSync("public/data/benchmarks.json", "utf8"));
  const artifact = bundle.artifacts.find((a: Artifact) => a.name === "gpu_final_comparison.json") as Artifact;
  const summary = adaptGpuComparison(artifact);
  expect(artifact.source_sha256).toBe(createHash("sha256").update(rawText.replace(/\r\n/g, "\n")).digest("hex"));
  expect(summary.source_commit).toBe(raw.source_commit);
  expect(summary.measured_cohorts).toBe(60);
  expect(summary.excluded_warmup_cohorts).toBe(24);
  expect(summary).not.toHaveProperty("raw_repetitions");
  expect(summary).not.toHaveProperty("warmups");
  summary.aggregates.forEach((row, index) => {
    expect(row.runtime).toEqual(raw.aggregates[index].runtime);
    expect(row.vllm).toEqual(raw.aggregates[index].vllm);
    expect(row.comparison.vllm_over_runtime_token_throughput).toBe(raw.aggregates[index].comparison.vllm_over_runtime_token_throughput);
    expect(row.comparison).not.toHaveProperty("outputs");
  });
  render(<GpuView artifact={artifact} />);
  expect(screen.getByRole("heading", { name: "Same model, same GPU, different engine design." })).toBeInTheDocument();
});

// Synthetic presentation fixture only: never exported as benchmark evidence.
function fixture(): Artifact {
  const spread = (median: number) => ({ median, p95: median * 2, p90: median * 1.5, max: median * 3 });
  const engine = {
    measured_repetitions: 10, successful_requests: 240, failed_requests: 0,
    request_latency_ms: spread(100), ttft_ms: spread(10), mean_itl_ms: spread(2),
    cohort_wall_time_ms: spread(1000), generated_token_throughput: spread(20),
    request_throughput: spread(3), torch_peak_allocated_bytes: 2 ** 30,
    torch_peak_reserved_bytes: 2 ** 31,
  };
  const data: GpuComparison = {
    schema_version: 1, status: "pass", final_eligible: true,
    model: "Qwen/Qwen2.5-0.5B", revision: "b".repeat(40), dtype: "float32",
    execution_policy: "fp32-math", source_commit: "a".repeat(40),
    started_at_utc: "test-only", finished_at_utc: "test-only",
    hardware: { cuda_available: true, cuda_runtime: "test-only", python: "test-only",
      versions: { torch: "test-only" }, gpus: [{ name: "Synthetic GPU", total_vram_bytes: 1, compute_capability: [7, 5] }] },
    correctness_evidence: { status: "pass" },
    workload: Array.from({ length: 24 }, (_, i) => ({ id: `unit-${i}`, prompt: "Synthetic prompt", max_new_tokens: [32, 64, 128][i % 3] })),
    concurrencies: [1, 2, 4],
    sessions: ["runtime", "vllm", "vllm", "runtime"].map((engine) => ({ engine, environment: { versions: { vllm: "test-only" } } })),
    aggregates: [1, 2, 4].map((concurrency) => ({ concurrency, runtime: { ...engine }, vllm: { ...engine, torch_peak_allocated_bytes: null, torch_peak_reserved_bytes: null },
      comparison: { vllm_over_runtime_token_throughput: 2, runtime_over_vllm_latency: 3, exact_output_parity_rate: 0.5 } })),
    limitations: ["Synthetic fixture, not real GPU evidence."],
  };
  return { name: "gpu_final_comparison.json", source: "benchmarks/results/gpu_final_comparison.json",
    source_commit: "c".repeat(40), source_sha256: "d".repeat(64), data };
}

test("GPU view retains unfavorable ratios, precision caveats and separate public CPU context", () => {
  render(<GpuView artifact={fixture()} />);
  expect(screen.getAllByText("2.00×")).toHaveLength(3);
  expect(screen.getAllByText("3.00×")).toHaveLength(3);
  expect(screen.getAllByText("50.0%")).toHaveLength(3);
  expect(screen.getByText(/FP32 was explicitly approved/)).toBeInTheDocument();
  expect(screen.getByText(/public Playground remains distilgpt2 on CPU/)).toBeInTheDocument();
  expect(screen.getByText(/not physical VRAM or vLLM memory/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Raw GPU JSON" })).toHaveAttribute("href", "/data/gpu_final_comparison.json");
  expect(screen.getByRole("link", { name: /publication/ })).toHaveAttribute("href", expect.stringContaining("c".repeat(40)));
});

test.each(["smoke", "failure", "cpu", "wrong-policy", "missing-cohort", "failed-request", "infinite", "insufficient", "bad-parity"])("GPU adapter refuses %s as final evidence", (problem) => {
  const artifact = fixture();
  const data = artifact.data as GpuComparison;
  if (problem === "smoke") data.final_eligible = false;
  if (problem === "failure") data.status = "fail";
  if (problem === "cpu") data.hardware.cuda_available = false;
  if (problem === "wrong-policy") data.execution_policy = "fp16-sdpa";
  if (problem === "missing-cohort") data.aggregates.pop();
  if (problem === "failed-request") data.aggregates[0].runtime.failed_requests = 1;
  if (problem === "infinite") data.aggregates[0].runtime.request_latency_ms.median = Infinity;
  if (problem === "insufficient") data.aggregates[0].runtime.measured_repetitions = 2;
  if (problem === "bad-parity") data.aggregates[0].comparison.exact_output_parity_rate = 2;
  expect(() => adaptGpuComparison(artifact)).toThrow();
  render(<GpuView artifact={artifact} />);
  expect(screen.getByRole("alert")).toBeInTheDocument();
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});
