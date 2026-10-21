import type { Artifact } from "./benchmarkAdapter";

export type GpuSpread = { median: number; p95: number; p90: number; max: number };
export type GpuEngine = {
  measured_repetitions: number;
  successful_requests: number;
  failed_requests: number;
  request_latency_ms: GpuSpread;
  ttft_ms: GpuSpread;
  mean_itl_ms: GpuSpread;
  cohort_wall_time_ms: GpuSpread;
  generated_token_throughput: GpuSpread;
  request_throughput: GpuSpread;
  torch_peak_allocated_bytes: number | null;
  torch_peak_reserved_bytes: number | null;
};
export type GpuComparison = {
  schema_version: number;
  status: string;
  final_eligible: boolean;
  model: string;
  revision: string;
  dtype: string;
  execution_policy: string;
  source_commit: string;
  started_at_utc: string;
  finished_at_utc: string;
  hardware: {
    cuda_available: boolean;
    cuda_runtime: string;
    python: string;
    versions: Record<string, string | null>;
    gpus: { name: string; total_vram_bytes: number; compute_capability: number[] }[];
  };
  correctness_evidence: { status: string };
  workload: { id: string; prompt: string; max_new_tokens: number }[];
  concurrencies: number[];
  measured_cohorts?: number;
  excluded_warmup_cohorts?: number;
  sessions: { engine: string; environment: { versions?: Record<string, string | null> } }[];
  aggregates: {
    concurrency: number;
    runtime: GpuEngine;
    vllm: GpuEngine;
    comparison: {
      vllm_over_runtime_token_throughput: number;
      runtime_over_vllm_latency: number;
      exact_output_parity_rate: number;
    };
  }[];
  limitations: string[];
};

export function adaptGpuComparison(artifact: Artifact): GpuComparison {
  const data = artifact.data as GpuComparison;
  if (
    data?.schema_version !== 1 || data.status !== "pass" || data.final_eligible !== true ||
    data.correctness_evidence?.status !== "pass" || data.hardware?.cuda_available !== true ||
    data.model !== "Qwen/Qwen2.5-0.5B" || data.dtype !== "float32" ||
    data.execution_policy !== "fp32-math" || !/^[a-f0-9]{40}$/.test(data.source_commit) ||
    !/^[a-f0-9]{40}$/.test(data.revision) || !data.hardware.gpus?.length ||
    !Array.isArray(data.workload) || data.workload.length < 16 ||
    ![32, 64, 128].every((budget) => data.workload.some((r) => r.max_new_tokens === budget)) ||
    JSON.stringify(data.concurrencies) !== "[1,2,4]" || data.aggregates?.length !== 3 ||
    data.sessions?.map((s) => s.engine).join(",") !== "runtime,vllm,vllm,runtime" ||
    !Array.isArray(data.limitations)
  ) throw new Error("GPU comparison unavailable: a passing final FP32 CUDA artifact is required.");
  for (const [index, row] of data.aggregates.entries()) {
    if (row.concurrency !== data.concurrencies[index]) throw new Error("GPU concurrency evidence mismatch.");
    for (const engine of [row.runtime, row.vllm]) {
      if (!engine || !Number.isInteger(engine.measured_repetitions) || engine.measured_repetitions < 7 || engine.failed_requests !== 0 ||
          engine.successful_requests !== data.workload.length * engine.measured_repetitions ||
          ![engine.request_latency_ms?.median, engine.request_latency_ms?.p95, engine.ttft_ms?.median,
            engine.generated_token_throughput?.median, engine.request_throughput?.median]
            .every((n) => typeof n === "number" && Number.isFinite(n) && n > 0)) {
        throw new Error("GPU comparison has missing, failed or insufficient measurements.");
      }
    }
    if (row.runtime.measured_repetitions !== row.vllm.measured_repetitions ||
        ![row.comparison?.vllm_over_runtime_token_throughput, row.comparison?.runtime_over_vllm_latency]
          .every((n) => typeof n === "number" && Number.isFinite(n) && n > 0) ||
        !(row.comparison.exact_output_parity_rate >= 0 && row.comparison.exact_output_parity_rate <= 1)) {
      throw new Error("GPU comparison has invalid ratios or output parity evidence.");
    }
  }
  return data;
}
