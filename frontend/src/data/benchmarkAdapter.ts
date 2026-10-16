export interface Artifact {
  name: string;
  source: string;
  source_commit: string;
  source_sha256: string;
  data: unknown;
}
export interface ModeMetrics {
  name: string;
  latency: number | null;
  ttft: number | null;
  tps: number | null;
  positions: number | null;
  weightBytes: number | null;
}
function object(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}
function field(value: unknown, key: string): unknown {
  return object(value)[key];
}
export function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
export function text(value: unknown): string {
  return typeof value === "string" ? value : "—";
}
function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
export function artifactMethod(artifact: Artifact) {
  const env = field(artifact.data, "environment"),
    config = field(artifact.data, "config"),
    server = field(env, "server");
  return {
    model: text(
      field(env, "model_name") ??
        field(env, "target_model") ??
        field(server, "model"),
    ),
    draft: text(field(env, "draft_model")),
    device: text(field(env, "device") ?? field(server, "device")),
    backend: text(field(env, "quantized_engine")),
    runs: number(
      field(config, "measured_runs_per_mode") ??
        field(config, "requests_per_level"),
    ),
    warmups: number(
      field(config, "warmup_runs_per_mode") ??
        field(config, "serial_warmups_per_level"),
    ),
    generated: number(field(config, "max_new_tokens")),
    prompt: text(field(config, "prompt")),
  };
}
function mode(data: unknown, name: string): ModeMetrics {
  const raw = field(data, name),
    summary = field(raw, "summary"),
    first = array(field(raw, "runs"))[0];
  return {
    name,
    latency: number(field(summary, "mean_total_latency_ms")),
    ttft: number(field(summary, "mean_ttft_ms")),
    tps: number(field(summary, "mean_generated_tokens_per_second")),
    positions: number(field(first, "total_input_token_positions_processed")),
    weightBytes: number(field(raw, "weight_payload_bytes")),
  };
}
export function adaptBenchmarks(artifacts: Artifact[]) {
  const get = (prefix: string) => {
    const artifact = artifacts.find((item) => item.name.startsWith(prefix));
    if (!artifact) throw new Error(`Missing committed ${prefix} artifact.`);
    return artifact;
  };
  const cache = get("m7_"),
    quant = get("m15_"),
    spec = get("m16_"),
    serving = get("m13_");
  const uncached = mode(cache.data, "uncached"),
    cached = mode(cache.data, "cached");
  const first = (name: string) =>
    array(field(field(cache.data, name), "runs"))[0];
  const lengths = (name: string) =>
    array(field(first(name), "forward_input_lengths"))
      .map(number)
      .filter((n): n is number => n !== null);
  return {
    cache,
    quant,
    spec,
    serving,
    uncached,
    cached,
    cacheSpeedup: number(
      field(
        field(cache.data, "comparison"),
        "latency_speedup_uncached_over_cached",
      ),
    ),
    lengths: { uncached: lengths("uncached"), cached: lengths("cached") },
    method: {
      device: text(field(field(cache.data, "environment"), "device")),
      model: text(field(field(cache.data, "environment"), "model_name")),
      promptTokens: number(field(first("uncached"), "prompt_token_count")),
      generatedTokens: number(
        field(first("uncached"), "generated_token_count"),
      ),
      warmups: number(
        field(field(cache.data, "config"), "warmup_runs_per_mode"),
      ),
      runs: number(
        field(field(cache.data, "config"), "measured_runs_per_mode"),
      ),
      threads: number(
        field(field(cache.data, "environment"), "torch_thread_count"),
      ),
    },
    quantModes: [mode(quant.data, "fp32"), mode(quant.data, "int8")],
    quantParity: field(
      field(quant.data, "comparison"),
      "exact_token_id_parity",
    ),
    specModes: [mode(spec.data, "ordinary"), mode(spec.data, "speculative")],
    specCounters: {
      accepted: number(
        field(
          array(field(field(spec.data, "speculative"), "runs"))[0],
          "accepted_draft_tokens",
        ),
      ),
      proposed: number(
        field(
          array(field(field(spec.data, "speculative"), "runs"))[0],
          "proposed_draft_tokens",
        ),
      ),
      calls: number(
        field(
          array(field(field(spec.data, "speculative"), "runs"))[0],
          "target_verification_calls",
        ),
      ),
    },
    servingRows: array(field(serving.data, "levels")).map((level) => {
      const summary = field(level, "summary");
      return {
        concurrency: number(field(level, "concurrency")),
        requests: number(field(summary, "successful_requests")),
        rps: number(field(summary, "requests_per_second")),
        tps: number(field(summary, "generated_tokens_per_second")),
        latency: number(field(summary, "mean_latency_ms")),
        p50: number(field(summary, "p50_latency_ms")),
        p95: number(field(summary, "p95_latency_ms")),
        p99: number(field(summary, "p99_latency_ms")),
      };
    }),
  };
}
export async function loadBenchmarks(
  signal?: AbortSignal,
): Promise<Artifact[]> {
  const response = await fetch(
    `${import.meta.env.BASE_URL}data/benchmarks.json`,
    { signal },
  );
  if (!response.ok)
    throw new Error(
      "Benchmark mirror is unavailable. Run npm run dev or npm run build to derive source data.",
    );
  const value = await response.json();
  if (!Array.isArray(value.artifacts))
    throw new Error("Invalid benchmark mirror.");
  return value.artifacts;
}
