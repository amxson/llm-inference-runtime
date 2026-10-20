import { readFileSync } from "node:fs";
import { adaptBenchmarks, type Artifact } from "../data/benchmarkAdapter";
import fixture from "../data/recorded-trace.json";
import {
  metric,
  positionWork,
  requestIds,
  stateAt,
  parseTraceFrames,
} from "../runtime/traceAdapter";
import { stepIndex, replayInterval } from "../runtime/replay";
const artifacts: Artifact[] = JSON.parse(
  readFileSync("public/data/benchmarks.json", "utf8"),
).artifacts;
test("benchmarks are adapted from committed source measurements, not typed-in UI values", () => {
  const data = adaptBenchmarks(artifacts);
  expect(data.uncached.positions).toBe(95);
  expect(data.cached.positions).toBe(14);
  expect(data.lengths.cached.reduce((a, b) => a + b, 0)).toBe(
    data.cached.positions,
  );
  expect(data.quantParity).toBe(false);
  expect(data.specCounters.accepted).toBe(0);
  expect(data.servingRows.map((row) => row.concurrency)).toEqual([1, 2, 4]);
});
test("missing metrics render as em dashes, never invented zeroes", () => {
  expect(metric(null)).toBe("—");
  expect(metric(undefined)).toBe("—");
  expect(metric(NaN)).toBe("—");
  expect(metric(0)).toBe("0");
  const data = adaptBenchmarks(
    artifacts.map((item) =>
      item.name.startsWith("m7_") ? { ...item, data: {} } : item,
    ),
  );
  expect(data.cached.latency).toBeNull();
  expect(metric(data.cached.latency)).toBe("—");
});
test("fixture encodes actual admission, per-row work, completion and compact cache reuse", () => {
  const frames = parseTraceFrames(fixture.traces);
  expect(requestIds(frames)).toHaveLength(6);
  expect(stateAt(frames, 0, "r4")).toBe("waiting");
  expect(stateAt(frames, 0, "r1")).toBe("prefill");
  expect(stateAt(frames, 1, "r1")).toBe("decoding");
  expect(stateAt(frames, 2, "r1")).toBe("finished");
  const work = positionWork(frames[0]);
  expect(work.decode).toBe(0);
  expect(work.prefill).toBe(frames[0].events[0].model_input_width * 3);
  expect(fixture.serving_cache).toBe(true);
  expect(frames[1].decode_input_shape).toEqual([3, 1]);
  expect(frames[1].logical_cache_lengths_before?.r1).toBe(
    fixture.requests[0].prompt_token_ids.length,
  );
  expect(frames[1].logical_cache_lengths_after?.r1).toBe(
    fixture.requests[0].prompt_token_ids.length + 1,
  );
  expect(
    frames.some(
      (f) => f.admitted.length > 0 && (f.decode_request_ids?.length ?? 0) > 0,
    ),
  ).toBe(true);
  expect(frames.at(-1)).not.toHaveProperty("logical_kv_before_forward");
  expect(fixture).not.toHaveProperty("observed_at_ms");
});
test("trace adapter rejects unsupported phases and incomplete events", () => {
  expect(() => parseTraceFrames([{ tick: 1 }])).toThrow("Invalid trace");
  const frames = structuredClone(fixture.traces);
  frames[0].events[0].phase = "preempted";
  expect(() => parseTraceFrames(frames)).toThrow("Invalid trace event");
});
test("replay stepping clamps, and speed is a presentation cadence", () => {
  expect(stepIndex(0, -1, 8)).toBe(0);
  expect(stepIndex(7, 1, 8)).toBe(7);
  expect(stepIndex(2, 1, 8)).toBe(3);
  expect(stepIndex(0, 1, 0)).toBe(0);
  expect(replayInterval(0.25)).toBe(4000);
});
