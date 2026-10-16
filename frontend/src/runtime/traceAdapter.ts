import type { TraceFrame } from "../api/types";
export function parseTraceFrames(value: unknown): TraceFrame[] {
  if (!Array.isArray(value)) throw new Error("Trace must be an array.");
  const record = (item: unknown): Record<string, unknown> => {
    if (typeof item !== "object" || item === null)
      throw new Error("Invalid trace object.");
    return item as Record<string, unknown>;
  };
  const strings = (item: unknown) =>
    Array.isArray(item) && item.every((value) => typeof value === "string");
  for (const raw of value) {
    const frame = record(raw);
    if (
      !Number.isInteger(frame.tick) ||
      ![
        "admitted",
        "active_request_ids",
        "model_request_ids",
        "completed",
      ].every((key) => strings(frame[key])) ||
      !Array.isArray(frame.events)
    )
      throw new Error("Invalid trace fields.");
    for (const rawEvent of frame.events) {
      const event = record(rawEvent);
      if (
        typeof event.request_id !== "string" ||
        !["prefill", "decode"].includes(String(event.phase)) ||
        ![
          "token_id",
          "logical_context_length",
          "model_input_width",
          "batch_size",
        ].every((key) => Number.isInteger(event[key]))
      )
        throw new Error("Invalid trace event.");
    }
    if (
      frame.waiting_request_ids !== undefined &&
      !strings(frame.waiting_request_ids)
    )
      throw new Error("Invalid waiting IDs.");
    for (const key of [
      "logical_kv_before_forward",
      "logical_kv_after_completion",
    ])
      if (frame[key] !== undefined) {
        const pool = record(frame[key]),
          tables = record(pool.request_block_tables);
        if (
          ![
            "total_blocks",
            "block_size",
            "free_blocks",
            "allocated_blocks",
          ].every((key) => Number.isInteger(pool[key])) ||
          typeof pool.utilization !== "number" ||
          !Object.values(tables).every(
            (blocks) => Array.isArray(blocks) && blocks.every(Number.isInteger),
          )
        )
          throw new Error("Invalid allocator snapshot.");
      }
  }
  return value as TraceFrame[];
}
export const missing = "—";
export const metric = (value: number | null | undefined, digits = 0) =>
  value == null || !Number.isFinite(value)
    ? missing
    : value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
export function requestIds(frames: TraceFrame[]): string[] {
  return [
    ...new Set(
      frames.flatMap((frame) => [
        ...frame.admitted,
        ...frame.active_request_ids,
        ...(frame.waiting_request_ids ?? []),
        ...frame.completed,
      ]),
    ),
  ];
}
export function stateAt(
  frames: TraceFrame[],
  index: number,
  request: string,
): string {
  const frame = frames[index];
  if (!frame) return missing;
  if (
    frames.slice(0, index + 1).some((item) => item.completed.includes(request))
  )
    return "finished";
  const event = frame.events.find((item) => item.request_id === request);
  if (event) return event.phase === "decode" ? "decoding" : "prefill";
  if (frame.waiting_request_ids?.includes(request)) return "waiting";
  return missing;
}
export function positionWork(frame: TraceFrame) {
  return {
    prefill: frame.events
      .filter((e) => e.phase === "prefill")
      .reduce((sum, e) => sum + e.model_input_width, 0),
    decode: frame.events
      .filter((e) => e.phase === "decode")
      .reduce((sum, e) => sum + e.model_input_width, 0),
  };
}
