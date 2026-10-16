import type { TraceFrame } from "../api/types";

// Browser-session aliases only; raw IDs remain the keys for all data/selection.
const aliases = new Map<string, string>();
export function traceTokenText(text: string): string {
  return text.replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}
export function requestAlias(id: string): string {
  if (!id) return "";
  if (/^r\d+$/.test(id)) return id; // Recorded fixture's own identifiers.
  if (!aliases.has(id)) aliases.set(id, `r${aliases.size + 1}`);
  return aliases.get(id)!;
}

export interface LogGroup {
  first: TraceFrame;
  last: TraceFrame;
  count: number;
}
export function groupSchedulerEvents(frames: TraceFrame[]): LogGroup[] {
  const groups: LogGroup[] = [];
  const key = (frame: TraceFrame) =>
    JSON.stringify({
      active: frame.active_request_ids,
      model: frame.model_request_ids,
      waiting: frame.waiting_request_ids,
      queue: frame.queue_depth,
      phases: frame.events.map((event) => [event.request_id, event.phase]),
    });
  for (const frame of frames) {
    const previous = groups.at(-1);
    // Admissions/completions are never merged. Token changes live in the token
    // panel; this groups only identical scheduler membership/queue/phase states.
    if (
      previous &&
      !frame.admitted.length &&
      !frame.completed.length &&
      !previous.last.admitted.length &&
      !previous.last.completed.length &&
      frame.tick === previous.last.tick + 1 &&
      key(frame) === key(previous.last)
    ) {
      previous.last = frame;
      previous.count++;
    } else groups.push({ first: frame, last: frame, count: 1 });
  }
  return groups;
}
