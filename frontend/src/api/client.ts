import type { Snapshot } from "./types";
import { parseTraceFrames } from "../runtime/traceAdapter";
export const apiBase = (
  import.meta.env.VITE_API_BASE_URL?.trim() ||
  (import.meta.env.DEV ? "http://127.0.0.1:8765" : "")
).replace(/\/$/, "");
export const liveUnavailable =
  "Live profiling is not connected in this deployment. Use Replay to explore recorded executions; no live data is fabricated.";
export async function getJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  if (!apiBase) throw new Error(liveUnavailable);
  const response = await fetch(`${apiBase}${path}`, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}
export async function getSnapshot(signal?: AbortSignal) {
  const snapshot = await getJson<Snapshot>("/runtime/snapshot", signal);
  snapshot.traces = parseTraceFrames(snapshot.traces);
  return snapshot;
}
