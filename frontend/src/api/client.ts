import type { Snapshot } from "./types";
import { parseTraceFrames } from "../runtime/traceAdapter";
export const apiBase = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8765"
).replace(/\/$/, "");
export async function getJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}
export async function getSnapshot(signal?: AbortSignal) {
  const snapshot = await getJson<Snapshot>("/runtime/snapshot", signal);
  snapshot.traces = parseTraceFrames(snapshot.traces);
  return snapshot;
}
