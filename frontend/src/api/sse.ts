import { apiBase, liveUnavailable } from "./client";
import type { StreamEvent } from "./types";

export function decodeEvent(event: string, text: string): StreamEvent | null {
  if (!["token", "error", "done"].includes(event)) return null;
  const data = JSON.parse(text);
  if (!data || typeof data !== "object")
    throw new Error("Malformed SSE payload");
  if (
    event === "token" &&
    (typeof data.request_id !== "string" ||
      !Number.isInteger(data.token_id) ||
      !Number.isInteger(data.token_index) ||
      typeof data.token_text !== "string" ||
      typeof data.generated_text !== "string")
  )
    throw new Error("Malformed token event");
  if (event === "error" && typeof data.message !== "string")
    throw new Error("Malformed error event");
  if (
    event === "done" &&
    (!Array.isArray(data.generated_token_ids) ||
      typeof data.generated_text !== "string" ||
      typeof data.stop_reason !== "string")
  )
    throw new Error("Malformed completion event");
  return { event, data } as StreamEvent;
}

export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<StreamEvent> {
  const reader = body.getReader(),
    decoder = new TextDecoder();
  let buffer = "",
    kind = "message",
    data: string[] = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });
      while (true) {
        const boundary = buffer.search(/[\r\n]/);
        if (
          boundary < 0 ||
          (!done && boundary === buffer.length - 1 && buffer[boundary] === "\r")
        )
          break;
        const line = buffer.slice(0, boundary);
        buffer = buffer.slice(
          boundary + (buffer.slice(boundary, boundary + 2) === "\r\n" ? 2 : 1),
        );
        if (!line) {
          if (data.length) {
            const parsed = decodeEvent(kind, data.join("\n"));
            if (parsed) yield parsed;
          }
          kind = "message";
          data = [];
        } else if (!line.startsWith(":")) {
          const colon = line.indexOf(":"),
            field = colon < 0 ? line : line.slice(0, colon);
          const value =
            colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") kind = value;
          if (field === "data") data.push(value);
        }
      }
      if (done) break; // SSE requires a blank-line boundary; never invent a terminal event.
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}

export async function streamGeneration(
  prompt: string,
  max_new_tokens: number,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void,
) {
  if (!apiBase) throw new Error(liveUnavailable);
  const response = await fetch(`${apiBase}/generate/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_new_tokens }),
    signal,
  });
  if (!response.ok)
    throw new Error(`Generation rejected (HTTP ${response.status}).`);
  if (!response.body) throw new Error("Streaming response has no body.");
  let terminal = false;
  for await (const event of parseSSE(response.body)) {
    onEvent(event);
    if (event.event === "done") {
      terminal = true;
      break;
    }
  }
  if (!terminal) throw new Error("Stream ended before a done event.");
}
