import { TextEncoder, TextDecoder } from "node:util";
import { parseSSE, streamGeneration } from "../api/sse";
import { vi } from "vitest";
Object.assign(globalThis, { TextDecoder, TextEncoder });
function body(text: string, split = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += split)
        controller.enqueue(bytes.slice(i, i + split));
      controller.close();
    },
  });
}
const token = {
  request_id: "r1",
  token_id: 3,
  token_index: 1,
  token_text: " é😀",
  generated_text: " é😀",
  tick: 1,
  phase: "prefill",
};
const done = {
  request_id: "r1",
  generated_token_ids: [3],
  generated_text: " é😀",
  stop_reason: "max_new_tokens",
  generated_token_count: 1,
};
test("SSE handles byte-split Unicode, CRLF boundaries, comments and unknown events", async () => {
  const events = [];
  for await (const event of parseSSE(
    body(
      `: ping\r\nevent: unknown\r\ndata: {}\r\n\r\nevent: token\r\ndata: ${JSON.stringify(token)}\r\n\r\nevent: done\r\ndata: ${JSON.stringify(done)}\r\n\r\n`,
    ),
  ))
    events.push(event);
  expect(events).toEqual([
    { event: "token", data: token },
    { event: "done", data: done },
  ]);
});
test("multi-line data and LF-only frames parse; malformed token payloads fail", async () => {
  const events = [];
  for await (const event of parseSSE(
    body('event: error\ndata: {\ndata: "message":"failed"}\n\n', 5),
  ))
    events.push(event);
  expect(events[0]).toEqual({ event: "error", data: { message: "failed" } });
  await expect(
    (async () => {
      for await (const event of parseSSE(body("event: token\ndata: {}\n\n")))
        void event;
    })(),
  ).rejects.toThrow("Malformed token");
});
test("connection without terminal done is an error, not fake completion", async () => {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue({
        ok: true,
        body: body(`event: token\ndata: ${JSON.stringify(token)}\n\n`),
      }),
  );
  await expect(
    streamGeneration("test", 1, new AbortController().signal, () => {}),
  ).rejects.toThrow("before a done");
  vi.unstubAllGlobals();
});
test("error then done are both delivered and only real request fields are sent", async () => {
  const fetcher = vi
    .fn()
    .mockResolvedValue({
      ok: true,
      body: body(
        `event: error\ndata: {"message":"failed"}\n\nevent: done\ndata: ${JSON.stringify({ ...done, stop_reason: "error" })}\n\n`,
      ),
    });
  vi.stubGlobal("fetch", fetcher);
  const received: string[] = [];
  await streamGeneration("test", 2, new AbortController().signal, (event) =>
    received.push(event.event),
  );
  expect(received).toEqual(["error", "done"]);
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
    prompt: "test",
    max_new_tokens: 2,
  });
  vi.unstubAllGlobals();
});
