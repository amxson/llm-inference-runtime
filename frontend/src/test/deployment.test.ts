import { vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.resetModules();
});
test("production static mode makes no request to localhost or an inferred backend", async () => {
  vi.stubEnv("DEV", false);
  vi.stubEnv("VITE_API_BASE_URL", "");
  vi.resetModules();
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  const client = await import("../api/client");
  const sse = await import("../api/sse");
  expect(client.apiBase).toBe("");
  await expect(client.getSnapshot()).rejects.toThrow(/not connected/);
  await expect(
    sse.streamGeneration("a", 1, new AbortController().signal, () => {}),
  ).rejects.toThrow(/not connected/);
  expect(fetch).not.toHaveBeenCalled();
});
test("an explicitly configured backend URL is used unchanged except trailing slash", async () => {
  vi.stubEnv("DEV", false);
  vi.stubEnv("VITE_API_BASE_URL", "https://runtime.example/");
  vi.resetModules();
  const client = await import("../api/client");
  expect(client.apiBase).toBe("https://runtime.example");
});
