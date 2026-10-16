import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { vi } from "vitest";
import { RuntimeView } from "../views/RuntimeView";
import { PlaygroundView } from "../views/PlaygroundView";
import { BenchmarksView, PositionFigure } from "../views/BenchmarksView";
import { ArchitectureView } from "../views/ArchitectureView";
import { App } from "../App";
import type { StreamEvent } from "../api/types";
const { streamMock, snapshotMock } = vi.hoisted(() => ({
  streamMock: vi.fn(),
  snapshotMock: vi.fn(),
}));
vi.mock("../api/sse", () => ({ streamGeneration: streamMock }));
vi.mock("../api/client", () => ({
  getSnapshot: snapshotMock,
  apiBase: "http://127.0.0.1:8765",
  liveUnavailable:
    "Live profiling is not connected in this deployment. Use Replay to explore recorded executions; no live data is fabricated.",
}));
beforeEach(() => {
  streamMock.mockReset();
  snapshotMock.mockReset();
  snapshotMock.mockRejectedValue(new Error("offline"));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

test("linked selection updates lane, block ownership, table and token output", async () => {
  const user = userEvent.setup();
  render(
    <RuntimeView
      snapshot={null}
      mode="replay"
      onMode={() => {}}
      onFrame={() => {}}
    />,
  );
  expect(
    screen.getByRole("heading", { name: "r2 / block table before forward" }),
  ).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Select request r4" }));
  expect(
    screen.getByRole("heading", { name: "r4 / block table before forward" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Pool block 5, owned by r4" }),
  ).toHaveClass("selected-block");
  expect(
    screen.getByRole("button", { name: "Select request r4" }),
  ).toHaveAttribute("aria-pressed", "true");
  expect(document.querySelector(".runtime-tokens")).not.toHaveTextContent("—");
});
test("transport steps with keyboard, disables at end, and live missing data stays absent", async () => {
  const frame = vi.fn();
  const { unmount } = render(
    <RuntimeView
      snapshot={null}
      mode="replay"
      onMode={() => {}}
      onFrame={frame}
    />,
  );
  fireEvent.keyDown(window, { key: "." });
  expect(frame).toHaveBeenLastCalledWith(4);
  await userEvent.click(
    screen.getByRole("button", { name: "Inspect iteration 9" }),
  );
  expect(screen.getByRole("button", { name: "Next iteration" })).toBeDisabled();
  unmount();
  render(
    <RuntimeView
      snapshot={null}
      mode="live"
      onMode={() => {}}
      onFrame={() => {}}
    />,
  );
  expect(screen.getByText(/No logical allocator snapshot/)).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: /Pool block/ }),
  ).not.toBeInTheDocument();
});
test("Playground renders real token events before completion and has only supported controls", async () => {
  let deliver: (event: StreamEvent) => void = () => {};
  let finish: () => void = () => {};
  streamMock.mockImplementation((_p, _b, _s, callback) => {
    deliver = callback;
    return new Promise<void>((resolve) => {
      finish = resolve;
    });
  });
  render(<PlaygroundView snapshot={null} />);
  expect(screen.getByLabelText("Prompt")).toBeInTheDocument();
  expect(screen.getByLabelText("max_new_tokens")).toHaveAttribute("max", "128");
  expect(
    screen.queryByLabelText(/temperature|top.k|cache/i),
  ).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Run" }));
  act(() =>
    deliver({
      event: "token",
      data: {
        request_id: "actual-r1",
        token_id: 12,
        token_index: 1,
        token_text: " emitted",
        generated_text: " emitted",
        tick: 7,
        phase: "prefill",
      },
    }),
  );
  expect(
    within(screen.getByRole("region", { name: "Streamed output" })).getByText(
      "emitted",
    ),
  ).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("running");
  await act(async () => {
    deliver({
      event: "done",
      data: {
        request_id: "actual-r1",
        generated_token_ids: [12],
        generated_text: " emitted",
        generated_token_count: 1,
        stop_reason: "max_new_tokens",
      },
    });
    finish();
  });
  expect(screen.getByRole("status")).toHaveTextContent("done");
  expect(
    screen.getByRole("button", { name: "Token 1, tick 7" }),
  ).toBeInTheDocument();
});
test("error followed by done cannot overwrite the error state", async () => {
  streamMock.mockImplementation(async (_p, _b, _s, callback) => {
    callback({ event: "error", data: { message: "Inference failed." } });
    callback({
      event: "done",
      data: {
        request_id: "r",
        generated_token_ids: [],
        generated_text: "",
        stop_reason: "error",
        generated_token_count: 0,
      },
    });
  });
  render(<PlaygroundView snapshot={null} />);
  await userEvent.click(screen.getByRole("button", { name: "Run" }));
  expect(screen.getByRole("status")).toHaveTextContent("error");
  expect(screen.getByRole("status")).toHaveTextContent("Inference failed.");
});
test("cancellation and navigation abort the actual fetch signal", async () => {
  let signal: AbortSignal | undefined;
  streamMock.mockImplementation((_p, _b, s) => {
    signal = s;
    return new Promise(() => {});
  });
  const { unmount } = render(<PlaygroundView snapshot={null} />);
  await userEvent.click(screen.getByRole("button", { name: "Run" }));
  await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(signal?.aborted).toBe(true);
  unmount();
});
test("zero-token completion has no fabricated token events or TTFT", async () => {
  streamMock.mockImplementation(async (_p, _b, _s, callback) =>
    callback({
      event: "done",
      data: {
        request_id: "zero",
        generated_token_ids: [],
        generated_text: "",
        generated_token_count: 0,
        stop_reason: "max_new_tokens",
      },
    }),
  );
  render(<PlaygroundView snapshot={null} />);
  await userEvent.clear(screen.getByLabelText("max_new_tokens"));
  await userEvent.type(screen.getByLabelText("max_new_tokens"), "0");
  await userEvent.click(screen.getByRole("button", { name: "Run" }));
  expect(screen.getByRole("status")).toHaveTextContent("done");
  expect(
    screen.queryByRole("button", { name: /Token \d/ }),
  ).not.toBeInTheDocument();
  const ttft = screen.getByText("Browser-observed TTFT").parentElement!;
  expect(ttft).toHaveTextContent("— ms");
});
test("benchmark page retains negative results and raw source links", async () => {
  const bundle = JSON.parse(
    readFileSync("public/data/benchmarks.json", "utf8"),
  );
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, json: async () => bundle }),
  );
  render(<BenchmarksView />);
  expect(
    await screen.findByRole("heading", { name: /Speculation.*slower/ }),
  ).toBeInTheDocument();
  expect(screen.getByText(/Generated output changed/)).toBeInTheDocument();
  expect(
    screen.getAllByRole("link", { name: "Raw JSON" }).length,
  ).toBeGreaterThanOrEqual(6);
});
test("position figure draws one actual cell for each forwarded position", () => {
  const { container } = render(
    <PositionFigure uncached={[5, 6, 7]} cached={[5, 1, 1]} />,
  );
  const groups = container.querySelectorAll("svg>g");
  expect(groups[0].querySelectorAll("[data-kind=forwarded]")).toHaveLength(18);
  expect(groups[1].querySelectorAll("[data-kind=forwarded]")).toHaveLength(7);
  expect(groups[1].querySelectorAll("[data-kind=reused]")).toHaveLength(11);
});
test("architecture visibly separates HTTP from cached decoding", () => {
  render(<ArchitectureView />);
  expect(
    screen.getByRole("list", { name: "Live uncached HTTP data path" }),
  ).toHaveTextContent("cache=False");
  expect(
    screen.getByRole("list", { name: "Separate cached decoding data path" }),
  ).toHaveTextContent("HF DynamicCache");
  expect(
    screen.getByText(/Capacity modeling—not physical/),
  ).toBeInTheDocument();
});

test("replay advances exactly one tick at the selected presentation cadence", () => {
  vi.useFakeTimers();
  const frame = vi.fn();
  const { unmount } = render(
    <RuntimeView
      snapshot={null}
      mode="replay"
      onMode={() => {}}
      onFrame={frame}
    />,
  );
  try {
    fireEvent.change(screen.getByLabelText("Replay speed"), {
      target: { value: "0.25" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => vi.advanceTimersByTime(3999));
    expect(frame).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1));
    expect(frame).toHaveBeenLastCalledWith(4);
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    act(() => vi.advanceTimersByTime(8000));
    expect(frame).toHaveBeenCalledTimes(1);
  } finally {
    unmount();
    vi.useRealTimers();
  }
});
test("shell routes remain text-only and skip link does not change route", async () => {
  window.location.hash = "#/playground";
  render(<App />);
  expect(screen.getByRole("link", { name: "Playground" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  await userEvent.click(
    screen.getByRole("link", { name: "Skip to instruments" }),
  );
  expect(document.activeElement).toBe(screen.getByRole("main"));
  expect(window.location.hash).toBe("#/playground");
  expect(screen.getByLabelText("Runtime status")).toHaveTextContent(
    "live snapshot",
  );
  expect(screen.getByLabelText("Runtime status")).not.toHaveTextContent(
    "recorded trace",
  );
});
