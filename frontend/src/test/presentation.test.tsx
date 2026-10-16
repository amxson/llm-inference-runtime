import { fireEvent, render, screen } from "@testing-library/react";
import {
  groupSchedulerEvents,
  requestAlias,
  traceTokenText,
} from "../runtime/presentation";
import { RuntimeView } from "../views/RuntimeView";
import type { Snapshot, TraceFrame } from "../api/types";

const id = "1234567890abcdef1234567890abcdef";
const frame = (tick: number): TraceFrame => ({
  tick,
  admitted: [],
  completed: [],
  active_request_ids: [id],
  model_request_ids: [id],
  waiting_request_ids: [],
  queue_depth: 0,
  events: [
    {
      request_id: id,
      token_id: tick,
      phase: "decode",
      model_input_width: tick,
      logical_context_length: tick,
      batch_size: 1,
      token_text: " x",
    },
  ],
});

test("aliases are stable while real IDs are preserved", () => {
  expect(requestAlias(id)).toMatch(/^r\d+$/);
  expect(requestAlias(id)).toBe(requestAlias(id));
  expect(requestAlias(id + "a")).not.toBe(requestAlias(id));
  expect(requestAlias("r4")).toBe("r4");
});
test("newline-heavy traces remain explicit without rendering blank lines", () => {
  expect(traceTokenText("\n\n\t\r")).toBe("\\n\\n\\t\\r");
  const traces = Array.from({ length: 128 }, (_, i) => ({
    ...frame(i + 1),
    events: [
      {
        ...frame(i + 1).events[0],
        token_text: "\n",
        generated_text: "\n".repeat(i + 1),
      },
    ],
  }));
  render(
    <RuntimeView
      snapshot={{ traces, max_active: 4 } as Snapshot}
      mode="live"
      onMode={() => {}}
      onFrame={() => {}}
    />,
  );
  const tokens = document.querySelector(".runtime-tokens")!;
  expect(tokens.textContent).not.toContain("\n");
  expect(tokens.textContent).toBe("\\n".repeat(128));
  expect(
    screen.getByText(/Trace inspection escapes control characters/),
  ).toBeInTheDocument();
  expect(screen.getByText(/"generated_text":/)).toHaveTextContent(
    '"generated_text"',
  );
});
test("only consecutive identical scheduler states collapse, with exact counts", () => {
  const frames = Array.from({ length: 190 }, (_, i) => frame(i + 1));
  const groups = groupSchedulerEvents(frames);
  expect(groups).toHaveLength(1);
  expect(groups[0].count).toBe(190);
  expect(groups[0].first.tick).toBe(1);
  expect(groups[0].last.tick).toBe(190);
  const admission = { ...frame(191), admitted: ["new"] };
  const queue = { ...frame(193), queue_depth: 1, waiting_request_ids: ["new"] };
  const changed = groupSchedulerEvents([
    ...frames,
    admission,
    frame(192),
    queue,
    frame(194),
  ]);
  expect(changed).toHaveLength(5);
  expect(changed.reduce((n, g) => n + g.count, 0)).toBe(194);
  expect(groupSchedulerEvents([frame(1), frame(3)])).toHaveLength(2);
  expect(
    groupSchedulerEvents([frame(1), { ...frame(2), completed: [id] }]),
  ).toHaveLength(2);
});
test("losing live data clears frozen selection and dependent panels immediately", () => {
  const snapshot = { traces: [frame(1), frame(2)], max_active: 4 } as Snapshot;
  const props = { mode: "live" as const, onMode: () => {}, onFrame: () => {} };
  const { rerender } = render(<RuntimeView {...props} snapshot={snapshot} />);
  expect(
    screen.getByRole("button", { name: `Select request ${requestAlias(id)}` }),
  ).toHaveAttribute("title", `Request ID: ${id}`);
  fireEvent.click(screen.getByRole("button", { name: "Pause" }));
  rerender(<RuntimeView {...props} snapshot={{ ...snapshot, traces: [] }} />);
  expect(
    screen.queryByRole("button", { name: /Select request/ }),
  ).not.toBeInTheDocument();
  rerender(<RuntimeView {...props} snapshot={null} />);
  expect(
    screen.queryByRole("button", { name: /Select request/ }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "Block table / no request selected" }),
  ).toBeInTheDocument();
  expect(
    screen.getByText(/Live profiling is not connected/),
  ).toBeInTheDocument();
  expect(document.body).not.toHaveTextContent(id);
  rerender(<RuntimeView {...props} snapshot={{ ...snapshot, traces: [] }} />);
  expect(
    screen.queryByRole("button", { name: /Select request/ }),
  ).not.toBeInTheDocument();
});
