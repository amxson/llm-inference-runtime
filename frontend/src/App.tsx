import { useEffect, useState } from "react";
import type { Snapshot } from "./api/types";
import { apiBase, getSnapshot } from "./api/client";
import { RuntimeView } from "./views/RuntimeView";
import { PlaygroundView } from "./views/PlaygroundView";
import { BenchmarksView } from "./views/BenchmarksView";
import { ArchitectureView } from "./views/ArchitectureView";
import fixture from "./data/recorded-trace.json";
import { metric } from "./runtime/traceAdapter";

export const projectUrl = "https://github.com/amxson/llm-inference-runtime";
const routes = ["Playground", "Runtime", "Benchmarks", "Architecture"] as const;
type Route = (typeof routes)[number];
function currentRoute(): Route {
  return (
    routes.find(
      (name) => location.hash.toLowerCase() === `#/${name.toLowerCase()}`,
    ) ?? "Runtime"
  );
}
export function App() {
  const [route, setRoute] = useState(currentRoute),
    [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [mode, setMode] = useState<"replay" | "live">("replay");
  const displayMode = route === "Playground" ? "live" : mode;
  const [frameIndex, setFrameIndex] = useState(3);
  useEffect(() => {
    const change = () => setRoute(currentRoute());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    // Replay/static pages remain independent and don't wake an inference instance.
    if (
      !apiBase ||
      (route !== "Playground" && !(route === "Runtime" && mode === "live"))
    )
      return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        setSnapshot(await getSnapshot(controller.signal));
      } catch {
        if (!controller.signal.aborted) setSnapshot(null);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 500);
      }
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [route, mode]);
  const frame =
    displayMode === "replay"
      ? fixture.traces[frameIndex]
      : snapshot?.traces.at(-1);
  const allocator =
    frame && "logical_kv_before_forward" in frame
      ? frame.logical_kv_before_forward
      : null;
  const model = displayMode === "replay" ? fixture.model : snapshot?.model;
  const max =
    displayMode === "replay" ? fixture.max_active : snapshot?.max_active;
  const status =
    displayMode === "replay"
      ? "replay"
      : snapshot
        ? snapshot.in_flight
          ? "running"
          : "idle"
        : "offline";
  return (
    <div className="app-shell">
      <a
        className="skip-link"
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to instruments
      </a>
      <header className="topbar">
        <span>LLM Inference Runtime</span>
        <span className="config mono">
          {model ?? "backend unavailable"} ·{" "}
          {displayMode === "replay"
            ? fixture.device
            : (snapshot?.device ?? "—")}
          {allocator ? ` · block=${allocator.block_size}` : ""} · max_batch=
          {metric(max)}
        </span>
        <span className={`engine-status ${status}`}>
          <span className="status-square" />
          {status}
        </span>
      </header>
      <div className="statusline mono" aria-label="Runtime status">
        <span>
          {displayMode === "replay" ? "recorded trace" : "live snapshot"}
        </span>
        <span>
          iteration{" "}
          {metric(
            frame?.tick ??
              (displayMode === "live" ? snapshot?.tick : undefined),
          )}
        </span>
        <span>
          batch{" "}
          {metric(
            displayMode === "live"
              ? snapshot?.batch_size
              : frame?.model_request_ids.length,
          )}
          /{metric(max)}
        </span>
        <span>
          logical blocks{" "}
          {allocator
            ? `${allocator.allocated_blocks}/${allocator.total_blocks}`
            : "—"}
        </span>
        <span>
          queue{" "}
          {metric(
            displayMode === "live" ? snapshot?.queue_depth : frame?.queue_depth,
          )}
        </span>
      </div>
      <aside className="navigation">
        <nav aria-label="Main navigation">
          {routes.map((name) => (
            <a
              key={name}
              aria-current={route === name ? "page" : undefined}
              href={`#/${name.toLowerCase()}`}
            >
              {name}
            </a>
          ))}
        </nav>
        <div className="rail-foot">
          <a href={projectUrl}>GitHub</a>
          <a
            href={`${projectUrl}/tree/${fixture.engine_commit}`}
            className="mono"
          >
            engine {fixture.engine_commit.slice(0, 7)}
          </a>
          <span>Trace / runtime instruments</span>
        </div>
      </aside>
      <main id="main" tabIndex={-1}>
        {route === "Runtime" ? (
          <RuntimeView
            snapshot={snapshot}
            mode={mode}
            onMode={setMode}
            onFrame={setFrameIndex}
            initialIndex={frameIndex}
          />
        ) : route === "Playground" ? (
          <PlaygroundView snapshot={snapshot} />
        ) : route === "Benchmarks" ? (
          <BenchmarksView />
        ) : (
          <ArchitectureView />
        )}
      </main>
    </div>
  );
}
