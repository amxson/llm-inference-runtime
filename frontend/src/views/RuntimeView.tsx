import { useEffect, useState } from "react";
import type { Snapshot, TraceFrame } from "../api/types";
import fixture from "../data/recorded-trace.json";
import {
  metric,
  positionWork,
  requestIds,
  stateAt,
} from "../runtime/traceAdapter";
import { replayInterval, stepIndex } from "../runtime/replay";
import { recordedFrames } from "../runtime/recording";
import {
  groupSchedulerEvents,
  requestAlias,
  traceTokenText,
} from "../runtime/presentation";
import { liveUnavailable } from "../api/client";

export function RuntimeView({
  snapshot,
  mode,
  onMode,
  onFrame,
  initialIndex = 3,
}: {
  snapshot: Snapshot | null;
  mode: "live" | "replay";
  onMode: (mode: "live" | "replay") => void;
  onFrame: (index: number) => void;
  initialIndex?: number;
}) {
  const [index, setIndex] = useState(initialIndex),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [selection, setSelected] = useState("r2");
  const [frozen, setFrozen] = useState<TraceFrame[] | null>(null);
  const frames: TraceFrame[] =
    mode === "replay"
      ? recordedFrames
      : snapshot?.traces.length
        ? (frozen ?? snapshot.traces)
        : [];
  const cursor =
    mode === "live" && !frozen
      ? frames.length - 1
      : Math.min(index, frames.length - 1);
  const frame = frames[cursor];
  const ids = requestIds(frames),
    pool = frame?.logical_kv_before_forward;
  ids.forEach(requestAlias);
  const selected = ids.includes(selection) ? selection : (ids[0] ?? "");
  const label = requestAlias;
  const selectedState = stateAt(frames, cursor, selected);
  const chooseIndex = (next: number) => {
    setIndex(next);
    onFrame(next);
  };
  const step = (delta: number) => {
    setPlaying(false);
    if (mode === "live" && !frozen) {
      setFrozen(frames);
      chooseIndex(stepIndex(frames.length - 1, delta, frames.length));
    } else chooseIndex(stepIndex(cursor, delta, frames.length));
  };
  useEffect(() => {
    if (mode === "live" && !snapshot?.traces.length) setFrozen(null);
  }, [mode, snapshot]);
  useEffect(() => {
    if (!playing || mode !== "replay") return;
    const timer = setTimeout(() => {
      if (cursor >= frames.length - 1) setPlaying(false);
      else chooseIndex(cursor + 1);
    }, replayInterval(speed));
    return () => clearTimeout(timer);
  }, [playing, cursor, mode, speed, frames.length]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (
        event.target instanceof HTMLElement &&
        event.target.matches("input,textarea,select,[contenteditable=true]")
      )
        return;
      if (event.key === "," || event.key === ".") {
        event.preventDefault();
        step(event.key === "." ? 1 : -1);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  const maxWork = Math.max(
    1,
    ...frames.map((item) => {
      const work = positionWork(item);
      return work.prefill + work.decode;
    }),
  );
  const ownership = pool?.request_block_tables[selected] ?? [];
  const selectedTokens = frames
    .slice(0, cursor + 1)
    .flatMap((item) =>
      item.events.filter((event) => event.request_id === selected),
    );
  return (
    <>
      <div className="page-heading">
        <p className="eyebrow">Execution / continuous batching</p>
        <div className="heading-line">
          <h1>Runtime trace</h1>
          <span className="mode-label">
            {mode === "replay"
              ? "Recorded trace"
              : frozen
                ? "Live capture · paused"
                : "Live · completed worker boundaries"}
          </span>
        </div>
        <p>
          One iteration axis. One request selection. Watch admission, shared
          work, and completion.
        </p>
      </div>
      <div className="transport" aria-label="Trace transport">
        <div className="segmented">
          <button
            aria-pressed={mode === "replay"}
            onClick={() => {
              onMode("replay");
              setFrozen(null);
              setPlaying(false);
              chooseIndex(3);
            }}
          >
            Replay
          </button>
          <button
            aria-pressed={mode === "live"}
            onClick={() => {
              onMode("live");
              setFrozen(null);
              setPlaying(false);
            }}
          >
            Live
          </button>
        </div>
        <button
          aria-label="Previous iteration"
          onClick={() => step(-1)}
          disabled={cursor <= 0}
        >
          ◀
        </button>
        <button
          onClick={() =>
            mode === "replay"
              ? setPlaying(!playing)
              : setFrozen(frozen ? null : frames)
          }
        >
          {mode === "replay"
            ? playing
              ? "Pause"
              : "Play"
            : frozen
              ? "Resume"
              : "Pause"}
        </button>
        <button
          aria-label="Next iteration"
          onClick={() => step(1)}
          disabled={cursor >= frames.length - 1}
        >
          ▶
        </button>
        <label className="inline-label">
          Speed{" "}
          <select
            aria-label="Replay speed"
            disabled={mode === "live"}
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
          >
            <option value={0.25}>0.25×</option>
            <option value={1}>1×</option>
          </select>
        </label>
        <span className="mono">
          it {metric(frame?.tick)} / {metric(frames.at(-1)?.tick)}
        </span>
        <span className="muted shortcut">, / . to step</span>
      </div>
      {mode === "replay" ? (
        <p className="caption">
          Actual distilgpt2 CPU forwards through ContinuousBatcher; logical
          allocator enabled for this recording. Discrete ticks; playback cadence
          is not measured elapsed time.
        </p>
      ) : (
        <p className="caption">
          {snapshot
            ? "Continuous uncached greedy HTTP engine. No logical allocator is enabled."
            : liveUnavailable}
        </p>
      )}
      <section className="instrument-section" aria-labelledby="timeline-title">
        <div className="section-title">
          <h2 id="timeline-title">Padded input positions per iteration</h2>
          <span className="caption">
            Solid = prefill rows · outlined = decode rows · includes padding
          </span>
        </div>
        {frames.length ? (
          <div
            className="timeline-scroll"
            tabIndex={0}
            role="region"
            aria-label="Iteration timeline; scroll horizontally when needed"
          >
            <div
              className="timeline-grid"
              style={{
                gridTemplateColumns: `112px repeat(${frames.length},minmax(40px,1fr))`,
              }}
            >
              <span className="axis-label">positions</span>
              {frames.map((item, i) => {
                const work = positionWork(item);
                return (
                  <button
                    className={`work-column ${i === cursor ? "at-cursor" : ""} ${i > cursor ? "future" : ""}`}
                    key={item.tick}
                    aria-label={`Inspect iteration ${item.tick}`}
                    aria-pressed={i === cursor}
                    onClick={() => {
                      setPlaying(false);
                      if (mode === "live") setFrozen(frames);
                      chooseIndex(i);
                    }}
                  >
                    <span className="mono work-number">
                      {work.prefill + work.decode}
                    </span>
                    <span className="work-stack">
                      <span
                        className="work-decode"
                        style={{
                          height: `${(work.decode / maxWork) * 88}px`,
                          display: work.decode ? "block" : "none",
                        }}
                      />
                      <span
                        className="work-prefill"
                        style={{ height: `${(work.prefill / maxWork) * 88}px` }}
                      />
                    </span>
                  </button>
                );
              })}
              <span className="axis-label">iteration</span>
              {frames.map((item, i) => (
                <span
                  key={item.tick}
                  className={`tick-label mono ${i === cursor ? "at-cursor" : ""}`}
                >
                  {item.tick}
                </span>
              ))}
              {ids.map((id) => (
                <div
                  className={`lane ${id === selected ? "is-selected" : "is-dimmed"}`}
                  key={id}
                >
                  <button
                    className="request-label mono"
                    aria-label={`Select request ${label(id)}`}
                    title={`Request ID: ${id}`}
                    aria-pressed={selected === id}
                    onClick={() => setSelected(id)}
                  >
                    {label(id)}
                  </button>
                  {frames.map((item, i) => {
                    const event = item.events.find(
                      (event) => event.request_id === id,
                    );
                    const waiting = item.waiting_request_ids?.includes(id);
                    const completed = item.completed.includes(id);
                    return (
                      <button
                        key={item.tick}
                        className={`lane-cell ${event ? event.phase : waiting ? "waiting" : "empty"} ${completed ? "completed" : ""} ${i > cursor ? "future" : ""} ${i === cursor ? "cursor-cell" : ""}`}
                        disabled={!event && !waiting}
                        aria-label={`${label(id)}, iteration ${item.tick}, ${event ? event.phase : waiting ? "waiting" : "not observed"}${completed ? ", finishes" : ""}`}
                        onClick={() => {
                          setSelected(id);
                          if (mode === "live") setFrozen(frames);
                          chooseIndex(i);
                        }}
                      >
                        {event
                          ? completed
                            ? "end"
                            : event.phase === "prefill"
                              ? "P"
                              : "D"
                          : waiting
                            ? "wait"
                            : ""}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="empty-state">
            {snapshot
              ? "No live iterations observed yet. Run a request in Playground."
              : "Choose Replay above to explore an engine recording."}
          </p>
        )}
        <p className="caption">
          Every decode row recomputes its full context. The trace does not imply
          cached batched inference.
        </p>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Request state</h2>
          <span className="mono selected-label">
            {selected
              ? `${label(selected)} · ${selectedState}`
              : "No request selected"}
          </span>
        </div>
        <div className="state-machine">
          {["waiting", "prefill", "decoding", "finished"].map((state, i) => (
            <span
              className={state === selectedState ? "current-state" : ""}
              key={state}
            >
              {i > 0 && <span className="state-arrow">→</span>}
              {state}
              <span className="mono state-count">
                {
                  ids.filter((id) => stateAt(frames, cursor, id) === state)
                    .length
                }
              </span>
            </span>
          ))}
        </div>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Logical KV block allocator</h2>
          <span className="caption">
            Block ownership model · not physical tensor paging
          </span>
        </div>
        <div className="allocator-region">
          <div>
            {pool ? (
              <>
                <div className="allocator-grid">
                  {Array.from({ length: pool.total_blocks }, (_, block) => {
                    const owner = Object.entries(
                      pool.request_block_tables,
                    ).find(([, blocks]) => blocks.includes(block))?.[0];
                    return (
                      <button
                        className={`block ${owner ? "owned" : "free"} ${owner === selected ? "selected-block" : owner ? "is-dimmed" : ""}`}
                        key={block}
                        disabled={!owner}
                        aria-label={`Pool block ${block}, ${owner ? `owned by ${label(owner)}` : "free"}`}
                        onClick={() => owner && setSelected(owner)}
                      >
                        <span className="block-id">{block}</span>
                        <span>{owner ? label(owner) : "free"}</span>
                      </button>
                    );
                  })}
                </div>
                <p className="mono caption">
                  {pool.allocated_blocks}/{pool.total_blocks} owned · block=
                  {pool.block_size} tokens · {metric(pool.utilization * 100, 1)}
                  % pool utilization
                </p>
              </>
            ) : (
              <p className="empty-state">
                — No logical allocator snapshot available on the live HTTP path.
              </p>
            )}
          </div>
          <div>
            <h3 className="minor-title">
              {selected
                ? `${label(selected)} / block table before forward`
                : "Block table / no request selected"}
            </h3>
            <table>
              <thead>
                <tr>
                  <th>Logical index</th>
                  <th className="numeric">Pool block ID</th>
                  <th>Owner</th>
                </tr>
              </thead>
              <tbody>
                {ownership.map((block, i) => (
                  <tr key={block}>
                    <td className="mono">{i}</td>
                    <td className="numeric">{block}</td>
                    <td className="mono">{label(selected)}</td>
                  </tr>
                ))}
                {!ownership.length && (
                  <tr>
                    <td colSpan={3}>— No blocks held in this snapshot</td>
                  </tr>
                )}
              </tbody>
            </table>
            <p className="caption">
              Request tables and ownership only. No reference counts, shared
              prefixes, or tensor addresses. HF cache storage is a separate
              layer.
            </p>
          </div>
        </div>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Selected request / token output</h2>
          <span className="mono">
            {selected ? label(selected) : "No request selected"}
          </span>
        </div>
        <div className="runtime-tokens">
          {selectedTokens.length
            ? selectedTokens.map((event, i) => (
                <span
                  key={i}
                  className={
                    i === selectedTokens.length - 1 ? "last-token" : ""
                  }
                  title={`token ${event.token_id} · index ${event.token_index ?? "—"} · ${selected}`}
                >
                  {traceTokenText(event.token_text ?? "—")}
                </span>
              ))
            : "—"}
        </div>
        {selectedTokens.length > 0 && (
          <p className="caption">
            Trace inspection escapes control characters as \n, \r and \t. This
            preserves newline-heavy model output without reserving empty lines;
            the raw decoded prefix is retained in details.
          </p>
        )}
        {selectedTokens.length > 0 && (
          <details className="result-details">
            <summary>Raw selected-request output</summary>
            <pre>
              {JSON.stringify(
                {
                  request_id: selected,
                  generated_text: selectedTokens.at(-1)?.generated_text ?? null,
                  token_ids: selectedTokens.map((event) => event.token_id),
                },
                null,
                2,
              )}
            </pre>
          </details>
        )}
      </section>
      {selected && mode === "live" && (
        <details className="result-details">
          <summary>{label(selected)} / full request ID</summary>
          <code>{selected}</code>
        </details>
      )}
      <section className="instrument-section scheduler">
        <div className="section-title">
          <h2>Scheduler log</h2>
          <span className="caption">
            Newest first · consecutive scheduler states grouped · {cursor + 1}{" "}
            observed ticks in retained window
          </span>
        </div>
        {groupSchedulerEvents(frames.slice(0, cursor + 1))
          .reverse()
          .map(({ first, last: item, count }) => (
            <div
              className={`log-line mono ${item.active_request_ids.includes(selected) ? "selected-log" : "is-dimmed"}`}
              key={item.tick}
            >
              <span>
                it {first.tick}
                {count > 1 ? `–${item.tick}` : ""}
              </span>
              <span>
                admit {item.admitted.map(label).join(", ") || "—"} · batch{" "}
                {item.model_request_ids.length}/
                {mode === "replay"
                  ? fixture.max_active
                  : (snapshot?.max_active ?? "—")}{" "}
                · queue {metric(item.queue_depth)} · complete{" "}
                {item.completed.map(label).join(", ") || "—"}
                {count > 1 ? ` · ×${count}` : ""}
              </span>
            </div>
          ))}
      </section>
    </>
  );
}
