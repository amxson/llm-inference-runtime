import { useEffect, useRef, useState } from "react";
import type { GenerationResult, Snapshot, TokenEvent } from "../api/types";
import { streamGeneration } from "../api/sse";
import { metric } from "../runtime/traceAdapter";
import { requestAlias } from "../runtime/presentation";
import { apiBase, liveUnavailable } from "../api/client";

export function PlaygroundView({ snapshot }: { snapshot: Snapshot | null }) {
  const [prompt, setPrompt] = useState("The capital of France is"),
    [budget, setBudget] = useState(10);
  const [tokens, setTokens] = useState<TokenEvent[]>([]),
    [result, setResult] = useState<GenerationResult | null>(null);
  const [state, setState] = useState<
      "idle" | "running" | "done" | "error" | "cancelled"
    >("idle"),
    [error, setError] = useState("");
  const [ttft, setTtft] = useState<number | null>(null),
    [latency, setLatency] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null),
    first = useRef(false);
  useEffect(() => () => controller.current?.abort(), []);
  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (state === "running") return;
    const active = new AbortController();
    controller.current = active;
    first.current = false;
    setTokens([]);
    setResult(null);
    setSelected(null);
    setError("");
    setTtft(null);
    setLatency(null);
    setState("running");
    const start = performance.now();
    let failed = false;
    try {
      await streamGeneration(prompt, budget, active.signal, (event) => {
        if (event.event === "token") {
          if (!first.current) {
            first.current = true;
            setTtft(performance.now() - start);
          }
          setTokens((previous) => [...previous, event.data]);
        } else if (event.event === "error") {
          failed = true;
          setError(event.data.message);
          setState("error");
        } else {
          setResult(event.data);
          setLatency(performance.now() - start);
          if (
            event.data.stop_reason === "error" ||
            event.data.stop_reason === "cancelled"
          ) {
            failed = true;
            setError(
              (previous) => previous || `Generation ${event.data.stop_reason}.`,
            );
          }
          setState(failed ? "error" : "done");
        }
      });
    } catch (cause) {
      if (active.signal.aborted) setState("cancelled");
      else {
        setState("error");
        setError(
          cause instanceof Error
            ? cause.message
            : "Unable to stream generation.",
        );
      }
    } finally {
      if (controller.current === active) controller.current = null;
    }
  }
  const text = tokens.at(-1)?.generated_text ?? "";
  // Some token fragments split Unicode. Never replace the correct decoded prefix
  // with naive concatenation merely to support hover wrappers.
  const canWrap = tokens.map((token) => token.token_text).join("") === text;
  const selectedToken = tokens[selected ?? tokens.length - 1];
  const workerStep = (token?: TokenEvent) =>
    snapshot?.traces.find((frame) => frame.tick === token?.tick)
      ?.worker_step_ms;
  return (
    <>
      <div className="page-heading">
        <p className="eyebrow">Live inference / explicit greedy decoding</p>
        <div className="heading-line">
          <h1>Run a forward loop</h1>
          <span className="mode-label">
            Live SSE · {snapshot?.model ?? "backend unavailable"}
          </span>
        </div>
        <p>
          Submit a prompt to the real runtime. Each emitted token comes from a
          model forward—not a typewriter animation.
        </p>
      </div>
      {!snapshot && (
        <p className="caption">
          {liveUnavailable} <a href="#/runtime">Explore Replay</a>
        </p>
      )}
      <form onSubmit={run} className="prompt-form">
        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          maxLength={4096}
          required
          rows={3}
        />
        <div className="input-controls">
          <label htmlFor="budget">
            max_new_tokens{" "}
            <input
              id="budget"
              className="mono"
              type="number"
              min={0}
              max={128}
              step={1}
              required
              value={budget}
              onChange={(event) => setBudget(Number(event.target.value))}
            />
          </label>
          <span className="caption">greedy serving</span>
          <span className="caption">
            serving cache: off{" "}
            <span className="muted">(uncached continuous batcher)</span>
          </span>
          <div className="run-actions">
            <button
              type="submit"
              className="primary"
              disabled={!apiBase || state === "running" || !prompt.trim()}
            >
              Run
            </button>
            <button
              type="button"
              disabled={state !== "running"}
              onClick={() => controller.current?.abort()}
            >
              Cancel
            </button>
          </div>
        </div>
      </form>
      <div
        className="stream-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <span
          className={`status-square ${state === "error" ? "error-square" : ""}`}
        />
        {state}
        {error && <span className="error-text"> · {error}</span>}
      </div>
      <section
        className="instrument-section output-section"
        aria-label="Streamed output"
      >
        <h2>Output</h2>
        <div className="stream-output" aria-live="off">
          {text ? (
            canWrap ? (
              tokens.map((token, i) => (
                <span
                  tabIndex={0}
                  key={`${token.request_id}-${token.token_index}`}
                  className={i === tokens.length - 1 ? "last-token" : ""}
                  title={`token ID ${token.token_id} · index ${token.token_index} · request ${token.request_id} · worker step ${metric(workerStep(token), 2)} ms (dispatch + forward)`}
                  onFocus={() => setSelected(i)}
                  onMouseEnter={() => setSelected(i)}
                >
                  {token.token_text}
                </span>
              ))
            ) : (
              text
            )
          ) : (
            <span className="output-placeholder">
              {state === "running"
                ? "Waiting for the first emitted token…"
                : state === "done" && result?.generated_token_count === 0
                  ? "Completed with zero new tokens."
                  : "Output will appear here when the backend emits it."}
            </span>
          )}
        </div>
        <p className="caption token-detail">
          {selectedToken ? (
            <>
              token <span className="mono">{selectedToken.token_id}</span> ·
              index <span className="mono">{selectedToken.token_index}</span> ·
              request{" "}
              <span
                className="mono"
                title={`Request ID: ${selectedToken.request_id}`}
              >
                {requestAlias(selectedToken.request_id)}
              </span>{" "}
              · worker step {metric(workerStep(selectedToken), 2)} ms (dispatch
              + forward)
            </>
          ) : (
            "Hover or focus a token to inspect its ID and request. Worker timing appears only when a matching telemetry tick is available."
          )}
        </p>
      </section>
      <section className="instrument-section">
        <div className="section-title">
          <h2>Execution strip</h2>
          <span className="caption">
            Discrete engine ticks · not milliseconds
          </span>
        </div>
        <div className="execution-strip">
          {tokens.length ? (
            tokens.map((token, i) => (
              <button
                key={i}
                aria-label={`Token ${token.token_index}, tick ${token.tick ?? "unknown"}`}
                className={i === selected ? "selected-event" : ""}
                onClick={() => setSelected(i)}
              >
                <span className="mono">{metric(token.tick)}</span>
                <span>{token.phase ?? "—"}</span>
                <span className="mono">{token.token_id}</span>
              </button>
            ))
          ) : (
            <span className="caption">— No token events received</span>
          )}
        </div>
      </section>
      <dl className="facts-row">
        <div>
          <dt>Generated</dt>
          <dd>
            {metric(tokens.length)} <span>tok</span>
          </dd>
        </div>
        <div>
          <dt>Current batch</dt>
          <dd>
            {metric(snapshot?.batch_size)}/{metric(snapshot?.max_active)}
          </dd>
        </div>
        <div>
          <dt>Server forwards</dt>
          <dd>{metric(snapshot?.model_forward_count)}</dd>
        </div>
        <div>
          <dt>Current queue</dt>
          <dd>{metric(snapshot?.queue_depth)}</dd>
        </div>
        <div>
          <dt>Browser-observed TTFT</dt>
          <dd>
            {metric(ttft, 1)} <span>ms</span>
          </dd>
        </div>
        <div>
          <dt>Done arrival</dt>
          <dd>
            {metric(latency, 1)} <span>ms</span>
          </dd>
        </div>
      </dl>
      <p className="caption">
        {snapshot?.model === "distilgpt2" &&
          "The small distilgpt2 demo model uses greedy decoding; repetitive continuations can occur. "}
        Arrival metrics include the browser/HTTP path and are not model-step
        latency. Batch, queue and forwards are server-wide snapshot values, not
        per-request work. Zero-token requests have no TTFT. Sampling and
        KV-cached decoding exist as separate paths, not live controls.
      </p>
      {result && (
        <details className="result-details">
          <summary>Result and token IDs</summary>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </details>
      )}
    </>
  );
}
