# Finalization evidence — browser verification pending

Production: [LLM Inference Runtime](https://llm-inference-runtime.vercel.app/#/runtime).

Static-only application revision before backend deployment: `85365edfca78665068d16bd5f653ac719555e518`. Main implementation/measurement commit: `173025c01468e74d24865f96aaca152a394cb550`. The measured runtime baseline was `ffdd08c`; raw measurement source revision identifies that baseline, not a claim that the new harness existed in its parent commit. The subsequent public API change is documented in [public runtime deployment](../../public-runtime.md).

**Not signed off as complete.** Browser access reports `Local browser use is disabled in Work settings`, including after a session reset following the owner's enablement confirmation. No alternative browser automation was used to bypass this restriction. Current final screenshots, connected/offline/replay interaction review, responsive review, public browser console/network review and manual public-site verification are still pending. Historical screenshots are not relabeled as current evidence.

## Verified checks

| Check | Evidence | Result |
| --- | --- | --- |
| Full Python suite | [raw output](checks/python-tests.txt) | 130 passed; six existing deprecation warnings |
| Frontend suite | [raw output](checks/frontend-tests.txt) | 30 passed across six files |
| TypeScript | [output](checks/typecheck.txt) | Exit 0; silent on success |
| Production build | [raw output](checks/production-build.txt) | Passed |
| Design audit | [output](checks/design-audit.txt) | Passed |
| Python compile | [output](checks/python-compile.txt) | Exit 0; quiet-mode success |
| Whitespace | [output](checks/git-diff-check.txt) | Exit 0 |
| Derivation | [output](checks/benchmark-sync.txt) | Seven artifacts |
| Exact commands/times/exit codes | [results](checks/results.json) | All exit codes 0 |
| Historical static-only checks | [old deployment report](deployment-static-only.json) | Retained as static evidence only, not live deployment success |
| Real public API | [health, CORS and SSE evidence](public-runtime.json) | JSON health; allowed-origin preflight 200, untrusted-origin 400; eight contiguous tokens, one done, no errors |

Logs retain actual test/build output, with user/workspace paths redacted and trailing whitespace normalized. Empty compiler/compile outputs are expected; exit codes are recorded separately. No hosted CI result or browser behavior is inferred from these checks.

## Measurement / streaming

- [Raw final measurement](../../../benchmarks/results/final_cpu.json): all paired run timings, token availability timestamps/gaps/IDs, five SSE event sequences/wire lines, worker-step observations and pooled distributions.
- [Methodology and limits](../../finalization.md): two warmups/mode, five alternating CLI pairs, 128 tokens, plus separately scoped serial HTTP/SSE requests. AC confirmed; existing power plan retained; CPU samples 21%/32% show residual background activity. Network server thread count was not recorded.
- All five measured SSE requests: 128 contiguous tokens, exactly one completion, zero errors; IDs match completion and paired CLI output. Download-free artifact-integrity tests enforce this.
- [Generated timing chart](../../images/final-token-gaps.svg), derived entirely from the raw artifact. UI computes the same per-position means.
- Worker steps: count 640, median 127.34 ms, p90 186.96 ms, p95 215.34 ms, maximum 1,404.60 ms. Dispatch/forward/observation wall time, not pure kernel time.
- Historical measurements/earlier local evidence ZIP remain unchanged. Historical M7/M13 potential host contention is a hypothesis, not established causality. The new run is not an isolated-host certification or a universal optimization claim.

## Implementation review status

Aliases preserve full IDs; unsupported decode-throughput placeholder removed; intentional static mode never probes localhost; disconnected/empty live datasets clear paused selection; identical consecutive scheduler states group without losing counts/admission/completion; repetition is explained; existing `#EFA23A` is canonical. The blank band was actual newline output in a pre-wrapped trace panel; escaped control characters plus raw-output details fix inspection without modifying model output. Tests cover these behaviors, but manual browser signoff remains required.

Vercel project: `amxsons-projects/llm-inference-runtime`. CLI production deployment now includes the real CPU Python function at `/api`; the frontend's production API URL is explicit. GitHub auto-deploy integration did not connect and is not claimed. No paid host/plan upgrade or local tunnel was provisioned. Local credentials/environment files and unrelated `assignment_review/` were excluded. Public HTTP validation does not substitute for the still-pending browser Playground/console/network check.
