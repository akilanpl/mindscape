# V4 fresh-data qualification — frozen, awaiting native inference

PR #7 research CI passed: Research v2 CPU validation, V3 foundation, mathematics, coding controls, and physics CPU boundaries. Vercel deployment failure is separate and does not invalidate research checks. Raw GitHub receipts are under `experiments/v4/qualification_preflight/`.

The existing `configs/v4/fresh-development-proposal.json` is unchanged. Complete public tasks and private scorers are frozen in `configs/v4/qualification_v1/tasks.json`; `freeze.json` binds executable sources, proposal, manifest and original specialist assets, including the actual physics checkpoint. These files must be committed before inference; runtime checks committed bytes and rejects changes.

Exactly 36 fresh identities are planned, at most 36 requests may execute. Order: first four training tasks by math/physics/coding; four qualification tasks by the same order; then the remaining four training tasks by the same order. Gate status stays pending until its four-task cohort is complete. The first failed gate stops before another request. Generation, integrity and replay failures also stop. If all gates pass and all 36 requests finish, DATA_READY is recorded and the sprint stops. There is no training, automatic continuation, replacement pool, retries or seed search. A deterministic output directory also prevents accidental repeat execution of the screen.

Math incoming contexts are balanced; canonical old V4 math states are excluded. Physics scenes use seed 2026101404 and indices 2000–2011, horizons 8/16, and public observed position plus the predeclared offset. Freeze construction executes only the initial three public-observation warmup steps, never a candidate action or future trajectory. Coding uses literals 101–112 and four declared repair families; AST duplicates are removed before scoring and task-hash shuffle preserves score/source bindings. Private scorer fields are stripped before calling the unchanged specialist proposal function.

Public features and raw predicted scores/proposals are persisted before independent candidate verification. All scored labels, including a row whose replay fails, are retained. Every completed candidate is independently executed again before the next request. Raw generations, deduplication groups, per-task positive/negative labels, gate values, early-stop/unexecuted task counts, asset integrity and complete compressed evidence are saved. The frozen specialist score remains an input to the existing encoder; it is not a private scorer outcome.

Run once in normal macOS Terminal:

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v4/qualify.command
```

The pinned native Python and MPS are required. No CPU neural fallback or model download is permitted. Output prints the fixed absolute run path and DATA_NO_GO or DATA_READY. If execution fails, preserve the failure and return its path; do not retry automatically.

Seven focused qualification tests cover manifest determinism/freshness, exact label/headroom gates, early stop, AST deduplication/shuffle, no training entrypoint, and actual arithmetic replay/tamper rejection. Qualification commits use `[skip ci]` because legacy coding CI launches a training job; local qualification tests and Ruff are recorded separately. PR #7's completed original research checks remain intact.

Current status: **NOT_RUN**. No observed gate values or qualification decision exist yet. After the native receipt returns, replay/preserve it, finalize the result and factual website handoff, then stop the research sprint. No website edits are authorized.
