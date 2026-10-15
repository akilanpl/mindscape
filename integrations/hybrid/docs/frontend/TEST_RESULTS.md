# Frontend test results

Before editing: original hybrid suite **173 passed**. After implementation: **173 passed**, including solver, acceptance, server and architecture-boundary tests. Frontend Node suite: **14 passed**, no dependencies required. `node --check` passes for changed JavaScript.

Original native pytest in a disposable clean worktree: **146 passed, 22 skipped, 6 subtests passed**. After deleting only `integrations/hybrid` there: **146 passed, 22 skipped, 6 subtests passed**. Native no-site-packages imports initialize no hybrid/torch/transformers/SymPy; original episode23*17 returns391. Original optional-runtime skips remain skips; this does not certify GPU/cloud paths.

| Real functional check | Result |
|---|---|
| Mathematics / real hybrid | Passed — 3*x**2 + cos(x); exact local MathML and original raw expression. |
| Physics / real hybrid | Passed — 3.0 A; I=V/R and units preserved. |
| Python / real hybrid | Passed — Exact stdout 10 and real bounded interpreter trace. |
| Unsupported / real hybrid | Passed — Missing circuit data refused, tool_verified=false. |
| SQL / real hybrid | Passed — Asha92, Ravi84 rendered from the actual returned two-row table. |
| Literal stdout / actual tools-only | Passed — **2** and <b>literal</b> remain literal; no HTML nodes created. |
| Copy code/output | Passed — Browser clipboard exactly matches submitted code and returned stdout. |
| Busy / explicit isolated fixture | Passed — First HTTP429 preserves input; explicit retry yields one user message and one actual tools answer. |
| Cancellation / real pretrained request | Passed — Stop aborts waiting, asks existing cancel API, restores text and warns that in-flight work may finish. |
| Persistence | Passed — Chats, active conversation and saved answers survive reload; legacy store remains intact. |
| Projects | Passed — Real local project creation and selection; no cloud persistence implied. |
| Library / recovery | Passed — Save, confirmed move to Recently deleted, restore and local search exercised. |
| Keyboard | Passed — Enter submits; Shift+Enter produces two lines and autosizes; Control+K creates new chat; Escape returns drawer focus. |
| Sidebar during inference | Passed — Collapse did not cancel the real request; response still arrived. |
| Mobile drawer | Passed — Inert background/offscreen sidebar, Escape close and focus return checked. |
| Browser console | Passed — No errors/warnings in final inspected tab. |
| Offline startup | Passed — External socket connections denied; five local assets loaded; real pinned Qwen+solver response verified in 3.265s; zero optimizer updates. |

Real mathematics/physics/Python/refusal results were retrieved via the running browser, not fabricated DOM responses. Their actual backend provenance shows pretrained_used=true, zero optimizer updates and the correct verified/unsupported flags. The additional literal-output case uses explicitly labelled tools-only mode. The429 case is an intentional fixture on a different loopback port; it is not counted as a real model prediction. A few automation locator waits/selectors needed retargeting; completed app results and state were inspected before proceeding, with no extra inference retries claimed.

The Node tests cover exact powers/fractions, safe Markdown/code highlighting, preserved signed precision/units, unverified provenance, legacy migration, malformed-store recovery, structured SQL cells including negative zero, draft recovery, literal Python output, and ** operators surviving prose formatting. They execute trusted frontend source in a minimal test DOM, not a real-browser substitute; browser actions provide the separate integration evidence.

## Reproduce locally

```sh
cd integrations/hybrid
PYTHONPATH=. .venv/bin/python -m pytest -q tests
node --test tests/frontend/rendering.test.cjs
node --check web/app.js
node --check web/math.js
# Uses an already installed pinned official model; no downloads:
PYTHONPATH=. .venv/bin/python tests/offline_startup.py /path/to/Qwen3-0.6B
# Explicit non-production fault fixture; first chat429, retry uses real tools:
PYTHONPATH=. .venv/bin/python tests/serve_ui_faults.py
```

The fixture starts port8773 and is not part of normal startup. Stop it with Ctrl-C. Screenshots and detailed local evidence are in the delivered frontend-redesign output directory; tests do not write research checkpoints. No native CI workflow or dependency manifest was changed. These frontend tests were run locally; GitHub's original checks are tracked separately in PUSH_VERIFICATION.json and the PR.
