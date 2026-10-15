# Grounded conversational follow-ups

`Assistant.chat` validates optional history and a conversation identifier before routing follow-ups. Explicit supported new problems still take the original solver route, including Python questions beginning with “Explain”. Existing request fields and legacy history-only callers remain supported.

Hybrid and Tools only modes recognize simple explanations, worked steps, numbered/ordinal steps, formula questions, negative-sign questions, examples, and explanations of the previous answer. Answers are deterministic commentary on a matching server-owned verified problem, solver result and worked solution. Differentiation rules describe the parsed expression's operations without evaluating code or modifying a solver. Physics explanations repeat the checked relation and supplied problem; Python explanations use the actual execution trace and exact output. Examples illustrate a rule or reuse the supplied checked calculation. New parameter values require a new problem.

The response has `status: explained`, exact `verified_answer` and `verified_units`, and provenance `previous_result_verified: true`, `tool_verified: false`, `explanation_verified: false`. This distinguishes a previously checked final answer from explanatory prose that was not independently verified. No additional solver or pretrained inference is used. Native and Pretrained modes request a switch to Hybrid/Tools for grounded explanations, preserving their separate provenance.

## Context boundaries

- At most 24 submitted messages, 16,000 characters per message, 64,000 characters total, subject also to the existing HTTP byte limit.
- Messages contain only a `user`/`assistant` role and string content. Client-supplied provenance, system messages and malformed schemas are rejected.
- Server stores at most 32 recent turn records globally, each expiring after 30 minutes. Entries are deep copies and protected by the assistant lock; the cache is memory-only and disappears on restart. Responses longer than 16,000 characters are not retained.
- Lookup binds the conversation identifier and the exact latest user/assistant pair to a response actually produced by this server. Neither assistant text nor a claimed solver result supplied by a client can establish verification. Follow-up chains retain the original verified context. A failed, missing, edited, expired, wrong-domain or unverified response yields clarification; there is no global last-answer fallback.
- The browser sends the current chat's identifier and only bounded history from that chat. Legacy callers omitting the identifier must provide a matching pair previously returned to an identifier-less request. Identifiers are routing boundaries, not authentication credentials: the server remains a local, single-user loopback service.
- This adds explanations of the latest matched result, not arbitrary conversational memory, cross-chat retrieval or newly trained model capabilities. Imported/pre-restart conversations must resend their original problem. Numeric/physical sign questions without a supported explanation request clarification instead of invented conventions.

## Validation on 2026-10-15

- Original 173 hybrid tests plus 50 context regressions: **223 passed**.
- Original 14 frontend tests plus 2 context tests: **16 passed**.
- Actual model-backed localhost HTTP run: **50 checks passed**, including required derivative follow-ups, physics, Python, failure barriers, two conversations, new-chat clarification, forged history, tools-only follow-ups and 30 latency samples.
- Browser: exact requested derivative followed by “Explain in simple terms.” displays product/chain rules, original worked solution and the unchanged derivative, with accurate prose provenance.
- Native suite with integration present: **146 passed, 22 skipped, 6 subtests passed**. Same results with `integrations/hybrid` removed in a temporary detached worktree. Dependency-free native episode: **391**, verification passed, with and without integration.
- Actual HTTP follow-up latency: median **2.51 ms**, p95 **2.65 ms** (30 samples); median server processing **0.575 ms**. No additional model inference. Existing HTTP slot-release pacing is outside these timings.
- Controlled 600-sample comparison against baseline `503e098`, using identical solver fixtures to isolate orchestration: median added overhead **0.0145 ms**. This excludes actual solver/model/HTTP execution and is not a claim about end-to-end solve speed.

Reproduce automated checks from the integration directory with `PYTHONPATH=. python -m pytest -q tests` and `node --test tests/frontend/rendering.test.cjs`. Run the actual HTTP checks with `python tests/context_http_check.py --base http://127.0.0.1:8772 --output /path/to/results.json` after starting the model-backed server.

All changes are contained in this optional integration. Solvers, native code, weights and historical results are untouched. Revert the dedicated context recovery commit to remove this feature; removing the integration still leaves the native project functional.
