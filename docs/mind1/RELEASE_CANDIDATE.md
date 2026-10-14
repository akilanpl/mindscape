# Mind 1 / R5.6 review candidate — 2026-10-14

**Acceptance verdict: INCOMPLETE. This is a working local review build, not a qualified full product release.**

Source branch: development/mind1-r56. Draft PR: https://github.com/akilanpl/mindscape/pull/27
Reviewed integration baseline: 8c141d8 (research/alpha-r54, PR23).

| Status | Evidence and scope |
|---|---|
| IMPLEMENTED | Loopback API, configurable Groq/OpenAI-compatible provider, bounded multi-step tool routing, context handling, cancellation, responsive persistent chat, safe Markdown/MathML/tables, actual data charts, provenance, capabilities page |
| TESTED | 146 passing tests, 14 skips, 9 passing subtests across product/R5.6 plus preserved kernel and authorization safeguards. Test XML supplied. |
| VERIFIED | Browser → API → symbolic integration → derivative residual 0 → displayed result and provenance. Desktop 1440×900 and mobile 390×844 inspected. Actual WASI print/sum and host-file denial checked. Matrix/ODE/physics tests check objective expectations. |
| EXPERIMENTAL | R5.6 concept/state/action/composition/verifier interfaces, bounded carry/AST/impulse examples, candidate untrained neural core, matched-control protocol. Generalization NOT demonstrated. No training executed. |
| UNAVAILABLE | Live provider credential, streaming, public accounts/authentication, formal theorem proving, non-Python guest execution, full repository patch/test writeback, broad expert-reviewed advanced coverage. Live arXiv request returned unavailable. |
| BLOCKED | Credential-backed Groq acceptance: MIND1_API_KEY absent in agent environment. Twelve live cases are explicitly skipped; no external-model answers are presented as verified. |

## Product and scientific boundaries

Groq endpoint defaults to https://api.groq.com/openai/v1, model openai/gpt-oss-120b,
reasoning_effort high. Credentials are read only from server environment and never
sent to browser or tool workers. Foundation-model capabilities are not attributed
to R5.6. Live model access, quality, quota and latency are not established by the
configuration or contract fixtures. Graduate and research responses require expert
review; live tests do not automatically certify those answers as correct.

Local symbolic tools deliberately bound syntax, polynomial degrees and dimensions.
Numerical minimization assumes a unimodal bounded interval. Fixed-step ODE/physics
results are approximate; residuals are evidence of specific checks, not general
physical truth. Static analysis outside Python uses limited patterns and inventory,
not full compiler semantics. Only WASI Python is executable. A passing isolation
probe is required before source execution. Operator must supply a trusted runtime;
no WASI runtime or compiled cache is bundled in the source zip.

Provider contract fixtures are used only in tests. The application never substitutes
a mock answer backend when inference is unavailable. No private verifier data is
available to provider integration. Optional public retrieval uses a fixed arXiv API;
its public-query restriction is instructional, not a semantic privacy classifier.
Leave it disabled for private conversations.

## Security and preservation

Loopback binding only; Host/Origin checks, JSON/size limits, four concurrency slots,
rate limiting, CSP, safe DOM rendering, allowlisted tool dispatch, resource ceilings,
redacted errors/logs. Backend does not retain conversations; browser localStorage
provides persistence and is shared by users of the same browser profile. No public
account isolation claim. Cancellation terminates tool workers; in-flight provider
calls can continue until bounded socket timeout.

Dependency audit: nine resolved requirement packages, zero known vulnerabilities
at audit time. This does not establish absence of vulnerabilities. Historical
website/, configs/, experiments/, research_checkpoint/ and protected evidence have
zero diff against baseline. No merge, tag or deployment occurred.

Initial broader historical test collection could not import torch; it was not
silently treated as passing. Kernel physics initially lacked pymunk; installing
its historical pin 7.2.0 allowed the final kernel regression suite to pass.
Protected training/optimizer campaigns and historical full workloads were not run
or reduced. No R5.5 historical release was invented.

Final end-to-end evidence found and repaired an inherited file-size limit mismatch
between the product tool worker and nested WASI worker. A regression now executes
actual guest source through the complete product worker path.

## Handoff

See README.md for reproducible setup, API and limits, COVERAGE.md for the tiered
live evaluation, SECURITY_REVIEW.md for independent review, and ../r56/RESEARCH.md
for protocol/status. The source zip is a standalone product subset; the Git branch
preserves the complete research repository. A baseline-relative patch is supplied.

Next qualification requires setting MIND1_API_KEY in the server/test environment,
restarting the server, running all twelve live cases, reviewing advanced answers,
and resolving the unavailable acceptance items before a release claim.
