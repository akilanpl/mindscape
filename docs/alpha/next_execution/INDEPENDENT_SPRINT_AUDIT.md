# Independent sprint audit — nontraining preparation only

Root independently inspected and reproduced the final submissions. Scientific status remains **UNRUN** for R5.4, OOD and transfer; Research Alpha remains **NO-GO**. Fully specified numerical candidates are reviewable, not native/scientific execution-qualified.

## Exact review sources and preserved scope

| Track | Immutable final implementation head | Reviewed parent |
|---|---|---|
| A Decision A | `7d022a53154e06ac8e7eca2e605d986a8218390e` | `950a17519f43fbc887e41f29ce2e24d4b6036ead`, PR20 |
| B native preflight | Clean detached `950a17519f43fbc887e41f29ce2e24d4b6036ead` | Exact supplied-script pin; output outside repo |
| C OOD numerical | `690e989ea59bc48cc6c9b424f774c7ae8292018e` | `d457428540b34a96e16c31fc4ffe3d13e8cba3fa`, PR19 |
| D transfer numerical | `0d2b25c43df003b4879e63f28c43847c1ab2abce` | `03fc481192dca47410dd809de1303658c2f556cb`, PR21 |
| Root audit/readiness | Final Git head exported after commit, avoiding self-reference | `c7b6638d434b9a5a597e49f9217ee57905ca11a8`, PR22 |

The live development baseline remains `6c558caa3c3a1a09ec15e286a408cbb6d5076b03`. Full pinned-parent byte preservation, clean worktrees, add-only committed diffs, exact protocol/source locks, every workload count and every mandatory gate were independently checked in FROZEN_SCOPE_CHECK.json. Root's Git batch audit hashes stored bytes, including binary evidence; it never interprets sealed reference labels. Source hashes of all committed additions are recorded. No baseline, frozen study, original NO_GO/FAILED_CAMPAIGN, release tag or frontend changed.

## Independently reproduced tests and logs

Root reproduced **185 checks**, counting each suite once:

- R5.4 exact77 nontraining tests: 24 storage/readiness +33 frozen gate +20 original engineering/diagnostic. A and B separately reproduced this same suite; repeated runs are not additional tests or native evidence.
- OOD58 synthetic tests: prior24 plus34 numerical/custody/reference/source/runtime/report-schema checks. CPU forward/backward uses invented inputs with no optimizer updates. No real suite or OOD labels were generated.
- Transfer27 synthetic tests: prior9 plus18 numerical/Adam/signature/source/runtime/receipt checks. CPU placement mocks are explicitly fixtures, not native PASS. Adam objects may be constructed/inspected but never stepped.
- Audit23 checks: prior13 generic/adversarial checks plus10 new scope checks. Seven new Git cases were authored and challenged independently by A, covering dirty baseline/candidates, unknown files, wrong ancestry, symlink checkout and committed-byte preservation.

Complete lead logs are under evidence/root; author logs under evidence/a, evidence/c and evidence/d. B's native evidence is under evidence/B and evidence/native; the approved script's raw exit1 is retained. A independently verified all48 B ledger hashes, runtime/readiness mismatch and77-unit-check log. B independently challenged transfer scoring with ten separate synthetic cases; its initial six failures and final corrected run remain in evidence/b-review. Reviewer-specific source identities are recorded there; this separate10-case review is not inflated into the lead185 count.

Compile-only static checks cover every added Python source without importing models or running campaigns. Exact native/independent executor source/WASI asset hash guards and explicit workload locks are implemented; real executor outcomes were not run. New numerical choices are versioned PROPOSED_REQUIRES_REVIEW or PROPOSED_REQUIRES_FREEZE_AND_REVIEW.

## Findings and corrections

A found the scope auditor accepted alias roots and dirty additions; root now rejects them and hashes all additions. B reproduced malformed validity/empty-effect scorer cases; D added strict binary, shape, domain, graph and denominator checks. Root required exact per-parameter Adam counters/moments, private training-target commitments, seeded CPU initialization parity, actual raw tensors, exclusive fsynced checkpoint/seal writes, separate prediction authority and signed independent metric verdicts. D implemented those safeguards and actual locked executor calls. Current source verification binds the executing checkout and immutable runtime snapshot.

Root found C's initial preseal procedural-row input leaked reference-derived data and removed it. Nonprocedural predictions now seal first; separately authorized release precedes independent procedural execution. Root required overall and per-family baseline gain/bootstrap gates, strict full verdict/denominator schema, actual source/runtime checks, checkpoint integrity and independent reference equality rather than copying labels. C implemented these without modifying prior frozen files. All failures, invalid outputs and incomplete campaign receipts remain visible; no silent retries occur.

## Native and trust boundaries

Observed supplied preflight exited1 at MPS=false on Darwin/arm64, Torch2.14.1. It stopped before its readiness/test phases. Separate saved-evidence review and77 nontraining checks succeeded but cannot turn that failure into native PASS. Original144 disposable engineering steps remain archived evidence; this sprint executed zero optimizer steps. Full native800-update timing and live AST numerical equivalence remain UNTESTED. Original Python/device/runHEAD cannot be retrospectively authenticated.

Production approvals and purpose-specific keys/custody providers remain unprovided. Ephemeral TEST_ONLY fixture signatures have no operational authority. Signature checks prove configured key possession; local hashes prove byte consistency, not human identity, trusted time or truth of unrecorded events. Gate computation is pending a distinct signed independent metric audit, not scientific PASS from an author's receipts. No scientific campaign exists to audit here. Before any actual claim, an uninvolved reviewer must recompute sealed raw predictions, checkpoints/optimizer states, execution references and every frozen gate after explicit stage authorization.

No merge, deployment, tag change, protected training, real dataset draw, neural OOD evaluation or Alpha announcement occurred.
