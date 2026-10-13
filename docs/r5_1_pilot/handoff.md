# Authorized R5.1 first learning pilot — NOT RUN

Human authorization: one native experiment, up to 27 fits. Original V5/R5.1 model, gates/evidence and the earlier nine-fit proposal remain unchanged. The new all-domain freeze is `configs/r5_1/learning_v2/{protocol,data-binding,freeze}.json`; its scope implements the expanded authorization. No learned result exists yet.

Architecture: unchanged V5 GraphTransition, 6,967 parameters. Primary math+physics -> coding; secondary pooled-source transfer to math/physics measures regression. Three seeds 11/23/47 and domain-only/shared/randomized-source conditions yield 27 fits. Each fit has 120 updates x16 graphs, Adam .003; total <=3,240 updates/51,840 presentations. First 60 updates are target-only for B or alternating source domains for C/R; last60 target-only for all. Target labels are the same18 development tasks in every condition; source domains have18 development tasks each. Repetition/encoder-update differences are disclosed. Qualification labels never optimize a model.

Effect loss is per-graph .5 mean changed-entity squared error +.5 mean unchanged-active-entity squared error, plus .1 validity BCE; absent entity groups contribute no fabricated success. Supervised changed masks accompany labels only; inference receives public graphs/actions. R permutes paired source effects/validity within domain using fixed seed-derived permutations, preserving input graphs. Exact assignments are saved without searching for favorable corruption.

Before any update: validate complete committed protocol/source/data/runtime freezes and original native receipts, independently replay81 existing readiness outcomes, and run native MPS engineering checks. Failure stops without training. A fixed unique output directory prevents another attempt or continuation; there is no retry/resume/settings override. The launcher forbids CPU fallback/downloads.

All27 fits persist final public-graph predictions and model+Adam checkpoints (updates60/120) before any of27 composition outcomes are executed. Only then execute nine held-out tasks/domain, independently replay them, compute saved-output metrics and inspect all54 checkpoint/Adam states without neural forwards. Three seeds repeat27 independent task identities, not81 tasks. Record effects, probabilities, curves at0/60/90/120, loss parts, gradient/update allocations, active/inactive block changes, hashes, per-fit/inference/checkpoint time, paired task differences and exploratory task bootstrap intervals. Whole/changed/unchanged errors, invalid no-op errors, validity accuracy/Brier, integer/kinematic consistency and complete transition correctness remain visible. Persistence, public local heuristics and perfect procedural expert are reported.

Frozen scientific GO gates: mean coding changed-MSE improvement >=10% over BOTH B/R and every seed's gain strictly positive; shared coding improvement >=20% over persistence and >=10% over local-action; validity balanced accuracy>=.70 for every shared coding seed; mean whole-graph error no worse than either control; coding mean unchanged and invalid-noop errors each<=.0001; math/physics mean relative regression vsB<=10%; each domain-only fit improves development changed error from initialization>=20%; all outputs/gradients finite and integrity checks PASS. Zero-error controls cannot yield epsilon-assisted gains. These thresholds are fixed before outcomes; primary/negative-transfer limits retain original V5 objectives, adapted to changed-entity measurement.

Any numeric gate fails -> NO_GO. All pass -> GO_TO_CONFIRMATION only. Integrity/engineering/budget/nonfinite failure -> FAILED_PILOT with preserved partial evidence and stop; no numerical failure is rescued as INCONCLUSIVE. Any incomplete evidence will be explicitly assessed after native execution. No second run, tuning, larger dataset, extra seed or automatic confirmation.

Expected output: `results/r5_1_learning/native-one-pilot`, plus complete archive/preservation hashes. Initial loss/state hashes do not prove exact bitwise continuation; no resume is implemented. Nine compositions/domain and six expected changed tasks limit uncertainty estimates. Task-bootstrap intervals are descriptive and do not treat seeds/entities as extra independent tasks. No claims of general arithmetic, full program repair or general physics follow this bounded microdomain test.

Native command (run once in normal macOS Terminal):

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/r5_1_pilot/native.command
```

Focused pure-arithmetic/source-boundary tests pass. No sandbox neural forward, optimizer step or composition execution occurred during preparation. After the native result returns, independently audit original bytes and checkpoint/metrics, preserve evidence, and report the honest scientific verdict. No website work or public release.
