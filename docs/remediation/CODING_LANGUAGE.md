# Supplemental semantics and language candidates

These are **separate unreviewed proposals**, not additions to the frozen 72-fit /
144,000-update primary study. No training, optimizer steps, scientific approval,
signatures, qualification or model-success claims are issued.

## Program semantics candidate

`scope_studies.semantic_candidates()` returns eight source-only program problems
and a separate protected reference mapping. Training examples cover straight-line
arithmetic, a single loop, a single branch and a straight-line function. Held-out
structures combine loop+branch, nested loops, function+branch and function+loop.
The task predicts the integer value of `result`; it is more substantive than
rename-candidate selection, but these eight illustrative programs are **not a
sufficient scientific benchmark** or evidence of program synthesis.

Reference values are computed by a separately implemented bounded AST interpreter.
It never invokes Python `eval`, `exec`, compilation or host task execution. It
allows integer literals/variables, addition/subtraction/multiplication/modulo,
single comparisons, assignment, bounded `range` loops, branches, plain functions
and returns. It rejects imports, attributes, arbitrary builtins, decorators,
annotations, while loops and other syntax. Source/AST/step/call-depth/range/integer
magnitude budgets bound evaluation. Targets do not enter source vocabulary fitting.
The original frozen tokenizer fits **training sources only**, and held-out unknown
words remain unknown.

`wasi_crosscheck` validates source against the restricted interpreter, then invokes
only the existing `WasiSandbox` with its probe and bounded execution. The guest
receives source plus `print(result)`, never the expected reference value. Agreement
is checked after execution. Missing WASI returns unavailable and no agreement;
there is no host fallback. References must obtain independent real WASI agreement
before use in a scientific study. Transport-fixture agreement tests are plumbing
evidence, not actual WASI execution claims.

`freeze_supplement` writes problem/reference/vocabulary artifacts separately with
hashes and a `NO_GO_UNREVIEWED` manifest, zero optimizer updates and zero primary
campaign fits added. This is a candidate freeze only. A future independent review
must expand family-balanced examples, prevent structural and lexical leakage,
freeze model/control/compute/inference schedules and scoring, and obtain authorized
independent gates. It cannot silently convert the primary study's coding claim
into general program semantics or synthesis.

## Language extraction evaluation

`language_metrics` consumes existing frozen language annotations **after predictions
exist**, and uses the existing training-only vocabulary. It reports by named split:
intent exact counts, all-argument exact examples, present-slot exact rate, null-slot
accuracy, strict exact-extraction precision/recall, unsupported-request abstention
precision/recall, and unknown-token totals. A span true positive requires exact
annotated token boundaries; wrong spans contribute predicted positives without
true positives. Null-heavy examples cannot inflate the separate present-slot score.
Abstention never counts as successful extraction. Zero denominators return null.

These metrics assess intent and token-boundary extraction only. Paraphrase,
reordered structure, unknown vocabulary and operand-composition splits retain their
existing narrow synthetic scope. OOV counts do not demonstrate OOV comprehension;
exact identifiers copied from text do not demonstrate scientific reasoning.
Unsupported labels cover only their frozen examples and do not establish broad
out-of-distribution detection. No trained language model or general semantic
qualification is claimed.

Validation: eleven nontraining tests check independently calculated semantics,
structural separation, training-only vocabulary, forbidden/unbounded source,
fail-closed missing WASI, reference-free guest plumbing, present/null/abstention
metrics and separate proposal freezing. No actual WASI runtime was used in this
module's test run; an available runtime must be tested independently before claiming
cross-execution agreement.
