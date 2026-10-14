# Unseen-composition generalization — UNRUN

Baseline: `origin/development/r5.3-robust-physics`, commit
`6c558caa3c3a1a09ec15e286a408cbb6d5076b03`. This new, independently versioned
`alpha-ood-v1` proposal does not change R5.2, R5.4, their labels or their gates.
Scientific status is **UNRUN**. Benchmark engineering is prepared; data qualification,
human protocol adoption, training authorization and independent audit remain pending.
No real OOD predictions, sealed reference inspections, optimizer steps or fits occurred.
Synthetic CPU forwards/backwards validate architecture only; no optimizer is constructed.

## Historical motivation, preserved evidence

`experiments/r5_2_math/recovery_v1/REPORT.md` preserves the original FAILED_CAMPAIGN
and a post-hoc NO_GO. Recovered recurrent mean exactness was 8.33%, reset-hidden
25.00%, memory 21.53%; recurrent intermediate accuracy was 81.7453%.
Each seed reused the same 144 tasks. Local accuracy did not imply complete-task
accuracy. Signs, accumulated transition errors and recurrent negative transfer were
material failures. Attribution depended on frozen dictionary order after an overwritten
condition variable; no new neural re-forward was authorized. These aggregate results
motivate this proposal; historical holdouts were not read or used for selection.

## Implemented nontraining framework

`mindscape.alpha_ood` contains deterministic input-only task generation, identities
that exclude split/family labels, duplicate/historical identity rejection, exact-task
memorization and predicted-state-feedback transition lookup, a procedural reference,
matched-feature feedforward/recurrent architecture adapters, preparation receipts and
sealed-prediction replay. The arithmetic scheduler and observable feature function are
reused from existing R5.2 sources without modifying them. A serialized callback workflow enforces independent receipt verification against
frozen protocol/source/input/label hashes before reaching a fit callback. Numerical
training is unavailable here: loss/exposure/teacher-forcing specifications still need
review, and no numerical fit callback is supplied. Readiness always fails closed.

Feedforward and recurrent models use identical observable digit/operation/incoming
features, embeddings and sign/digit/carry heads. Common layers initialize identically
for matched seeds. GRU core has 15,840 parameters; feedforward core has 15,744,
a disclosed 0.61% difference. Feedforward ignores temporal hidden state; recurrent
retains it inside arithmetic rows, with the supplied scheduler resetting row state.
Both retain supplied signed-magnitude/place-value scheduling, multiplication rows
and serialization. This is learned transition evaluation with a declared scaffold,
not discovery of arithmetic algorithms. No old checkpoint is promoted to evidence.

## Frozen proposal and eligibility boundary

`configs/alpha_ood/protocol-v1.json` fixes 768 training tasks, 96 validation tasks,
36 OOD tasks in each of six families (216 unique OOD identities), seeds 11/23/47,
800 updates per learned fit, batch 32, Adam .003, final checkpoint only and six fits.
Validation has the familiar training distribution and is identity-disjoint; it is for
predeclared engineering diagnostics, not checkpoint selection or protocol tuning.

The families separate novel operand combinations, 5–8 operation sequences,
4–8 digit carry chains, 4–8 digit borrow chains, unseen operation schedules, and
alternating combinations of familiar transition constituents. Training uses two or
three homogeneous familiar operations. No family is automatically a causal isolation:
longer sequences also change schedules; chains change length and operands.

**DATA_UNQUALIFIED:** family labels are intended constructions, not evidence that all
constituent contexts were exposed or all context combinations are novel. An independent
custodian must certify this before any execution. Certify each reference local context
`(operation,left_digit,right_digit,incoming_carry_or_borrow)` against supervised training
contexts, full ordered context-sequence novelty, operand constituent coverage, operation
schedule novelty, and carry/borrow run length. Require all individual local contexts
in the transition-combination family to be familiar and each ordered sequence unseen;
require familiar operands but unseen complete compositions in operand-combination tasks.
Reject the entire dataset if any designated eligibility condition fails. Do not silently
replace tasks, filter tasks based on neural outcomes, alter this frozen proposal or
weaken its gates. Any failed qualification requires a new reviewed protocol version.

The historical identity registry must normalize historical *public* inputs and include
opaque lockbox identities. Implemented compatibility covers V3 operand/operation and
R5.2 namespace digests (including commutative single-operation normalization); no lockbox contents may be
opened. Empty registries used in synthetic tests do not qualify scientific data.
No actual suite or sealed labels were generated during this work. Custodian generation
must commit input, reference and eligibility hashes; author receives only public OOD
inputs. The code's procedural reference is available to a custodian and synthetic
fixtures; its existence is not permission for the model author to inspect sealed labels.

## Metrics, comparisons and release gates

Replay requires complete condition × seed × task coverage, exact record schema,
integer trace lengths, matching prediction and label commitments and independent
seal-before-release attestation. Hash checks establish immutable content, not temporal
ordering: an external custodian must verify attestation/provenance. Scoring reports
UNVERIFIED until independent audit; it never declares PASS on its own.

Full-task exactness requires every operation state correct; final-answer exactness is
reported separately. Intermediate accuracy is over operation states, not digit columns;
local digit/carry correctness and predicted-carry feedback must additionally be retained
by the arithmetic adapter and independently audited before a scientific claim. Per-family,
per-seed and length-stratified metrics and every failed trace are preserved. Task-level
Wilson 95% intervals are per seed. Paired percentile bootstrap resamples identities after
averaging matched seeds within each identity (10,000 draws, seed 540052). Seeds never
multiply the independent task denominator. Missing/bad predictions stop scoring.

Proposed mandatory gates: mean full-task exactness ≥80%, every family mean ≥70%,
intermediate operation-state accuracy ≥90%, ≥10 percentage points over each of
feedforward/memorization/transition-lookup controls, positive feedforward gain for every
seed, and strictly positive paired-bootstrap lower bound. Comparisons must pass overall
and per family; all seeds and baseline failures remain visible. A perfect procedural
reference is a correctness reference only; its score is excluded from neural claims.
Any mandatory failure is NO_GO; independent audit is required for PASS. Scientific
model qualification and local-trace recording remain a prerequisite, not a completed fit.

## Concrete approval package

Review the immutable protocol/source lock and synthetic tests, then separately decide:

1. Adopt this new proposal after independent review of task eligibility and learned
   baseline capacity. No scientific fitting follows merely from adoption.
2. Approve custodian-only dataset qualification/sealing, including the historical identity
   registry and context certificate. Reject infeasible family constructions as DATA_NO_GO.
3. Review a final training/execution adapter and receipts for common initialization,
   matched target exposures, local-label loss, inactive-head handling and optimizer state.
4. Explicitly authorize six fits ×800 updates (4,800 scientific updates), serialized on
   the native runtime, with no tuning, extra fits or checkpoint selection.
5. Seal all raw predictions and arithmetic traces before independent label release and
   scoring; obtain independent evidence verification before any scientific PASS.

Missing these decisions leaves **UNRUN**, and Research Alpha generalization unsupported.
The current preparation receipt helper cannot authenticate approval documents. The
workflow requires an independently supplied receipt verifier and numerical fit callback;
none is provided or authorized for scientific use. Receipt hashes must be reviewed as references to independently
signed decisions, never interpreted as self-authorization.

## Engineering verification

24 synthetic tests pass under the existing final-venv Python: deterministic/disjoint
unlabelled input construction, identity relabeling, historical overlap rejection,
predicted-feedback controls, exact scoring denominators, uncertainty, failure retention,
tampered/missing/duplicate predictions, absent precedence attestation, boolean/invalid
labels, common initialization, near-matched capacity, temporal behavior, synthetic
recurrent gradients, fail-closed preparation, protected callback exclusion and
serialized failure preservation with fabricated fixture callbacks, exact-boolean
receipt verification and interruption sealing before KeyboardInterrupt/SystemExit
propagate. This is engineering evidence only.
