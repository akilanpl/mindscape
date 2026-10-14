# Native R5.6 training infrastructure

**EXPERIMENTAL — GENERALIZATION NOT YET DEMONSTRATED**

Implemented preparation generates deterministic bounded synthetic primary tasks
for three domains: decimal single-digit addition/carry and subtraction/borrow, AST rename dependency
examples with an explicitly requested candidate, and two-dimensional force
updates. Seeds are 17/29/43; exposure budgets are 32/128/512; Adam budget is 2,000
steps, batch size one, CPU. Model agent supplies the actual typed graph R56Model
and pack_problem interfaces; trainer uses original model logits/deltas directly,
never symbolic correction before loss. Labels are separate from predictor inputs.

Train/dev/test problem hashes are disjoint (math/physics assigned by hash bucket;
coding names are split-specific). Repeated problems may occur within a split.
The splits are bounded IID families, **not OOD or generalization evidence**.
Paraphrase-family labels are split-specific metadata only: a language paraphrase
benchmark is not yet implemented. Extension fixtures document borrow, impulse,
equation update and multibody collision; equation/multibody neural heads are
unsupported and these fixtures are not included in primary training claims.

## Freeze and external authorization

```sh
PYTHONPATH=src python -m mindscape.r56.learning freeze --root "$PWD" --output work/native-freeze
```

Preparation writes full deterministic datasets, their SHA256 identity, source
locks for every r56 and mind1 Python module (including language/runtime/inference), protocol hash, and clearly UNISSUED artifact
templates. Generate a freeze only after final model code is integrated. Freeze
must be checked into review evidence before any approved campaign. No generated
artifact confers permission or a scientifically qualified flag.

`train(...)` validates source/data/protocol hashes and **two independently signed
external artifacts before importing torch or creating a model/optimizer**.
Authorization and qualification must have different trusted issuer keys; signatures
are Ed25519 over canonical JSON excluding signature_hex. Trusted issuer public
keys must come from the external operator's established trust store, never this
agent, a model output, or a generated local template. Signature verification needs
optional `cryptography`; if absent, the gate fails closed. This API cannot
establish real-world signer independence by itself; trust-store governance is an
external requirement. No authorization is issued here and no optimizer step ran.

Original protected historical R5.4/OOD/cross-domain studies retain their full
workloads and independent approval requirements. This new bounded campaign does
not replace or reduce them.

## Actual training and checkpoint path

After externally approved gate validation only: deterministic torch seeding,
shared graph forward pass, domain cross entropy/MSE, backward pass, Adam step,
loss evidence and finite-loss rejection. Samples cycle deterministically. JSON
tensor checkpoints record source/data-bound freeze, seed, budget, control, steps,
losses, parameter count, wall seconds, torch version and status. Failure evidence
records completed steps and exception type. Checkpoints are safe JSON, not pickle;
loading requires an expected SHA256, exact tensor keys/shapes/dtypes and finite
values, then strict state loading/eval mode. Trained checkpoints still explicitly
set `scientifically_qualified=false`; separate independent evaluation is required.
Optimizer state/resume is not implemented; replay starts from the frozen seed.
CPU deterministic replay is intended but not yet demonstrated through training.

## Controls and statistics

Exact-rule and exact-task memorization controls use no neural training. MLP pooled
node and GRU node-sequence controls share output heads but discard edges, types
and concepts; their information/parameter mismatch must be reported and matched
before a scientific comparison. Graph controls use the original graph architecture:
concept-on, concept-off (empty IDs), shuffled (fixed invertible 32-ID permutation),
and ablated (all IDs zero while retaining count). Controls are distinguishable
interventions, not demonstrated improvements. No parameter/compute matching is
claimed simply from identical step counts.

Paired bootstrap confidence intervals and Holm correction helpers are implemented.
Callers must aggregate dependent samples to independent problem families before
resampling. No p-values, significance, learning curves or transfer success are
invented. Record original predictions, coverage, verifier acceptance separately,
held-out metrics, model parameters, runtime and peak memory in actual approved
runs. Tensor-storage and process-lifetime peak RSS accounting are implemented; isolated
campaign activation peak and automated independent qualification remain pending.

Current nontraining evidence: curriculum replay, disjoint split identities,
label/rule agreement, content locks, unissued/forged approval rejection,
pre-neural-import gate rejection, memorization abstention and statistical fixtures.
No scientific training or model optimizer execution was performed.

## Additional zero-update verification

With PyTorch 2.14.1, nine nontraining tests pass. Actual graph forward passes,
domain losses and backward connectivity produce finite nonzero shared gradients;
state-dict weights remain bitwise unchanged. Adam construction is blocked in the
connectivity test. Safe checkpoint load reproduces original stateless predictions
bitwise; zero optimizer updates are recorded. MLP/GRU output dimensions and
concept intervention consistency are checked. These are architecture tests, not
training experiments or demonstrations of learning.

Measured default parameter counts: graph 44,962 (179,848 parameter bytes), pooled
MLP 1,902 (7,608 bytes), pooled GRU 9,822 (39,288 bytes). Graph concept variants
share the same architecture and parameter count. Pooled baselines are explicitly
**not parameter matched**; scientific comparison must match these budgets before
claiming an advantage. Accounting includes trainable/total parameters, tensor
storage, observed gradients and estimated Adam moment storage (no Adam created
for the measurement). No trained weights or performance claims were generated.

## Frozen matched-capacity candidates

The success-comparison candidates now use shared observable input tokens containing
all node features/types, edge endpoint indices/relations, action, domain and explicit
concept IDs. No labels enter tokens. MLP pools tokens; basic RNN and GRU read the
observed sequence. These architectural bottlenecks remain part of the comparison;
parameter parity does not establish equal FLOPs or latency. The earlier eight-feature
pooled MLP/GRU remain legacy informational controls, excluded from success comparisons.

Widths are selected solely by deterministic parameter arithmetic before seeing any
loss or performance: MLP163 and basic RNN163 each have 45,022 parameters; GRU85 has
44,744, versus graph44,962. Gaps are 0.134%, 0.134%, 0.485%, below the frozen5% limit.
Actual constructors verify these counts. All graph concept variants share capacity.

The preparation manifest freezes eight neural controls: matched MLP/RNN/GRU,
graph concept-on/off/shuffled/ablated, and graph recurrent memory. Concept-on uses
stateless memory; the recurrent control carries detached hidden state in frozen
sample order, resetting at run/evaluation start. The primary comparison has three
seeds × three exposure budgets × eight neural controls = **72 runs**, each with
**2,000 optimizer steps**, totaling **144,000 steps** if externally authorized.
Deterministic/memorization controls have no optimizer. No original historical
experiment workload is replaced or reduced by this proposal.

Proposed qualification thresholds are frozen in the preparation manifest:
classification target90%, guided advantage at least5 percentage points at budgets32
and128, paired95% CI lower bound above0 with Holm alpha.05; physics is separate,
absolute error tolerance.1 in stated units and MSE target.01. These are candidate
criteria requiring independent preregistration review/power analysis, not approved
claims or measured outcomes. Composition-depth, OOD and cross-domain-transfer axes
are designated preparation only; full datasets and independent qualification for
those studies are not implemented or implied by the IID curriculum.

Current additional evidence: ten tests pass, including matched constructor counts,
observed action/domain/concept sensitivity, stateless deterministic replay and
RNN/GRU hidden-state effects. No optimizer was constructed or stepped in these tests.

## Exposure and evidence integrity

Budgets32/128/512 are **sample exposures**, not promises of that many distinct
problems. The finite digit-state space causes within-split repeats. Freeze now
records per-seed/per-budget actual domain exposures, distinct problem identities
and repeated exposures. All candidate controls receive the identical ordered
prefix and repeated training cycle; optimizer steps do not create new examples.
This limitation must accompany any data-efficiency claim.

Source locks additionally cover tests_native evaluation/qualification tests,
configs/r56_native JSON and scripts/native Python evaluators (plus native-named
scripts). Altering any of these invalidates signed qualification. Freeze marks
success thresholds as candidates requiring independent review. Training refuses
existing checkpoint or failure destinations; tensor checkpoints and failure
records use exclusive creation to preserve previous evidence.

Current evidence: twelve nontraining tests pass, including repeated-exposure
reporting and rejection of checkpoint/failure overwrites.
