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
