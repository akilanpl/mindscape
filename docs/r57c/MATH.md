# Focused math correction: both candidates rejected

Status **NO DEVELOPMENT IMPROVEMENT; NOT ADOPTED**. Old R5.7 and R5.6 sources,
checkpoints and final evaluation were untouched. Only old development data was
used. All nine original R5.7 initial/final checkpoints were inspected on80
balanced unseen development math examples per seed, with raw digit frequencies,
true-class margins, per-example gradient norms and incoming carry recorded in
`evidence/math/r57-all9-dev-audit.json`.

| Seed | Condition | Old R5.7 initial → trained dev exact |
|---|---|---|
|17|on / off / GRU|2.5→2.5% /2.5→3.75% /2.5→5%|
|29|on / off / GRU|7.5→10% /7.5→10% /2.5→10%|
|43|on / off / GRU|8.75→6.25% /8.75→6.25% /3.75→7.5%|

Two new corrections were tested, preserving byte-identical initial weights of
all old core/coding/physics parameters. Math bypasses the shared core entirely:
this removes math-to-core gradients and changes cross-domain interference, which
is an explicit confound, not a neutral head-only comparison. Coding and physics
use the original preparation/forward path. Primary data remains exact legacy128
records, mixed43math/43coding/42physics. Every condition receives2000 Adam.001
updates, batch1, seed17; two candidates ×threeconditions = **12000 actual CPU
engineering updates**, separately counted from MPS campaign budgets. No additional
candidate fit was run after this cap.

Candidate1: operation-specific trainable affine latent from raw left/right/incoming,
then learned scalar MLP digit decoder and carry head. Candidate2: same learned
latent with fixed periodic Gaussian digit lattice centers (digit+10k) and learned
precision/carry head. Candidate2 has an explicit decimal modular-lattice prior;
it is **not prior-free learned arithmetic**. Neither calls modulo/floor/reference
answers in prediction or inserts target features. Affine weights initialize from
Normal(.5,.1), not true arithmetic coefficients. The periodic decoder is nonconvex
and local phases can produce wrong learned arithmetic.

| Candidate | On initial→trained | Off initial→trained | GRU initial→trained |
|---|---|---|---|
|Scalar MLP|5→5%|5→5%|5→6.25%|
|Periodic lattice|7.5→5%|7.5→5%|6.25→6.25%|

The tiny gain for one MLP condition does not establish improvement across conditions
or seeds; both candidates are rejected for adoption. Borrow weights remain
positive rather than learning subtraction; a shared scalar also constrains two
heads in an unhelpful way. There is no evidence this correction transfers a learned
rule. Do not launch native fits with this module as a claimed improvement.
`make_model(condition,seed)` exists for reproducibility of the rejected lattice
candidate only. Original coding/physics remain usable through wrappers but these
experiments do not justify integrating a math replacement.

Evidence: `outputs/r57-focused-correction-v1/evidence/math/development-probe-12000.json`
contains all initial/trained dev predictions, coefficients and exact update counts.
The separate all9 audit executes backward measurements but zero optimizer updates.
One unit test passes for original-state preservation, operand sensitivity and
trainable connectivity. Passing that test is not learning success. Exact deterministic
arithmetic remains a separate transparent product reference.

## One separately authorized final ordinal-output probe

A final fixed candidate uses operation-specific learned affine weights on raw
left/9, right/9, incoming and constant1. Weights initialize Normal(0,.1), not the
true rule. The sole math target is a bijective ordinal recoding of existing
labels: (digit+10carry)/20 for addition, (digit−10borrow)/20 for subtraction.
This supplies no new ground-truth information, but **explicitly changes the loss
and encodes decimal output geometry**. Prediction decodes the learned scalar
against fixed20 joint digit/carry centers with Gaussian scores and marginalizes
to the usual logits. No arithmetic solver is applied to input operands.

The fixed three-condition probe ran exactly2000 full-mixed Adam.001 updates each,
seed17, same original128 records: **6000 additional CPU updates**. Combined with
the rejected earlier probes, this workstream executed18000 CPU engineering updates.
No more trials were run. Initial/trained development exact accuracy was3.75→13.75%
for all three conditions, versus old trained R5.7 seed17 on2.5%, off3.75%, GRU5%.
These are observed gains on80 unseen development cases, with no new final access.

Decision: **retain as an explicitly structured ordinal-output-prior candidate**
for parent review; not a qualified improvement or native-MPS result. Accuracy
remains poor, arithmetic coefficients remain far from the full true rule, and
one seed cannot establish generalization across seeds. The candidate bypasses
shared-core math gradients, preserving initial coding/physics/core weights but
changing task interference. All three conditions receive the same output prior.
No unconstrained neural arithmetic or general reasoning claim is justified.

Raw evidence: `evidence/math/ordinal-development-probe-6000.json` contains all
initial/trained predictions, learned coefficients and6000 update accounting.
`ordinal_model.make_model(condition,seed)` and its custom `domain_loss` must be
used together: math uses scalar ordinal MSE; other tasks keep the old loss.
One additional unit test passes for preserved old initial weights, gradient
connectivity, removed shared math gradient and output shapes.
