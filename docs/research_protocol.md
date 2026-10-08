# Numerical-claim study protocol v1

Frozen config: `configs/experiments/research_study.toml`; frozen dataset:
`datasets/generated/research_v1`. Both implementation and protocol were committed
before held-out evaluation (`df98891`). Earlier results remain exploratory evidence.

## Information audit

| Condition | Model inputs | Labels/feedback at training | Inference feedback |
|---|---|---|---|
| Answer-only | Original operand digits/signs/lengths | Final answer digits/sign | None |
| Structured | Same operands plus static length/relation/goal representation | Final answer digits/sign | None |
| Mindscape B | Operand digits at current operation, previously self-proposed digit/carry, positions, goal/relation flags | Correct next numerical claim from training trajectories | Own written values/state only |
| Mindscape C | Same state features as B | Scalar acceptance/rejection of self-generated training proposals; accepted own values replayed | Own written values/state only |

No model feature contains expected arithmetic values, target answer, future actual
states, or corrected arithmetic. Primary execution **writes the proposed number even
when incorrect**. The model never sees the numerical verifier judgment during inference.
After an episode the common evaluator checks every claim and final goal privately.
Regime C training has intentional scalar correctness feedback; its query count is logged.
The verifier and target generator alone can evaluate the correct local arithmetic.
Hypothetical future states are self-generated, explicitly tagged and never inserted
into working/episodic observation memory as actual evidence.

All conditions use 52 inputs -> 64 tanh units -> eight ten-class heads plus sign,
8,722 parameters, same seeds, nested problem IDs, 600 Adam updates, batch 64 and
learning rate .003. Fixed final update is used for every major condition: validation
labels/diagnostics do not select checkpoints. C receives no validation trace labels.
No hyperparameter selection occurs after viewing IID/OOD results. The existing
82.0% exploratory OOD result is not transferred to this benchmark.

## What this comparison can and cannot isolate

It removes corrected-arithmetic inference tools, singleton answer masks and capacity
mismatches. The structured reduction to local multiplication/carry claims and automatic
schedule are hand-specified procedural knowledge, not learned planning. A/B represent
that knowledge differently: the intervention is representation/decomposition plus
training signal, not proof that JSON or autonomous cognition causes an advantage.
The concept seed describes decimal multiplication for every condition; task MLPs
consume its fixed operation/goal encodings, not natural-language understanding.
State features are functions of original observations and the learner's own previous
claims. Supervision count and environment queries differ by training regime; sample
budget comparisons must include those costs. Groundedness structurally requires a
trace, so baselines lacking traces cannot establish it even if answers are correct.
No memory or dream mechanism may consult held-out targets. Compute is recorded,
not matched. These remaining differences must accompany every interpretation.

## Action modes and numerical validation

Constrained: 100 syntactically legal numerical candidates (0..99), with no correctness
mask. Unconstrained primary: those 100 candidates plus an invalid skip candidate.
Every decision logs candidate count, proposed value, actual written result and a
post-episode verifier judgment. A wrong arithmetic claim remains in the executed
trajectory; no early correctness rejection rescues inference. Syntax-invalid actions
terminate. Final verification checks both numerical rules and observation binding.

## Regime C and replay

Exploration is bounded at 64 attempts per training problem. Each state is explored with
75% random untried candidates and 25% model scoring. Training rejection leaves the
state unchanged, returns only false, and does not reveal the expected number. Success
advances using the learner's own accepted number. SQLite records all state/action/
event/result/next-state/reward attempts in the train partition. Positive-feedback
replay supplies pseudo-labels equal to accepted learner proposals. No complete
reference trajectory or reference answer is consumed. This is batch off-policy
self-generated feedback learning, not general reinforcement learning or autonomous
concept discovery. Model weights are updated after collection; exploration during
that collection uses initial weights. A no-replay online variant updates each accepted
experience once; its update count/optimizer differ and must be reported as a method
contrast, not a clean matched-compute causal memory effect.

## Simulation and ablations

Generic TransitionModel and tagged bounded DreamEngine accept deterministic or learned
transition adapters. Primary numerical dreams branch three candidates at depth two;
claim-write dynamics predict hypothetical states without a numerical oracle. They
score model confidence, not ground-truth correctness. The standalone integer-navigation
demo demonstrates explicit goal-distance scoring. Simulation is a deterministic
prototype; a separately trained predictive world model is not implemented.

At budget 50, three seeds compare no state (retrained), no dream, constrained mode,
no actual environment evidence (open-loop recurrence), serialization/distractor
perturbation and online learning without episodic replay. Relation is constant in a
single-operation domain, so graph attribution is not meaningful. External goal
verification defines the common metrics and cannot be removed as an evaluator
ablation. B versus C already varies trajectory supervision; it is not a single-component
ablation. Single-budget ablations do not estimate data-efficiency changes.

## Statistics and interpretation

Primary curves use three seeds at all seven budgets. Report means/sample standard
deviations separately for IID/OOD and compatible conditions. Exact threshold crossing
uses measured points only; censored N*/DER stays not_reached. Wilson intervals describe
individual-run binomial accuracy, not cross-seed variance. Exploratory paired McNemar
tests use one preregistered comparison budget (1000), seed 0, and Holm correction over
six tests; do not pool repeated seeds as independent examples. Significance does not
establish practical importance or isolate all remaining architectural differences.
The generated development research set is not an independently audited final thesis
benchmark. Every failure and run directory is retained. No performance target is promised.
