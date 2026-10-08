# Research-core milestone: controlled numerical-claim study

## IMPLEMENTED
An additional numerical-claim environment and backend, preserving all previous phases.
Identical-capacity answer-only/structured baselines and numerical-claim Mindscape B/C;
non-singleton constrained/unconstrained execution, scalar-feedback experience collection
and positive SQLite replay, model-visible/oracle separation, bounded tagged simulation,
optional offline pretrained encoder/head backend, four-condition study orchestration,
component interventions, robustness analysis, statistics, fairness audit and SVG plots.

## TESTED
All prior tests retained. Added numerical-claim oracle cases, wrong-value non-correction,
100 candidates, target-mutation feature isolation, equal capacity, three-way evidence
records, no Regime C target access, non-mutating dream rollouts, generic learned-transition
adapter, action modes, open-loop evidence removal, serialization/distractor invariance,
all-condition training/reload/shared evaluation, optional offline backend contract (mocked),
final-attempt completion counters, statistical utilities and invalid configuration rejection.

## VERIFIED
90 trained checkpoints; 204 common-runner evaluations: 168 primary and 36 component/robustness
evaluations. Complete primary matrix: four conditions × seven budgets × three seeds × two splits.
All predictions/configs/checkpoints/failures retained locally. 17 artifact-based integrity
checks passed; primary candidate count is 101, constrained count is 100. No arithmetic
correction channel or model-visible inference verifier judgment is present.

## MODEL(S) USED
Local task-specific NumPy tanh MLPs. The prior model remains intact. Optional offline
LocalHFEncoder/FrozenPretrainedBackend supports pretrained text features plus a trainable
head and explicit download opt-in; no actual pretrained weights were trained/evaluated
in this study because PyTorch/Transformers are absent. Its offline interface is tested
with a mock encoder. No model download, paid API or foundation-model training occurred.

## MODEL SIZES
All four primary conditions: 52 inputs, 64 tanh units, eight ten-class output heads plus
sign; **8,722 trainable parameters** each. Policy uses first two heads for proposed
unit/carry and other heads as zero dummy labels. Exact capacity is matched, but usable
label semantics/task difficulty differ. Same model family, initialization policy and optimizer.

## TRAINING REGIMES
A: final-answer supervision. Structured baseline: static structured operands -> answer.
B: local numerical claims supervised by correct training transitions. C: bounded exploratory
proposals -> scalar accepted/rejected feedback -> replay of accepted own labels, with no
reference answers/trajectories read. Fixed 600 Adam steps, batch 64, learning rate .003;
validation is diagnostic only, not checkpoint selection. Online/no-replay intervention
uses one gradient update per accepted proposal and is not a matched-compute memory ablation.

## DATASET SIZES
Research v1 (new frozen dataset): train 1000, validation 100, IID test 200, OOD test 300.
Nested budgets: 10,25,50,100,250,500,1000. Train structures: 1x1/2x1/2x2; OOD:
3x2/3x3/4x2. Canonical operand pairs are disjoint, including swapped pairs. Finite 1x1
family exhausts in train; actual per-family/carry distributions remain in difficulty reports.

## SEEDS
Dataset seed 4242; training/experience seeds 0,1,2 for every primary budget/condition.
All interventions use three seeds at budget 50. All conditions share exact subset IDs.

## ACTUAL RESULTS
Observed performance is low. None of the requested accuracy/OOD/groundedness/goal or
data-efficiency targets is met. The largest primary IID score of any single run is 8%.
The prior 82% exploratory OOD result is **not reproduced** under numerical-claim controls.
No architecture-only superiority or 2×/5× data-efficiency conclusion is established.

## IID RESULTS
Mean ± sample standard deviation across three seeds, percentages:

| Budget | Baseline A | Baseline B | Mindscape B | Mindscape C |
|---:|---:|---:|---:|---:|
| 10 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.289 | 0.167 ± 0.289 |
| 25 | 0.333 ± 0.289 | 0.000 ± 0.000 | 2.333 ± 0.764 | 0.667 ± 0.289 |
| 50 | 0.500 ± 0.000 | 0.500 ± 0.000 | 3.333 ± 0.577 | 1.333 ± 0.577 |
| 100 | 1.000 ± 0.000 | 0.833 ± 0.289 | 4.000 ± 0.000 | 2.333 ± 0.764 |
| 250 | 0.333 ± 0.289 | 0.500 ± 0.000 | 4.333 ± 0.577 | 4.000 ± 1.000 |
| 500 | 0.333 ± 0.289 | 0.333 ± 0.289 | 4.000 ± 0.866 | 5.500 ± 2.291 |
| 1000 | 0.167 ± 0.289 | 0.333 ± 0.289 | 6.333 ± 1.443 | 6.000 ± 1.732 |

At budget 1000, Mindscape B groundedness/goal success = 6.167% ± 1.155%;
C = 5.667% ± 1.443%. Answer baselines have no supporting trajectory: groundedness
and goal success are 0%, supplied-trajectory validity is null. For policy conditions,
full-trajectory validity equals grounded/goal rates in these runs. Some wrong claims
cancel to a correct final answer, so exact accuracy can exceed groundedness.
Unsupported rate is the complement of groundedness. Every metric remains in aggregate.json.

## OOD RESULTS
All primary conditions/budgets/seeds have 0% exact OOD accuracy and 0% grounded/goal
success on the observed 300-example OOD set. This does not mean a universal probability
of zero; individual-run Wilson intervals are stored in statistics.json. Per-structure
and carry counts are reported in difficulty_failures.json. No held-out test tuning occurred.

## DATA-EFFICIENCY RESULTS
N*(.80), N*(.90), N*(.95) and both baseline-to-Mindscape DERs are **not_reached**
on IID and OOD. No interpolation or fabricated crossing. Ablation data-efficiency
changes are not estimated from single-budget interventions.

## ABLATION RESULTS
Mean IID accuracy changes at budget 50, percentage points (three seeds):

| Intervention | Reference | ΔAccuracy | ΔGroundedness | ΔGoal success |
|---|---|---:|---:|---:|
| constrained | trajectory | 0.000 | 0.000 | 0.000 |
| no_dream | trajectory | -0.333 | -0.167 | -0.167 |
| no_environment_interaction | trajectory | 0.000 | -2.833 | -2.833 |
| no_episodic_replay | experiential | -1.333 | -0.500 | -0.500 |
| no_state | trajectory | -3.333 | -2.833 | -2.833 |
| serialization_distractor | trajectory | 0.000 | 0.000 | 0.000 |

All OOD intervention changes are 0 percentage points. No-interaction preserves
model-computed answers while removing actual evidence; the groundedness drop follows
the metric definition, not proof of an arithmetic advantage. Serialization/distractor
invariance is a deterministic feature-adapter property, not semantic distractor reasoning.
Constant relation representation and mandatory external goal evaluator are not meaningful
isolated ablations in this domain. B versus C varies trajectory supervision by construction.
Reasons are recorded in ablations.json. The online/no-replay contrast also changes optimizer
and update count; it cannot isolate memory causally at equal compute.

## EXPERIENTIAL-LEARNING STATUS
Implemented limited batch off-policy self-generated feedback learning. At budget 1000,
accepted replay labels are 738/874/1074 at seeds 0/1/2, versus 4506 supervised transitions
per B run. Query budget: up to 64 per problem; exact counts are retained. No complete
expert trace is delivered to C. Exploration is collected with initial weights, then
positive feedback is replayed; no claim of online adaptive exploration or autonomous
concept discovery. Sparse feedback and limited capacity are potential limitations, not
experimentally isolated explanations for poor performance.

## DREAM-ENGINE STATUS
Runnable generic bounded branching with explicitly hypothetical states and deterministic
or learned-callable TransitionModel adapters. Primary claim dreams use proposed-write
dynamics, depth 2/branching 3, and learned confidence scoring without a numerical oracle.
Goal-distance scoring is demonstrated separately in examples/dream_goal_demo.json: real
state remains 0, goal is 3, selected first action is +1. A separately trained transition
world model is not implemented. Simulated values never become observed evidence.

## HISTORICAL-REFERENCE STATUS
Primary-source GPT-4/Gemini report records and exact prompting caveats are documented
in docs/historical_references.md. GSM8K word problems are not comparable to our explicit
integer multiplication task; historical checkpoint IDs are undisclosed and training
budgets are not controlled domain sample counts. No victory claim or historical DER.

## FAIRNESS CHECKS
17 computed checks passed: dataset integrity/disjointness, training IDs, equal parameter
count, dataset hashes, balanced matrix, shared optimizer configuration, no target reads
in C, no inference arithmetic-oracle calls, target-mutation isolation, >1 candidates,
no inference correctness feedback, own-proposal actual values, hypothetical tags and
train-only memory. This removes major earlier confounds, but hand-specified decomposition,
different label/query counts, compute and trace accessibility remain explicit differences.
No claim that architecture alone has been isolated. Exploratory paired McNemar tests
at budget 1000, seed 0, after Holm correction: B vs A IID p=0.0001831, C vs A IID
p=0.0390625; all other comparisons p=1. These within-seed tests concern small absolute
IID differences versus a weak baseline and do not establish practical success or OOD benefit.

## TRAINING TIME
32.965690 seconds across 90 trained primary/intervention models; actual per-run metadata retained.

## INFERENCE TIME
114.325489 seconds across 204 evaluations; each covers 200 IID or 300 OOD examples. Includes model execution and evaluator assessment.

## TEST COUNT
86 passed on Python 3.12.14 / NumPy 2.3.5; original tests preserved. Real optional
PyTorch/Transformers integration, pytest/Ruff/MyPy and hosted CI were not run.

## KNOWN LIMITATIONS
All primary OOD runs fail; IID success is far below targets. Task MLPs are small and
not pretrained; optional pretrained interface was only mock-tested. The procedure/control
schedule is hand-authored and narrows the scientific question to decomposed numerical
learning. Goal/relation are fixed domain signals, not variable-goal or graph reasoning.
Label/query budget and inference compute differ; replay uses only accepted labels and
does not optimize rejected-action likelihood. Baselines cannot meet trace-groundedness
without a trace emitter. Scalar C feedback uses a correctness oracle at training only,
an intentional experimental condition. No final thesis conclusions or measured 2× DER.

A bookkeeping edge case undercounted completed C episodes in two original metadata
files (1000/seed0: 23 vs stored 24; 1000/seed2: 76 vs stored 77). The audit retains both
values; completion reporting here uses stored outcomes. Future code counts final-attempt
completion correctly, with a regression test. Labels, weights and evaluation metrics
were unaffected. Original experiment files were not replaced. Analysis-only chart drafts
are retained; committed figures use corrected cost labels and uncertainty bars.
Local results occupy approximately 1.0 GiB, including replay/prediction detail. Checkpoints
and full predictions stay local under ignored results/; measured summary snapshots are
committed. Timing excludes dataset preparation and artifact I/O for evaluation; training
includes experience collection/replay and fitting. RAM size was unavailable. PNG rendering
is absent, so figures are standalone SVG vectors. Claims adapter is currently domain-specific.

## GIT STATUS
Branch main; earlier code/history/results preserved. Final clean status checked after commit.

## LATEST COMMIT
Protocol/run revision: df98891. Analysis/backend/diagnostics: 571147b. Final report and
measured snapshots are committed separately; latest hash is reported in chat.

## REPOSITORY TREE
```text
src/mindscape/environments/multiplication/claims.py
src/mindscape/data/claims_backend.py
src/mindscape/models/{study,study_encoding,pretrained}.py
src/mindscape/training/claims.py
src/mindscape/reasoning/simulator.py
configs/experiments/{research_dataset,research_study}.toml
datasets/generated/research_v1/{manifest.json,*.jsonl}
scripts/{train_research,evaluate_research,run_research_study}.py
scripts/{analyze_research_study,audit_research_study,run_dream_demo}.py
docs/{research_protocol,historical_references,milestone_04}.md
experiments/research_claims_v1/{aggregate,data_efficiency,ablations,statistics,...}.json
experiments/research_claims_v1/plots/*.svg
results/research_claims_v1/<condition_budget_seed>/{config.yaml,metadata.json,checkpoint,training.jsonl}
results/research_claims_v1/evaluations/<id>/{config.yaml,metrics.json,predictions.jsonl,failures.json,summary.md,plots/}
examples/dream_goal_demo.json
tests/test_research.py
```
