# First learned Mindscape milestone

## IMPLEMENTED
Small trainable NumPy backend; answer-only baseline (A); trajectory-supervised
state/history-conditioned policy (B); environment-interactive inference; action masking
and unmasked diagnostics; resettable working memory; partitioned SQLite episodic API;
prefix/final verification; deterministic Adam training, validation checkpoint selection;
save/load and CLI; shared evaluator integration; tracking, failures and preliminary curves.

## TESTED
All original 54 tests plus 15 learning tests passed. Coverage includes numerical gradient
checks, optimizer learning/reproducibility, regime adapters, serialization, valid masks,
memory reset, illegal actions, timeout/step limits, trajectories, verifier integration,
configuration, checkpoint reload, SQLite partitions and train-to-evaluation smoke runs.

## VERIFIED
10 actual trained checkpoints were reloaded before evaluation; 30 learned evaluations
used identical held-out splits. Checkpoint/dataset/budget/regime compatibility is checked.
The trained unmasked policy executes 19×3 to 57 with five legal actions and a complete
verified trajectory. The baseline checkpoint predicts 28 for this separate demonstration.
Each selected action, actual event/result and before/after state is retained in
`examples/19_times_3_learned.json`. Working memory resets and no episodic retrieval is used.

## EXPERIMENTS RUN
50 examples: baseline and policy at seeds 0,1,2. 100 and 250 examples: each at seed 0.
Each model evaluated on 500 IID and 500 OOD examples; policy also evaluated unmasked.
Two additional untrained unmasked controls (seed 0) reached 0% IID and 0% OOD success.
One early policy-training development smoke run is retained separately under results/.
No hyperparameters or test sets were changed after the fixed comparison started.

## ACTUAL RESULTS
All values below are measured preliminary development results, not final thesis findings.
At 50 examples: mean ± sample standard deviation across three seeds; other sizes: one seed.

| Training examples | Model | IID exact accuracy | OOD exact accuracy |
|---:|---|---:|---:|
| 50 | baseline | 0.267% ± 0.306% | 0.000% ± 0.000% |
| 50 | mindscape | 100.000% ± 0.000% | 100.000% ± 0.000% |
| 50 | mindscape_unmasked | 100.000% ± 0.000% | 82.000% ± 4.084% |
| 100 | baseline | 0.400% | 0.000% |
| 100 | mindscape | 100.000% | 100.000% |
| 100 | mindscape_unmasked | 100.000% | 76.400% |
| 250 | baseline | 0.400% | 0.000% |
| 250 | mindscape | 100.000% | 100.000% |
| 250 | mindscape_unmasked | 100.000% | 78.800% |

Policy supplied-trajectory validity is 100% for both masked and unmasked runs:
illegal proposals are rejected, so failed episodes retain valid incomplete prefixes.
Groundedness and goal success equal each policy run’s final accuracy; unsupported rate
is its complement. Baseline groundedness and goal success are 0%, unsupported rate 100%,
and trajectory validity null (no supporting trace). All five trained policies achieved
100% teacher-forced unmasked validation action accuracy. Seed-0 policy validation loss
at 50 examples fell from 1.409832 to 0.002929.

For the unmasked 50-example OOD policy: 106/67/97 invalid-action failures at seeds
0/1/2 respectively. Full taxonomy counts and every prediction are saved. No DER or
architecture-superiority conclusion is drawn. Masked 100% is guaranteed by one legal
action per state. The policy delegates exact arithmetic to the environment; the
baseline learns answer digits without that tool. Even unmasked action success does
not establish learned arithmetic or a fair data-efficiency advantage.

## NOT IMPLEMENTED
Regime C feedback learning (interface only), pretrained language-model integration,
PEFT/LoRA, memory-based learning/retrieval experiments, generated explanations, final
statistical studies, matched-tool controls, multiple-choice environment redesign, UI,
multimodality, dream engine and historical GPT/Gemini comparison.

## KNOWN LIMITATIONS
Singleton masking and exact arithmetic access confound comparison. The baseline is
very weak; failure is retained and cannot support general conclusions. More transition
labels per policy problem and differing heads mean supervision/capacity are unmatched.
Validation policy labels use correct histories (teacher forcing); OOD rollout errors
expose different execution behavior. Development dataset is unbalanced and not a final
untouched research set. Only 50-example points have three seeds. Training uses a fixed
400-update budget rather than equal epochs. The learned adapter is multiplication-only;
core/benchmark remain general. Features support six-digit operands, eight-digit answers.
TensorBoard and gradient-framework tooling are absent; plain JSON logs are used.
Inference checkpoints do not preserve resume-training Adam state. Optional learning tests
skip without NumPy. Floating point/RNG may differ across library/CPU versions.
RAM could not be queried; PNG chart rendering is unavailable, so standard ReportLab SVG
curves are delivered. Ruff/MyPy/pytest and hosted CI were not run.

## MODEL USED
Local randomly initialized task-specific NumPy tanh MLP, not a foundation model.
No downloaded model, API, pretrained weights or paid dependency. See docs/modeling.md.

## MODEL SIZE
Shared backbone: 52 inputs, 64 hidden units, 3,392 parameters. Baseline: 8,722 total;
policy: 3,652 total. Baseline eight decimal digit heads plus sign; policy four action logits.

## DATASET SIZES
Development v1: train 1000, validation 250, IID test 500, OOD test 500.
Learning budgets 50/100/250 use identical recorded nested subset IDs for each model.
Policy row counts: 50 examples → 376 transitions, 100 examples → 740 transitions, 250 examples → 1783 transitions.

## SEEDS
Dataset seed 42, fixed subsets. Training/mini-batch seeds 0,1,2 at budget 50;
seed 0 at budgets 100/250; untrained control seed 0.

## TRAINING TIME
Ten fixed-comparison optimization runs total 0.553455 seconds.
Per-run seconds (optimization plus validation only):

| Regime | Budget | Seed | Seconds |
|---|---:|---:|---:|
| answer_only | 50 | 0 | 0.076317 |
| trajectory_supervised | 50 | 0 | 0.035391 |
| answer_only | 50 | 1 | 0.074609 |
| trajectory_supervised | 50 | 1 | 0.035283 |
| answer_only | 50 | 2 | 0.074834 |
| trajectory_supervised | 50 | 2 | 0.035265 |
| answer_only | 100 | 0 | 0.077495 |
| trajectory_supervised | 100 | 0 | 0.034872 |
| answer_only | 250 | 0 | 0.074412 |
| trajectory_supervised | 250 | 0 | 0.034979 |

## INFERENCE TIME
Thirty learned evaluations total 20.205644 seconds, including prediction/assessment.
Each run covers 500 examples. Full per-run times and calls remain in
`experiments/first_learned/curves.json`; model calls are 500 per baseline run and
3,944 IID / 6,635 OOD per complete policy run. Failed unmasked OOD runs use fewer calls.

## TEST COUNT
69 passed on Python 3.12.14 / NumPy 2.3.5.

## GIT STATUS
Branch main; prior history preserved. Final clean status checked after committing.

## LATEST COMMIT
Training protocol committed before fitting: f33ce57; model implementation: ef3d94e.
This report and measured snapshots are included in the final milestone commit shown in chat.

## REPOSITORY TREE
```text
src/mindscape/models/{base,encoding,numpy_backend,learned}.py
src/mindscape/training/{adapters,optimizer,run}.py
src/mindscape/memory/{working,episodic}.py
src/mindscape/evaluation/{runner,evaluator,schemas}.py
configs/experiments/learning.toml
scripts/{train,evaluate,run_learned_episode,run_learning_experiment,plot_learning_curves}.py
docs/{modeling,training,experimental_protocol,milestone_03}.md
examples/19_times_3_learned.json
experiments/first_learned/{curves,failure_analysis,fairness,untrained_controls}.json
experiments/first_learned/{test_accuracy,ood_test_accuracy}.svg
results/learned_development_v1/<model_budget_seed>/
  config.yaml, config.json, metadata.json, metrics.json, training.jsonl, checkpoint/
results/learned_development_v1/evaluations/<unique_id>/
  config.json, metrics.json, predictions.jsonl, summary.md
results/learned_development_v1/{comparison,curves,failure_analysis,19_times_3_learned}.json
results/learned_development_v1/controls/<unique_id>/
tests/{test_foundation,test_benchmark,test_learning}.py
```
