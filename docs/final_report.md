# Final build completion report

## IMPLEMENTED

Stronger frozen local backbone adaptation, categorical task representation, four regimes, procedural100-candidate execution, posthoc verification, working/episodic/concept memory, bounded dreams, generic environment protocol, immutable result roots, statistics/failures/ten figures and an evidence UI.

## AUDITED

Historical encodings, targets, losses, optimization, splits, initialization/decoding, information and cost; executed sanity checks; final438-check artifact audit.

## FIXES APPLIED

Removed dummy policy-head losses and artificial skip/sign coupling in the final adapter; categorical numerical representation and a larger head; matched replay ordering/update budget; strict checkpoint identity/reload tests; separated real/hypothetical posthoc annotations and resource units. Historical code/results retained.

## MODELS USED

Pinned Apache2.0 SmolLM2-135M-Instruct frozen transformer plus learned task heads; historical NumPy MLP; deterministic ceiling, operand memorization and random references.

## MODEL SIZES

134,515,008 frozen backbone +337,618 trainable head =134,852,626 parameters per final primary condition. Historical MLP8722; deterministic/memorization/random references0 neural parameters.

## TRAINING REGIMES

Answer-only, structured answer-only, trajectory-supervised B, bounded binary-feedback/accepted-experience replay C. Same frozen backbone/head capacity; fixed1200 Adam updates/batch64. Labels and query amounts differ and are disclosed.

## DATASETS

Fresh final claims dataset seed 5151:1000 train/100 validation/200 IID/300 OOD. Train1x1/2x1/2x2; OOD3x2/3x3/4x2. Canonical identity leakage checks; old development/research datasets preserved.

## DATA BUDGETS

10,25,50,100,250,500,1000;2500/5000 deferred.

## SEEDS

0,1,2 for all primary condition/budget pairs and component controls.

## IID RESULTS

At1000 mean±SD:A5.000%±0.866; structured5.333%±1.041; B41.500%±2.179; C19.167%±3.014.

## OOD RESULTS

At1000:B6.111%±1.347; A/structured/C0%±0. All OOD figures remain separate from IID.

## DATA-EFFICIENCY RESULTS

Every N*(0.8/0.9/0.95) and DER is not reached; no2×/5× measured efficiency claim.

## GROUNDEDNESS

IID at 1000:B41.500%±2.179; C19.167%±3.014; answer baselines0% by evidence-required protocol. OOD B6.111%±1.347; others0%.

## TRAJECTORY VALIDITY

IID at 1000:B41.500%±2.179; C19.167%±3.014; OOD B6.111%±1.347/C0%; baselinesN/A.

## GOAL SUCCESS

Same numerical rates as groundedness in the measured final primary runs; below95% target.

## ABLATIONS

Three seeds at 50. No state:−3pp IID; C supervision contrast:−2.5pp IID; no real environment:0pp accuracy and−3pp grounded/goal by definition. Matched local policy, replay-store bypass, goal-flag removal and dream/no-dream:0pp accuracy. All OOD deltas0. ΔDER unavailable from a single budget.

## DREAM RESULTS

Goal-directed tagged integer rollout demonstrated; multiplication depth2/branch3 adds calls but no measured accuracy/ground/goal change at 50. No learned world model or utility claim.

## SECOND-VERTICAL STATUS

Deferred learned second environment. Generic interface and simulator reuse demonstrated; full learned architecture transfer not established.

## HISTORICAL GPT/GEMINI REFERENCE STATUS

Published GPT-4/Gemini GSM8K sources retained with prompting/training caveats. Direct multiplication comparison unsuitable; no GPT superiority claim.

## COMPUTE COST

Main:90 trained heads/210 evaluations; training171.947s/evaluation886.511s cached CPU;108,000 updates/6,912,000 sampled rows/379,153 feedback queries/303,924 inference-head calls;1,961,882 shared encoder tokens. Old-model/reference runs reported separately. No compute-efficiency superiority.

## FAILURE ANALYSIS

22,367 arithmetic errors;28,762 wrong-action outcomes;133 unsupported outcomes;1238 grounded correct outcomes across210 repeated-set evaluations. Other listed failure classes0. Grouped digit/carry/condition/budget/split tables and example-level failures retained.

## CLAIMS SUPPORTED

Narrow descriptive regime findings: B/C higher IID than answer-only under the tested unequal-supervision protocols; B nonzero verified OOD. Supervised pipeline can memorize; artifact/evidence separation verified. No broad architectural superiority claim is marked supported.

## CLAIMS NOT SUPPORTED

Threshold-based data efficiency, replay-storage contribution, dream usefulness, second learned-domain transfer. Accuracy/OOD/grounding/trajectory/state causal claims are inconclusive under the available controls.

## KNOWN LIMITATIONS

Hand-designed decomposition; unequal labels/queries; frozen indexed-feature backbone; one dataset/domain; small seed count; no independently reserved carry-only split; evidence metric privileges trajectory emitters; warm/shared CPU cache; second-domain and full fine-tuning deferred.

## TEST COUNT

92 tests passed, preserving all 86 previous tests and adding six final regressions. Editable install, real checkpoint CLI, generator, training/reload and analysis/audits executed. Browser checked baseline toggle, real failed step and plots.

## GIT STATUS

Final release on main with clean tracked working tree; historical commits preserved. Results/caches ignored; committed summary/figures/demo retained.

## LATEST COMMIT

Resolve the stable final release with `git rev-parse mindscape-final-study-v1`; protocol records training-source commit5cbda01. Release hash is provided in the completion message.

## FINAL REPOSITORY TREE

```text
mindscape/
├── README.md / pyproject.toml / Makefile
├── src/mindscape/
│   ├── core/ data/ environments/ verification/
│   ├── models/ training/ evaluation/
│   └── memory/ reasoning/ concepts/
├── configs/experiments/ [historical + final protocols]
├── datasets/generated/ [development_v1, research_v1, final_v1]
├── scripts/ [generation, training, evaluation, analysis, audit, freeze, demo]
├── tests/ [five suites,92 tests]
├── docs/ [architecture, protocol, results, conclusions, limits, reproduction]
├── experiments/ [historical snapshots, final_audit, final_smollm2_v1]
├── results/final/ [smollm2_v1, historical_mlp_v1, references_v1]
├── demo/ [index.html, screenshot.png]
└── work/ [ignored local runtime, pretrained weights, frozen-feature cache, logs]
```

[Detailed results](final_results.md) · [Reproduce](final_reproducibility.md) · [Limitations](limitations.md)
