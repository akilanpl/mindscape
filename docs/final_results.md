# Final results

Final main study:90 adapted heads (84 primary plus 6 retrained ablations),210 shared-runner evaluations, three seeds 0/1/2,200 IID and300 OOD examples per run. An additional12 historical MLP models/24 evaluations and6 non-neural reference evaluations use the same fresh final dataset. Total final research evaluations240;102 trained research models. Diagnostics are separate. Raw results remain in `results/final/`; the committed summary snapshot is `experiments/final_smollm2_v1/`.

## Primary results at 1000 training problems

| Condition | IID exact accuracy | Structural OOD exact accuracy | IID groundedness | IID goal success | IID trajectory validity |
|---|---:|---:|---:|---:|---:|
|Answer-only A|5.000% ± 0.866|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|N/A|
|Structured baseline|5.333% ± 1.041|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|N/A|
|Mindscape B|41.500% ± 2.179|6.111% ± 1.347|41.500% ± 2.179|41.500% ± 2.179|41.500% ± 2.179|
|Mindscape C|19.167% ± 3.014|0.000% ± 0.000|19.167% ± 3.014|19.167% ± 3.014|19.167% ± 3.014|

All rates are mean±sample SD across three training seeds. A/structured do not emit trajectories; validity isN/A and their evidence-required grounding/goal scores are0 by protocol. B OOD groundedness/goal/validity are6.111%±1.347; C OOD rates are0. No final success target was met.

## Learning curves

### IID

| Training problems | Answer-only A | Structured | B | C |
|---:|---:|---:|---:|---:|
|10|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|
|25|0.167% ± 0.289|0.000% ± 0.000|2.000% ± 0.000|0.500% ± 0.866|
|50|0.333% ± 0.289|0.167% ± 0.289|3.000% ± 1.323|0.500% ± 0.500|
|100|0.667% ± 0.289|0.667% ± 0.289|13.333% ± 1.041|2.667% ± 2.517|
|250|2.333% ± 0.577|0.833% ± 0.577|30.333% ± 1.041|9.667% ± 3.215|
|500|1.333% ± 0.764|2.333% ± 0.764|38.000% ± 0.866|15.000% ± 3.606|
|1000|5.000% ± 0.866|5.333% ± 1.041|41.500% ± 2.179|19.167% ± 3.014|
### Structural OOD

| Training problems | Answer-only A | Structured | B | C |
|---:|---:|---:|---:|---:|
|10|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|
|25|0.000% ± 0.000|0.000% ± 0.000|0.111% ± 0.192|0.000% ± 0.000|
|50|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|
|100|0.000% ± 0.000|0.000% ± 0.000|0.889% ± 0.192|0.000% ± 0.000|
|250|0.000% ± 0.000|0.000% ± 0.000|1.556% ± 0.694|0.111% ± 0.192|
|500|0.000% ± 0.000|0.000% ± 0.000|4.667% ± 0.333|0.111% ± 0.192|
|1000|0.000% ± 0.000|0.000% ± 0.000|6.111% ± 1.347|0.000% ± 0.000|

## Data efficiency and costs

All observed N*(0.8/0.9/0.95) thresholds were not reached on either split for every condition; all DERs are censored/not reached. There is no measured2× or5× threshold-crossing advantage. Problem budget is distinct from label, feedback-query and compute budget.

Main study measured training time 171.947s and evaluation time 886.511s. It performed108,000 optimizer updates,6,912,000 sampled supervision rows,379,153 scalar feedback queries and303,924 prediction-head calls. The shared frozen transformer processed1,961,882 tokens in23,463 batches; encoder feature requests726,352. These are cached CPU measurements, exclude dataset/artifact bookkeeping outside timed sections, and do not establish cold-deployment compute efficiency. Full per-model costs are in `costs.json`.

## Ablations at 50 problems

| Intervention | IID Δaccuracy (pp) | OOD Δaccuracy (pp) | IID Δgroundedness (pp) | IID Δgoal (pp) |
|---|---:|---:|---:|---:|
|dream|0.000|0.000|0.000|0.000|
|matched_information_control|0.000|0.000|0.000|0.000|
|no_dream|0.000|0.000|0.000|0.000|
|no_environment_interaction|0.000|0.000|-3.000|-3.000|
|no_episodic_memory|0.000|0.000|0.000|0.000|
|no_explicit_state|-3.000|0.000|-3.000|-3.000|
|no_goal_evaluator|0.000|0.000|0.000|0.000|
|no_trajectory_supervision_regime_C|-2.500|0.000|-2.500|-2.500|

Every intervention has three seeds. Δdata efficiency is not estimable from one budget. No goal evaluator means removal of a constant goal input flag while retaining common external scoring. C versus B is a supervision/feedback contrast, not an isolated removal. No real interaction preserves numerical predictions but loses real evidence by definition. The matched local supervised policy and matched-order/update no-SQLite-retrieval control have identical predictions to their references. Dream changed no measured accuracy/groundedness/goal; its extra calls do not support usefulness. The no-state mean reduction is not significant in every seed after the component-test Holm family, so state-contribution claims remain inconclusive at this budget.

## Statistics and errors

Per-run Wilson95 intervals report conditional example uncertainty; seed SD reports training variation. Exploratory paired tests use seed 0 at 1000, six Holm comparisons; component tests are a separate 18-comparison family across seeds/splits. No seed pooling. B versus A IID Holm p=2.2354e-17 and OOD p=1.90735e-6; C versus A IID p=1.80444e-7. These differences are observed regime associations, not controls for the unequal information/scaffold. Practical OOD performance remains6.11%, far below90%.

| Failure/outcome category | Count across210 repeated-set evaluations |
|---|---:|
|arithmetic_error|22367|
|wrong_action|28762|
|invalid_transition|0|
|malformed_state|0|
|trajectory_failure|0|
|goal_failure|0|
|unsupported_answer|133|
|timeout|0|
|other|0|
|correct|1238|

These counts describe repeated benchmark runs, not unique independent errors. `difficulty_failures.json` groups condition/budget/seed/IID-OOD/digit structure/carry count, and per-run `failures.json` retains example identities and original taxonomy. Main failures are wrong local claims and answer arithmetic errors; syntax/timeouts were not the limiting issue. At1000,108/200 IID examples and280/300 OOD examples contain reference local contexts absent from the training subset. Carry signatures are unseen for59/200 IID and300/300 OOD. The latter includes the structural length change; these are posthoc descriptive strata, not independently pre-reserved carry-only holdouts.

## Sanity and interpretive references

The final supervised representation memorized n=10, all 45 canonical1x1 problems and a mixed100 training set at 100% in diagnostics. Mixed100 validation reached3% answer-only and18% B; local claim accuracy62.47%, carry label accuracy82.60%. C n=10 same-set accuracy was0% with incomplete accepted experience; it did not receive complete trajectories. This separates fitting ability from generalization and exploration coverage.

| Historical MLP on fresh final dataset at 1000 | IID mean±SD | OOD mean±SD |
|---|---:|---:|
|Answer-only A|0.333% ± 0.289|0.000% ± 0.000|
|Structured baseline|0.333% ± 0.289|0.000% ± 0.000|
|Mindscape B|8.000% ± 0.866|0.667% ± 0.000|
|Mindscape C|8.333% ± 2.021|0.333% ± 0.333|

The original phase-4 dataset remains separate: its OOD accuracy was0% across all primary runs. Different sampled datasets can change rare correct counts. Larger final heads/categorical features/active losses/update budgets/pretrained features changed together, so gains over the MLP do not isolate pretraining. Deterministic execution scored 100% IID/OOD; operand memorization and the seeded uniform-integer random reference scored 0%. The deterministic ceiling is an algorithm, not a learning/data-efficiency baseline.

## Claims

| Claim | Status | Evidence | Limits |
|---|---|---|---|
|Mindscape improves accuracy|INCONCLUSIVE|See primary curves and matched-information control.|State/scaffold and supervision differ from answer baselines; wrapper causal benefit must exceed matched local policy.|
|Mindscape improves data efficiency|NOT SUPPORTED|Observed N* and DER only.|Unequal label/query counts; compute reported separately.|
|Mindscape improves structural OOD generalization|INCONCLUSIVE|Unseen structures evaluated separately.|One domain and hand-defined cursor.|
|Mindscape improves groundedness|INCONCLUSIVE|Verified executed evidence measured.|Answer-only protocol emits no trajectory; metric favors evidence emitters by definition.|
|Mindscape improves trajectory validity|INCONCLUSIVE|Complete verifier checks.|Baselines without trajectories have null validity; matched policy is required.|
|Explicit state contributes|INCONCLUSIVE|Retrained no-state intervention at n=50.|One budget and changes representation. Support, if present, is limited to IID at n=50; OOD is separately reported.|
|Episodic memory contributes|NOT SUPPORTED|Matched accepted rows/order/updates no-storage control.|Tests SQLite retrieval versus in-memory replay, not all reuse.|
|Dream simulation contributes|NOT SUPPORTED|Depth2/branch3 intervention n=50.|Confidence scoring, not learned world model; extra inference calls.|
|Architecture transfers to another environment|NOT SUPPORTED|Goal-directed generic integer dream demo retained.|No second learned benchmark; extension deferred.|

Supported descriptive findings: under these tested unequal-supervision protocols, B/C have higher IID accuracy than answer-only, and B executes some verified OOD trajectories. These observations do not establish an independent Mindscape wrapper advantage; matched-information policy predictions are identical. Data-efficiency, replay-storage, dream usefulness and second-domain-transfer claims are not supported. Broader causal claims remain inconclusive.

Artifact audit:438 actual checks passed, including nested training IDs, shared parameter count, no singleton forcing, proposal/result separation, posthoc judgments, hypothetical tags and matched-control prediction equality.

## Figures

- [accuracy](../experiments/final_smollm2_v1/plots/accuracy.svg)
- [ood_accuracy](../experiments/final_smollm2_v1/plots/ood_accuracy.svg)
- [groundedness](../experiments/final_smollm2_v1/plots/groundedness.svg)
- [goal_success](../experiments/final_smollm2_v1/plots/goal_success.svg)
- [trajectory_validity](../experiments/final_smollm2_v1/plots/trajectory_validity.svg)
- [data_efficiency](../experiments/final_smollm2_v1/plots/data_efficiency.svg)
- [ablation_effects](../experiments/final_smollm2_v1/plots/ablation_effects.svg)
- [errors](../experiments/final_smollm2_v1/plots/errors.svg)
- [training_compute](../experiments/final_smollm2_v1/plots/training_compute.svg)
- [inference_cost](../experiments/final_smollm2_v1/plots/inference_cost.svg)
