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

<!-- emergency-mps-results -->

# Mindscape bounded MPS completion evidence

Learning curve: **1122/1,920 episodes**, 798 not run. Lockbox: **400/400 condition-task episodes** (100 independent tasks). Preserved CPU work remains intact.

All metrics below derive from saved episodes. Missing cells are not zero scores. CPU float32 and MPS reduced-precision strata are kept separate. Supervision and interaction budgets differ across conditions; architecture-only causality is not established.

|Condition|Split|Completed/planned|Success|95% Wilson interval|
|---|---|---:|---:|---|
|model_only|test|50/50|16.0%|[0.08337420678033404, 0.2851421606303499]|
|model_only|ood_test|50/50|24.0%|[0.1429739139699173, 0.3741268375794292]|
|structured|test|50/50|100.0%|[0.9286524008666414, 1.0]|
|structured|ood_test|50/50|44.0%|[0.3116219921125717, 0.5769397197834315]|
|mindscape_b|test|50/50|90.0%|[0.7863976856252035, 0.9565242350681096]|
|mindscape_b|ood_test|50/50|48.0%|[0.34797135286578046, 0.6148825510995539]|
|mindscape_c|test|50/50|98.0%|[0.8950455641036219, 0.9964607407283539]|
|mindscape_c|ood_test|50/50|36.0%|[0.24138749651846741, 0.49858983123887307]|

Observed complete-cell sample thresholds and all partial cells are in `results/coding/emergency_analysis_v1/summary.json`. DER is undefined for a causal architecture comparison. No zero-budget control was run.

Task-paired confidence intervals are descriptive and unadjusted for multiple comparisons. Transition validity is conditional on recorded actions; malformed model proposals are reported separately.

Observed self-correction from saved offline first-edit assessments: `[{"condition": "mindscape_b", "split": "test", "completed_episodes": 50, "first_edit_failures": 0, "later_terminal_successes": 0, "conditional_self_correction_rate": null, "wilson_ci95": null, "definition": "Terminal success after an incorrect first accepted edit; first-edit assessment occurs offline after policy termination, never as hidden model feedback", "caution": "Descriptive within-episode self-correction; no injected-fault recovery intervention or causal memory/planner attribution"}, {"condition": "mindscape_b", "split": "ood_test", "completed_episodes": 50, "first_edit_failures": 38, "later_terminal_successes": 12, "conditional_self_correction_rate": 0.3157894736842105, "wilson_ci95": [0.1908460367559618, 0.4745575989060954], "definition": "Terminal success after an incorrect first accepted edit; first-edit assessment occurs offline after policy termination, never as hidden model feedback", "caution": "Descriptive within-episode self-correction; no injected-fault recovery intervention or causal memory/planner attribution"}, {"condition": "mindscape_c", "split": "test", "completed_episodes": 50, "first_edit_failures": 1, "later_terminal_successes": 0, "conditional_self_correction_rate": 0.0, "wilson_ci95": [0.0, 0.7934506856227626], "definition": "Terminal success after an incorrect first accepted edit; first-edit assessment occurs offline after policy termination, never as hidden model feedback", "caution": "Descriptive within-episode self-correction; no injected-fault recovery intervention or causal memory/planner attribution"}, {"condition": "mindscape_c", "split": "ood_test", "completed_episodes": 50, "first_edit_failures": 32, "later_terminal_successes": 0, "conditional_self_correction_rate": 0.0, "wilson_ci95": [0.0, 0.1071791982550706], "definition": "Terminal success after an incorrect first accepted edit; first-edit assessment occurs offline after policy termination, never as hidden model feedback", "caution": "Descriptive within-episode self-correction; no injected-fault recovery intervention or causal memory/planner attribution"}]`. This is separate from unrun injected-fault recovery tests.

Operational answer metrics: `{"definition": "Successful terminal program AND independent full-trajectory replay agreement; no semantic hallucination detector", "denominator": 400, "grounded_successful_answers": 228, "terminal_incorrect_answers": 172, "semantic_hallucination_rate": null, "semantic_hallucination_status": "Not measured by this program-repair protocol", "unverified_episodes": 0, "grounded_successful_answer_rate": 0.57, "unsupported_by_terminal_tests_rate": 0.43}`. Failed terminal tests are unsupported program answers under this oracle; they are not a measured semantic hallucination rate.

Original frozen configuration SHA-256: `11d2a6b5bef6517a5cae10dca46f1e94bfe2db3504afd3942434fcb78f496bb0`. Seeds: 11/23/37 for learning; seed 11 and budget 100 for lockbox. Exact checkpoint/source hashes are in the frozen protocol; host/runtime and training compute are in the machine-readable summary.

## Actual acceleration measurements

```json
{
  "cpu_seconds_per_episode": 111.3537078230629,
  "cpu_reference_scope": "Preserved fresh CPU float32 episodes; not rerun",
  "mps_throughput_seconds_per_episode": 27.54043339325017,
  "mps_mean_episode_wall_seconds": 372.8890333698155,
  "throughput_speedup": 4.043280882077708,
  "mps_episodes_per_hour": 130.7168971742586,
  "selected_batch_size": 16,
  "precision": "float16",
  "load_seconds": 22.024639666007715,
  "peak_process_rss_bytes": 3421274112,
  "mps_allocated_bytes": 3333364480,
  "mps_driver_bytes": 12526354432,
  "profiling_seconds": 600.09845495224,
  "interpretation": "Historical CPU vs fresh batched MPS throughput; per-case latency reported separately. Arithmetic precision differs; learning-curve hardware strata remain disclosed."
}
```

One phase-boundary memory release: `{"before_driver_bytes": 10687807488, "after_driver_bytes": 4346019840, "live_allocated_bytes": 3112987904, "reason": "Single profiling/evaluation boundary; no per-operation cache clearing", "configuration_reused": true, "time_unix": 1791468859.762655}`.

CPU reference episodes were previously measured with uncached inference; they were not rerun. MPS sec/episode is amortized throughput, distinct from per-episode latency while waiting for batches. Device tensors and model placement are checked; unsupported MPS operations fail with fallback disabled. Native allocation reported by the user is verified, but inference use is only established by native run outputs.

## Replay and latency

```json
{
  "replay": {
    "episodes": 400,
    "planned": 400,
    "final_outcome_agreement": 1.0,
    "structured_evidence_agreement": 1.0,
    "state_agreement": 1.0,
    "state_comparisons": 2584,
    "tool_result_agreement": 1.0,
    "tool_result_comparisons": 1092
  },
  "latency": [
    {
      "condition": "mindscape_b",
      "episodes": 16,
      "planned": 20,
      "measurement_source": "Reused actual cache-disabled representative MPS smoke benchmark; no additional neural run",
      "device_precision_counts": {
        "mps/float16": 128
      },
      "actual_batch_sizes": [
        16
      ],
      "peak_process_rss_bytes": null,
      "peak_mps_live_bytes": null,
      "peak_mps_driver_bytes": null,
      "peak_cuda_allocated_bytes": null,
      "peak_cuda_reserved_bytes": null,
      "sequential_episodes_per_hour": null,
      "percentiles": {
        "total": {
          "p50": 373.02724195850897,
          "p95": 373.10285165624737,
          "n": 16
        },
        "ttft": {
          "p50": 19.29911718749645,
          "p95": 22.887992582996958,
          "n": 128
        },
        "response": {
          "p50": 47.52857843750098,
          "p95": 62.74817655384686,
          "n": 128
        },
        "model_generation": {
          "p50": 368.19099453998206,
          "p95": 368.19099453998206,
          "n": 16
        },
        "environment_execution": {
          "p50": 0.016273770495899953,
          "p95": 0.024654030479723588,
          "n": 16
        },
        "verification": {
          "p50": 2.0404848954931367,
          "p95": 2.1578504689969122,
          "n": 16
        },
        "test_execution": {
          "p50": 0.0,
          "p95": 0.0,
          "n": 16
        }
      },
      "caution": "Descriptive sample; stage timers overlap. TTFT is model first-token time after CPU input encoding. Response is per-call wall time including encoding/decoding in dedicated measurements; total is multi-step episode wall time. Batched smoke model durations are shared across requests. Unmeasured timers are not inferred."
    }
  ]
}
```

## Preserved public benchmark

0.5B: 94/164 (57.32%); 1.5B: 98/164 (59.76%). Greedy one-sample full-module generation, 512-token cap, CPython 3.14.7 WASI. Foundation pretraining exposure is unknown. These local results do not exceed historical GPT-4 67.0% or Gemini Ultra 74.4%, and protocols differ. See `docs/historical_references.md` for pinned primary sources.

Dedicated post-lockbox latency is not complete. Any reused smoke measurements are labeled explicitly.

Complete suite: {'tests': 135, 'failures': 0, 'errors': 0, 'skipped': 0, 'pytest_passed_tests': 129, 'subtests_passed': 6, 'seconds': 23.933}. Local fairness/leakage assertions: 10158, PASS. Contamination-free status remains unknown. All 56 retained adapters validated. SQL environment implemented; SQL model study deferred. Recovery and memory/dream ablations not run; their claims remain inconclusive.
