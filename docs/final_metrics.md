# Actual completion metrics

Machine-readable full metrics: `results/coding/emergency_analysis_v1/summary.json`.

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
