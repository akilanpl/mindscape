# Actual completion metrics

Machine-readable full metrics: `results/coding/emergency_analysis_v1/summary.json`.

# Mindscape bounded MPS completion evidence

Learning curve: **1122/1,920 episodes**, 798 not run. Lockbox: **312/400 condition-task episodes** (100 independent tasks). Preserved CPU work remains intact.

All metrics below derive from saved episodes. Missing cells are not zero scores. CPU float32 and MPS reduced-precision strata are kept separate. Supervision and interaction budgets differ across conditions; architecture-only causality is not established.

|Condition|Split|Completed/planned|Success|95% Wilson interval|
|---|---|---:|---:|---|
|model_only|test|50/50|16.0%|[0.08337420678033404, 0.2851421606303499]|
|model_only|ood_test|50/50|24.0%|[0.1429739139699173, 0.3741268375794292]|
|structured|test|50/50|100.0%|[0.9286524008666414, 1.0]|
|structured|ood_test|50/50|44.0%|[0.3116219921125717, 0.5769397197834315]|
|mindscape_b|test|50/50|90.0%|[0.7863976856252035, 0.9565242350681096]|
|mindscape_b|ood_test|46/50|47.8%|[0.34124689434045685, 0.6186258692719597]|
|mindscape_c|test|16/50|93.8%|[0.7167126242970107, 0.9888806552353576]|

Observed complete-cell sample thresholds and all partial cells are in `results/coding/emergency_analysis_v1/summary.json`. DER is undefined for a causal architecture comparison. No zero-budget control was run.

Task-paired confidence intervals are descriptive and unadjusted for multiple comparisons. Transition validity is conditional on recorded actions; malformed model proposals are reported separately.

Operational answer metrics: `{"definition": "Successful terminal program AND independent full-trajectory replay agreement; no semantic hallucination detector", "denominator": 312, "grounded_successful_answers": 159, "terminal_incorrect_answers": 138, "semantic_hallucination_rate": null, "semantic_hallucination_status": "Not measured by this program-repair protocol", "grounded_successful_answer_rate": 0.5096153846153846, "unsupported_by_terminal_tests_rate": 0.4423076923076923}`. Failed terminal tests are unsupported program answers under this oracle; they are not a measured semantic hallucination rate.

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
    "episodes": 296,
    "planned": 400,
    "final_outcome_agreement": 1.0,
    "structured_evidence_agreement": 1.0,
    "state_agreement": 1.0,
    "state_comparisons": 1862,
    "tool_result_agreement": 1.0,
    "tool_result_comparisons": 783
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
          "p50": 47.492164208495524,
          "p95": 62.662245208004606,
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
      "caution": "Descriptive sample; stage timers overlap; TTFT is per call, other fields per episode. Batched model durations are shared across requests; environment covers timed transitions, verification is measured sandbox execution; no state-construction timer in reused smoke evidence"
    }
  ]
}
```

## Preserved public benchmark

0.5B: 94/164 (57.32%); 1.5B: 98/164 (59.76%). Greedy one-sample full-module generation, 512-token cap, CPython 3.14.7 WASI. Foundation pretraining exposure is unknown. These local results do not exceed historical GPT-4 67.0% or Gemini Ultra 74.4%, and protocols differ. See `docs/historical_references.md` for pinned primary sources.

Dedicated post-lockbox latency is not complete. Any reused smoke measurements are labeled explicitly.

Complete suite: {'tests': 135, 'failures': 0, 'errors': 0, 'skipped': 0, 'pytest_passed_tests': 129, 'subtests_passed': 6, 'seconds': 23.74}. Local fairness/leakage assertions: 10158, PASS. Contamination-free status remains unknown. All 56 retained adapters validated. SQL environment implemented; SQL model study deferred. Recovery and memory/dream ablations not run; their claims remain inconclusive.
