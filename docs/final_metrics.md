# Actual completion metrics

Machine-readable full metrics: `results/coding/emergency_analysis_v1/summary.json`.

# Mindscape bounded MPS completion evidence

Learning curve: **1122/1,920 episodes**, 798 not run. Lockbox: **0/400 condition-task episodes** (100 independent tasks). Preserved CPU work remains intact.

All metrics below derive from saved episodes. Missing cells are not zero scores. CPU float32 and MPS reduced-precision strata are kept separate. Supervision and interaction budgets differ across conditions; architecture-only causality is not established.

|Condition|Split|Completed/planned|Success|95% Wilson interval|
|---|---|---:|---:|---|

Observed complete-cell sample thresholds and all partial cells are in `results/coding/emergency_analysis_v1/summary.json`. DER is undefined for a causal architecture comparison. No zero-budget control was run.

## Actual acceleration measurements

```json
{}
```

CPU reference episodes were previously measured with uncached inference; they were not rerun. MPS sec/episode is amortized throughput, distinct from per-episode latency while waiting for batches. Device tensors and model placement are checked; unsupported MPS operations fail with fallback disabled. Native allocation reported by the user is verified, but inference use is only established by native run outputs.

## Replay and latency

```json
{
  "replay": {
    "episodes": 0,
    "planned": 400,
    "final_outcome_agreement": null,
    "structured_evidence_agreement": null,
    "state_agreement": null,
    "state_comparisons": 0,
    "tool_result_agreement": null,
    "tool_result_comparisons": 0
  },
  "latency": []
}
```

## Preserved public benchmark

0.5B: 94/164 (57.32%); 1.5B: 98/164 (59.76%). Greedy one-sample full-module generation, 512-token cap, CPython 3.14.7 WASI. Foundation pretraining exposure is unknown. These local results do not exceed historical GPT-4 67.0% or Gemini Ultra 74.4%, and protocols differ. See `docs/historical_references.md` for pinned primary sources.

Complete suite: {'tests': 135, 'failures': 0, 'errors': 0, 'skipped': 0, 'pytest_passed_tests': 129, 'subtests_passed': 6, 'seconds': 23.602}. Local fairness/leakage assertions: 10158, PASS. Contamination-free status remains unknown. All 56 retained adapters validated. SQL environment implemented; SQL model study deferred. Recovery and memory/dream ablations not run; their claims remain inconclusive.
