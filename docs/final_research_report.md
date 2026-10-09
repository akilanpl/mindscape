# Final frozen coding-study report

Verified coverage: **1920 learning / 400 lockbox / 400 independent audits / 16 fresh latency cases**. Original1634-row checkpoint and original1122CPU-row prefix remain byte-identical. Recovery archive is preserved. Neural PID1620 exited; supervisor completed every checked subprocess.

## Observed outcomes

|Condition|Split|Successes /50|95% Wilson interval|
|---|---|---:|---|
|model_only|test|8/50|0.083–0.285|
|model_only|ood_test|12/50|0.143–0.374|
|structured|test|50/50|0.929–1.000|
|structured|ood_test|22/50|0.312–0.577|
|mindscape_b|test|45/50|0.786–0.957|
|mindscape_b|ood_test|24/50|0.348–0.615|
|mindscape_c|test|49/50|0.895–0.996|
|mindscape_c|ood_test|18/50|0.241–0.499|

Task-paired statistics and complete learning cells are in `experiments/coding_completion_v2/final_summary.json`. Intervals are descriptive and unadjusted for multiple comparisons. Successful replay verifies evidence, not successful repair. Five zero-tool trajectories have undefined tool-result agreement.

## All dedicated latency observations

Sequential, cache-disabled MPSfloat16; no model reload per case. Stage intervals can overlap. Four task/condition pairs per condition; no repeated key counted twice.

|Condition|Task|Split|Success|Wall seconds|
|---|---|---|---|---:|
|model_only|fd23fa32e4f9|test|False|3.030|
|model_only|e5545f5d719e|ood_test|False|2.800|
|model_only|34d9c496b646|test|False|2.257|
|model_only|f7f984138799|ood_test|False|1.674|
|structured|fd23fa32e4f9|test|True|2.470|
|structured|e5545f5d719e|ood_test|False|3.142|
|structured|34d9c496b646|test|True|2.641|
|structured|f7f984138799|ood_test|False|3.241|
|mindscape_b|fd23fa32e4f9|test|True|33.891|
|mindscape_b|e5545f5d719e|ood_test|False|45.430|
|mindscape_b|34d9c496b646|test|True|37.958|
|mindscape_b|f7f984138799|ood_test|False|33.686|
|mindscape_c|fd23fa32e4f9|test|True|26.679|
|mindscape_c|e5545f5d719e|ood_test|False|32.615|
|mindscape_c|34d9c496b646|test|True|31.098|
|mindscape_c|f7f984138799|ood_test|True|32.388|

## Validation and reproducibility

Fresh original and clean-restored suites each passed **130tests plus6subtests, zero failures/errors/skips**. Scoped lint, compilation, wheel build, fairness/training checks and Nodeplayback verification passed. Snapshot `results/final/coding_research_v2` and restored file hashes independently verified:6242 files each. Full commands/outcomes, all16 stage measurements, scientific pins and SHA256checksums are in `experiments/coding_completion_v2/release_manifest.json`.

## Limitations

- Mixed hardware/precision and unmatched supervision/interaction compute prevent architecture-only causal attribution
- Synthetic bounded repair tasks; unknown foundation pretraining exposure
- 16latencycases are descriptive, not broad performance estimates
- No measured zero-budget/control component memory/dream/recovery intervention
- Foundation weights are pinned external downloads, excluded from package
- No configured type-check acceptance task; no visual browser QA claimed
- No GPT/Gemini superiority; negative and unsuccessful cases retained

Completion refers to frozen experimental coverage. Broad capability and GPT/Gemini superiority are unsupported; data-efficiency and unmeasured intervention claims remain inconclusive. Package publication/tag verification is recorded separately after export.
