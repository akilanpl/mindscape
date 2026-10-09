# Research v2: verified progress and bounded mechanism findings

Research v2 is not fully complete. Git preservation, clean v1 restoration, actual Linux cloud validation, numeric-core traces, prospective task freezing and a controlled numeric intervention study are verified. Coding-foundation telemetry, broader domain replication and exact historical GPT/Gemini comparisons remain outstanding.

## Provenance and validation

Research v1 remains at `2862f5ae54443b231abda4255bdd6c652eca2343` with its original completion tag and immutable archives. No frozen implementation under `src` was changed. The canonical restoration verified all 23,842 archived file hashes. An initial Linux run failed because an archived Apple ARM compiled WASI cache was incompatible; that failed evidence is retained. Recompilation from the same checksum-pinned WASM resolved portability without altering the frozen package.

[Cloud run 37946027443](https://github.com/akilanpl/mindscape/actions/runs/37946027443) executed commit `0c3f869077756f50e89a551a2fa2b9ca58f05728` on Linux x86_64 with four CPUs and Python 3.11.17. Dependency pins include CPU PyTorch 2.8.0+cpu; the actual studied NumPy MLP uses float64 CPU arithmetic. The full suite passed 137 tests plus six subtests, with no failures, errors or skips. Saved-evidence integrity, statistics, fairness, frontend playback, scoped lint and resume checks passed. Separate study peak memory and compute cost were not measured; validation-process memory is not a substitute.

## Frozen experiment and actual outcomes

The protocol and 80 novel, identity-disjoint tasks were committed before execution. Forty IID unsigned 2×2 and forty OOD signed 4×3 multiplication tasks were evaluated for three preserved budget-50 trained seeds across eight binary conditions. These are 1,920 repeated task/seed/condition executions, not 1,920 independent tasks. Every completed key was unique. All checkpoint hashes remained unchanged. Repeating the study verified existing traces and did not repeat inference.

| Mask/state/previous-action condition | IID successes / 120 | OOD successes / 120 |
|---|---:|---:|
| 000 | 0 | 0 |
| 001 | 0 | 0 |
| 010 | 61 | 2 |
| 011 | 120 | 43 |
| 100 | 120 | 120 |
| 101 | 120 | 120 |
| 110 | 120 | 120 |
| 111 | 120 | 120 |

All 734 unsuccessful executions are retained with actual invalid-action outcomes. No unrun case was scored as zero.

The masked controller achieved all goals even when both neural feature readouts were removed. The deterministic environment performs arithmetic and the legal-action mask can identify the next valid operation. Consequently, masked success does not demonstrate learned arithmetic or neural necessity. With both readouts present and masking removed, IID remained 120/120 while OOD fell to 43/120 (35.83%). The observed OOD gap under this controlled intervention is 64.17 percentage points; the task-cluster bootstrap 95% interval is 50.0–77.5 points, conditional on the three trained seeds. Per-seed exact paired McNemar tests use a predeclared Holm family of 18 comparisons; full values and cell Wilson intervals are in the machine-readable statistics.

With masking enabled, the two single-readout primary interventions had no observed effect. Their empirical bootstrap intervals collapse to zero because all paired observed differences are zero; this is not proof of universal equivalence. The unmasked condition table shows feature interactions descriptively: neither previous-action features alone nor the operand-only readout produced a successful case. No unadjusted significance claim is made for exploratory interactions.

## Observation and interpretation

Each execution has JSON and readable Markdown with exact numeric input, declared normalization, model/checkpoint identity, feature slices, neural logits, derived uncalibrated probabilities, shapes, actual action selection, working-memory changes, trajectory and independent result. No text tokenizer is present. Hidden activation summaries are explicitly reconstructed from observed features and checkpoint weights rather than falsely stack-captured. They do not explain causal influence. Instrumented episode time summed to 37.53 seconds; it excludes serialization and setup, includes observer overhead, and is not a dedicated latency benchmark or hardware speedup claim.

Feature-readout removal is not removal of the entire memory module. Verification and goals were held fixed. Planning, dreaming, feedback learning and recovery are absent from this backend. The experiment establishes effects of declared software interventions at fixed trained policies in this numeric environment, not broad cognitive or coding superiority. Training/evaluation identity separation was verified; semantic overlap and external pretraining exposure cannot be excluded universally. Historical unsuccessful numeric and coding evidence remains preserved.

## Durable artifacts and outstanding gates

The versioned archive `research_checkpoint/research_v2_numeric_2026-10-09` contains all cloud receipts and per-input evidence: 3,876 files, 11,957,855 compressed bytes, independently hash-verified after clean restoration. Its manifest binds every file and the exact source commit. Raw indexes, statistics, dashboard, tests and job receipts are under `experiments/research_v2/cloud_runs/37946027443`. Stage receipts explicitly identify the verified scope.

Exact historical GPT/Gemini releases, access conditions and an authorized spending limit have not been supplied. Their outcomes, latency, costs and performance gaps are unknown, not zero. The coding-foundation model and its 61 adapters remain in preserved v1 evidence, but their deep computation telemetry and controlled v2 interventions have not run. Independent broader-domain replication remains necessary before claiming scientifically competitive architecture results. No full Research v2 completion tag is created.
