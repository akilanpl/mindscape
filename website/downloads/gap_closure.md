# Scientific gap closure: actual evidence and remaining limits

The original 1,920 numeric interventions, 734 failures and all 3,846 study hashes were reused. All 19,448 directly observed action selections had exactly one legal action. No completed intervention was rerun. Research v1's 400 locked cases, 1,920 coding learning rows and 400 saved audits remain unchanged.

## Prospective controls and task-variant replication

Protocol/data commit `06d1f8e` froze 24 novel identities, disjoint from all preserved numeric datasets and the previous 80-task v2 evaluation, before new inference. Three historical trained seeds executed four conditions: trained or zeroed copied weights, each with exact legal masking or permissive all-action selection. Original operands, environment, independent scorer and 256-action budget were fixed. No training, tuning, model download or external API was used. [Cloud run 37949932056](https://github.com/akilanpl/mindscape/actions/runs/37949932056), at `667585131e96967dc53037e4d0f50d94f55c020d`, completed and safely resumed all 288 unique keys. Eighty-five unsuccessful cases remain in the evidence.

| Novel family | Trained masked / permissive successes | Masking effect, percentage points | Task-cluster bootstrap 95% interval |
|---|---:|---:|---:|
| Unsigned 3×2 | 24/24 versus 20/24 | +16.7 | 0 to +41.7 |
| Signed 4×2 | 24/24 versus 15/24 | +37.5 | +8.3 to +66.7 |
| Unsigned 1×4 | 24/24 versus 24/24 | 0 | 0 to 0 |

Task clusters, rather than 24 executions per family, are the resampling units: eight distinct tasks per family, conditional on three shared historical checkpoints. All 18 predeclared per-seed tests were Holm-corrected; none passes 0.05 (minimum adjusted p=0.140625). The small replication does not establish confirmatory significance. Degenerate bootstrap intervals at observed ceilings do not establish population equivalence.

Zero-weight policies succeeded with masking on every task and failed under permissive selection on every task. Seed copies have identical zero weights and are not independent controls: the effective comparison is 24/24 unique masked tasks versus 0/24 permissive tasks. The observed paired effect is +100 percentage points. An explicitly post hoc, conservative 95% interval is +66.6 to +100 points, using simultaneous Bonferroni Clopper-Pearson bounds on beneficial and harmful paired outcomes. Its sampling interpretation is confined to these three numeric task families, not arbitrary task distributions. This descriptive supplement does not replace the predeclared tests or their multiplicity correction.

The zero-weight intervention supports a causal statement confined to this implementation and these tasks: trained neural weights were unnecessary for goal attainment when the controller's exact legal-action filter remained enabled. Disabling that filter changes action selection while preserving the same environment/scorer. It exposes dependence on learned-policy behavior; it is not a harder scorer or a new broad-intelligence benchmark. The qualitative masking finding replicated across newly frozen task variants. It did not replicate a uniform learned-policy gap: the 1×4 family showed none. Models, implementation and training pipeline are shared, so this is task-variant replication, not independent laboratory or cross-domain replication.

## Coding-foundation computation telemetry

A hash-bound, nonduplicating index covers all 2,320 saved coding episodes. It links exact raw rows, model calls, responses, timings, failures, component actions, state transitions, adapters and protocol references. Historical exact serialized prompts, token IDs and logits were not retained; they remain unavailable rather than retroactively fabricated.

Two successful observer diagnostics used the existing pinned Qwen2.5-Coder-1.5B weights and original budget-100 adapters, on CPU float32 without generation caching. The same historical lockbox task was reused for instrumentation, not added to accuracy evidence:

- Model-only: one model call, 39 direct token-input/logit-output pairs; repair unsuccessful due to an invalid edit envelope; elapsed including loading 9.25 seconds, process peak RSS 8.46 GB.
- C condition: eight model calls, 352 direct token/logit pairs; eight retrieval, dreaming and planner invocations, three actual edits and three memory updates. Final hidden-test evaluation passed. Five later proposals hit the original edit budget. Elapsed including loading 71.98 seconds, process peak RSS 7.16 GB.

These are observed process memory maxima, not aggregate system memory. Timings include instrumentation and differ from original MPS benchmarking; no speedup or accuracy estimate is inferred. The first diagnostic failed because PEFT generation bypassed an outer hook. Its failure log is retained; the corrected observer hooks the actual base model. No tensors or lost response were invented for that failed diagnostic.

Live traces close exact-input, direct tokenization/model-forward, selected logits/probabilities, component invocation, state-transition, output, timing and failure-observation gaps for the tested paths. Probabilities are derived and uncalibrated; dreaming/planner predictions remain hypotheses. Hidden activations and live coverage of every historical adapter/condition are not measured. Historical input/logit gaps cannot be closed from saved rows alone. The reusable observer and report renderer are separate from unchanged v1 source. Observer reporting and exception handling evolved after the diagnostics; stored events and model/checkpoint hashes remain immutable.

## Validation and reproduction

Cloud validation passed 138 tests plus six subtests, with zero failures, errors or skips. Saved v1 integrity, fairness, frontend playback, scoped lint and safe resume passed. Two focused local observer tests pass, including PEFT base-forward capture, absence of extra forwards, memory observation and cleanup after an exception. Independent verification checks all new trace/outcome hashes, all historical pointers, the original study manifest, direct legal-action cardinalities, raw/statistic agreement and exact paired tests by independently enumerating every fair binary allocation of the observed discordant pairs.

The immutable gap-closure export stores the cloud artifacts, both actual coding traces, readable reports, environments and failed diagnostic log. Cached model/adapter/tokenizer/teacher hashes were independently rechecked unchanged. Existing v1 and v2 archives remain preserved. Restore using `scripts/research_v2/evidence_archive.py --verify-existing`; the release manifest records the concrete archive path and checksum. Then:

```sh
python scripts/research_v2/verify_gap_closure.py --package work/gap_closure_restore --old-study work/numeric_study_restore/mechanism_study --historical-evidence work/cloud_research_v1 --receipt work/gap_closure_verification.json
python scripts/research_v2/coding_telemetry.py --trace work/gap_closure_restore/coding_probe_c/trace.json --report work/coding_trace_report.md
```

The historical evidence and numeric study are restored by the already validated v1 and numeric archive utilities. Verification performs no neural inference. Live diagnostics require existing checksum-pinned local weights; absent weights are an explicit dependency, not permission for a download.

This phase establishes bounded mechanism evidence and targeted coding observability. Cross-domain independent replication, all-adapter live coverage and historical GPT/Gemini comparison remain unrun. Exact reference releases, access and an authorized API budget are still unspecified. No Research v2 completion tag or general-model superiority claim is justified. No public website was built.
