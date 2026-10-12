# Mindscape V3 physics: completed simulated study and handoff

Run: `native-full-1791811860380119000`. Qualified source hash: `1b74f4654ab1a6b1ea7f57abe5e66f8427f1a69250cd44ca5f1b4b546249b86a`. Protocol was frozen before native model results.

**Engineering GO:** the frozen simulated study completed and passed independent verification. **Scientific NO-GO:** the proposed concept/memory mechanism failed the preregistered 20% structural-OOD improvement target and should not be scaled up as a successful mechanism. REAL-WORLD VALIDATION: NOT RUN. PHYRE: NOT RUN. No reasoning breakthrough is claimed.

## Completed coverage and independent checks

Apple M5, PyTorch2.14.1, Python3.12.14 arm64; native MPS float32, batch128. All neural computation used MPS with fallback rejected. CPU work comprised simulation, preprocessing, serialization, replay and statistics. Seeds11,23,37 and trajectory budgets10,25,50,100,250 all completed.

- 90 dynamics fits and six model-free policy fits, each400 updates:38,400 gradient updates total.
- 6,000 primary +6,000 matched +6,000 component-ablation forecasts;1,200 nonneural forecasts;2,160 actual planning episodes. No required episodes are missing.
- Independent verification reconstructed400 IID/OOD scenes, recomputed all19,200 forecast metrics and independently replayed all2,160 selected actions. It used zero neural calls.
- All96 final checkpoint hashes, steps400 and experiment identities passed, including safe weights_only=True loading. Native pilot continuation independently demonstrated exact parameters, Adam state, memory, step12 and provenance after6+6 versus uninterrupted12 updates in both families.
- All1000 frozen dataset scenes were separately replayed exactly in CPU preparation. Pilot/main scene identities are disjoint; training receipts admitted only approved training IDs. Evaluation outcomes were not added to persistent memory. Frozen validation was not used for tuning.
- 101 cross-domain V3 tests and three subtests passed. Scoped Ruff passed. Historical574 pre-V3 files,667 prior-V3 files and all release tags remain unchanged. Original failed native pilot and all diagnostic evidence are preserved.

## Prediction results

Tables average each scene over three seeds and then average scenes. Each final-budget neural cell has100 independent scenes, not300 independent observations. Lower MSE is better. Contact accuracy is a separate frame-level diagnostic, not task success.

| Family | Variant | IID normalized rollout MSE | OOD normalized rollout MSE | OOD one-step MSE | OOD nominal95% coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| motion | mindscape | 90.0596 | 9.33957e+07 | 0.175049 | 36.8% |
| motion | matched | 13.4854 | 18.1353 | 0.453476 | 48.9% |
| interaction | mindscape | 4.55259 | 7.74368 | 1.15671 | 47.9% |
| interaction | matched | 2.25712 | 3.66973 | 1.15778 | 55.9% |

The motion final-budget error is about5.15 million times matched error. The paired scene-bootstrap95% interval for that worsening factor is about2.11–10.51 million. Interaction error is111.0% worse, with paired95% interval86.6–134.8% worse. Both preregistered comparisons fail. Bootstrap resamples100 scene clusters after seed averaging,2000 draws; multiple exploratory comparisons are unadjusted.

Best descriptive primary means across the five budgets: motion IID90.0596 at250 and OOD31,397 at10; interaction IID2.8950 and OOD4.6227 at10. These selections are post hoc descriptions, not replacement acceptance criteria. Smaller budgets contain catastrophic finite motion errors near1e20; all are retained in raw evidence and learning curves.

| Final-budget variant / nonneural control | Motion OOD MSE | Interaction OOD MSE |
| --- | ---: | ---: |
| mindscape | 9.33957e+07 | 7.74368 |
| matched | 18.1353 | 3.66973 |
| no_relationships | 167.389 | 85.6984 |
| no_concepts | 9388.72 | 4.73733 |
| no_memory | 9.31524e+07 | 8.23195 |
| no_history | 26.1186 | 4.09654 |
| no_feedback | 17.3483 | 2.35818 |
| persistence | 2.87679 | 3.0214 |
| constant_velocity | 2.26968 | 6.81757 |
| analytical_available | 4.24611 | 6.7678 |

The supplemental analytical contact/spring predictor scored motion OOD0.431907 and interaction OOD1.75947 on the same400-scene panel, using only visible attributes and nominal hidden parameters. It is a nonneural approximate equation baseline, not learned reasoning or an exact oracle. The core analytical_available baseline is deliberately limited to free flight and declared known gravity/force.

## Actual planning success

| Family / split | Mindscape | Matched dynamics | Model-free policy | Random | Simple | Analytical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| motion / iid | 9/60 (15.0%) | 10/60 (16.7%) | 16/60 (26.7%) | 1/20 (5.0%) | 0/20 (0.0%) | 5/20 (25.0%) |
| motion / ood | 8/60 (13.3%) | 11/60 (18.3%) | 16/60 (26.7%) | 0/20 (0.0%) | 1/20 (5.0%) | 5/20 (25.0%) |
| interaction / iid | 14/60 (23.3%) | 38/60 (63.3%) | 39/60 (65.0%) | 3/20 (15.0%) | 4/20 (20.0%) | 4/20 (20.0%) |
| interaction / ood | 20/60 (33.3%) | 32/60 (53.3%) | 32/60 (53.3%) | 3/20 (15.0%) | 8/20 (40.0%) | 8/20 (40.0%) |

Neural denominators60 share20 scenes across three seeds; control denominators20 are one fixed control panel. Rates are descriptive. Individual seed Wilson intervals are in analysis.json; paired scene-bootstrap intervals and success differences are in acceptance.json. Mindscape-minus-matched success differences are−1.7/−5.0 percentage points for motion IID/OOD and−40.0/−20.0 for interaction. The maintained-planning criterion also fails.

Each dynamics planner evaluated nine action candidates across24 hypothetical steps, made zero authoritative simulator queries during selection, then executed one chosen impulse. Actual execution used27 steps including initial history. Random/simple policies provide planning-removal controls; they are not an equal-compute architecture-only planning ablation. The model-free policy receives additional training-only action-choice labels and is not an equal-information dynamics baseline. Teacher generation executed121,500 unique simulator steps across the two families; receipt totals per seed repeat shared teacher work. Policy training used2400 updates and600,000 labeled action presentations.

## Resources, failure mechanisms and revised hypothesis

Native training/evaluation wall time was204.454 seconds. Summed dynamics fit time was67.020 seconds for36,000 updates and4,608,000 transition presentations:about537 updates/s or68,755 sampled transitions/s, including fit checkpoint/memory overhead. Six policy fits added2.322 seconds. World-model parameter count38,409; maximum sampled driver allocation96,305,152 bytes (91.8MiB). This is a receipt sample, not a continuously sampled peak or full process RAM. No paid service or large pretrained model was used. Hardware-audit synthetic MLP throughput is separately labeled and must not be substituted for these measurements.

Observed motion failures concentrate on configuration and previously unseen interaction strata: final-budget means454.9 million and212.3 million respectively. The largest single failure was configuration, seed23, normalizedMSE6.76 billion. Removing memory leaves similarly catastrophic error; memory is not the sole explanation. Removing relational or concept features substantially changes instability but does not deliver reliable OOD dynamics. Collapsing history is less unstable, consistent with recursively derived features becoming unreliable during free-running prediction. These are diagnostic associations, not proven causal attribution.

One-step motion error improves while long rollout error deteriorates. The current unconstrained autoregressive delta model, trained with one-step supervision, is insufficient to control compounding error when object count and interactions change. Contact accuracy near92–96% does not imply accurate collision timing, conserved momentum or goal attainment. Nominal95% interval coverage is only36.8% motion and47.9% interaction for final-budget Mindscape OOD, so uncertainty is seriously undercalibrated.

Revised hypothesis: object-centric, permutation-consistent pairwise messages with learned stability constraints and training-only multi-step rollout loss may generalize more reliably than derivative concept features plus fixed residual memory. First test that hypothesis on newly frozen training/validation and independent evaluation scenes, with the same matched supervision and explicit calibration controls. Do not tune a revision on the current final OOD panel. Test history/relationships separately before reintroducing memory or scaling model size.

Limitations: two compact simulated families, circles/floor only, at most three objects; no calibrated direct mass/friction estimator; aleatoric marginal variance ignores temporal correlations; finite outputs can still be arbitrarily wrong. No real-world recording or PHYRE experiment ran. Full isolated no-planning neural ablation was not scheduled; random/simple controls investigate removal of hypothetical selection. No-feedback deliberately permutes supervision and cannot be treated as a pure architecture ablation.

Planning model-free envelopes retain an inner hash from their initial random proposal while the outer evidence hash binds the actual selected policy action. Independent replay uses the outer-bound action and verifies its actual outcome. Some trace metadata also retains a legacy model_free_policy label. Condition/identity and actual actions are authoritative. These metadata defects did not change executed outcomes; repair them only in a new version and requalify that source.

## Reproduction and evidence handoff

Native commands are in commands.md. The full run used the qualified pilot receipt and remained on the same model-source identity throughout. Safe interruption recovery retains the original run directory, code and dataset and uses --resume; do not force source-incompatible checkpoints or disable weights_only loading.

All25,670 physics evidence files, including frozen datasets, checkpoints, raw predictions, CPU controls and prior failed pilot evidence, are preserved in experiments/v3_physics/recovery_evidence_v1:14 checksum-bound32MiB parts,468,711,474 compressed bytes; restored expansion1,240,205,734 bytes. Restoration is bounded, rejects traversal/symlinks/corruption, and was tested in a fresh directory. Original artifacts remain present.

```sh
cd /Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3
export PYTHONPATH=src:work/physics-deps/site-packages
physics_python=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python
$physics_python -m pytest tests/v3 tests_math_v3 tests_coding_v3 tests_physics_v3 -q
$physics_python scripts/v3_physics/evidence.py restore --package experiments/v3_physics/recovery_evidence_v1 --destination work/new-physics-restoration
$physics_python scripts/v3_physics/verify.py --dataset work/new-physics-restoration/dataset-full-v1 --run work/new-physics-restoration/native-full-1791811860380119000/study --output work/recomputed-verification.json
$physics_python scripts/v3_physics/acceptance.py --dataset work/new-physics-restoration/dataset-full-v1 --run work/new-physics-restoration/native-full-1791811860380119000/study --output work/recomputed-acceptance.json
```

No primary episodes remain. The fixed simulated study is complete; the broader claim of reusable physical understanding is unsupported. Next research work requires a new versioned hypothesis and fresh holdouts.
