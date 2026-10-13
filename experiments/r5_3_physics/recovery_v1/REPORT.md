# R5.3 forensic recovery

**POST-HOC RECOVERY FROM FROZEN EVIDENCE — scientific NO_GO.** Original native FAILED_CAMPAIGN remains unchanged. No neural forward, forecast, optimizer step, new scene identity or changed gate.

## Exact failure

First audited mismatch: scene `5f326a293ab9a2cab64369c8f379cb12e146c3ef0647af23ccd3ff2d45f061e5`, method `structured`, seed 11, horizon 8, metric `interval_score`. Saved **0.72815668489784**, recomputed **0.7281566829769872**; absolute difference **1.9208528101444244e-09**, relative difference **2.6379663195894701e-09**.

Original audit.py requires exact equality for every recomputed metric. Under NumPy2.3.5, quantile bounds from float32 particles remain float32. The interval-score expression computes upper−lower in float32 before adding float64 target-penalty terms. JSON preserves those numeric values losslessly but not array dtype; replay constructs float64 bounds and subtracts in float64. This is a different numerical evaluation path, not loss of decimal digits or simulator nondeterminism.

All2,496 cells (19,968 metric comparisons) were inspected. Exactly1,245 interval-score discrepancies: structured623,matched622. Maximum absolute difference1.714215613901615e−8; maximum relative difference1.644770564647299e−8. All other7 metric types match exactly. Casting losslessly representable learned bounds back to float32 reproduces ALL recorded metrics exactly; every difference is also independently explained by the mean rounded-width minus float64-width.

Recovery tolerance is derived per cell before comparison: absolute mean half-ULP of float32 width plus64·float64-epsilon times maximum score-component magnitude; relative64·float64-epsilon (1.4210854715202004e−14). It is not fitted to the largest observed discrepancy. Native-dtype reproduction remains exact; identity,hash,event count,coverage/contact discrete comparisons,actions,selected index and success remain exact. Four regression tests reproduce the failure and reject altered evidence.

## Evidence and verification

All1266 original files and original93,537,798-byte archive match the native preservation receipt and archive restoration hashes. Six checkpoints pass safe CPU tensor loading,hashes,identities,parameter counts,finite weights/moments,160 losses and160 optimizer steps each (960 total). Native engineering/preflight receipts are authenticated; their neural checks were not rerun.

Coverage:52 unique evaluation scenes (12 IID+14 OOD per family),624 complete forecast/planning condition executions,2,496 unique horizon cells and5,616 saved hypothetical planning trajectories. All nine candidate risk scores and selected actions rederive exactly. Every chosen32-action sequence and all624 actual Pymunk execution traces replay exactly. Existing scene targets/observations/actions are replayed twice with exact equality and hashed separately. No new scene was generated.

Qualification receipt remains DATA_READY:48 training scenes,175 contact frames,12 spring scenes,analytical baseline MSE0.026223445401887535. Training scene/contact/spring counts replay independently. Baseline-MSE headroom is the authenticated receipt only: its forecast was not regenerated.

Complete native evidence is preserved in evidence/part-000.bin through part-002.bin; concatenation reproduces the ORIGINAL archive bytes/hash. original-inventory.json binds every original file and archive; MANIFEST.json binds recovery artifacts.

## Primary32-step scientific measurements

Each cell averages seeds within independent scenes; OOD denominator14 scenes/family, IID12. Repeated seeds and horizons do not increase these counts.

| Family | Method | IID MSE | OOD MSE | OOD velocity MSE | Coverage | Interval score | OOD planning |
|---|---|---:|---:|---:|---:|---:|---:|
| motion | structured | 0.06872261 | 0.07904477 | 1.21804194 | 97.14% | 1.99337946 | 78.57% |
| motion | matched | 0.07044487 | 0.07787777 | 1.22584773 | 97.67% | 1.91563176 | 80.95% |
| motion | v3_1_frozen | 0.35776784 | 0.42898787 | 5.16700624 | 99.90% | 14.80746716 | 17.86% |
| motion | analytical_contacts | 0.05219996 | 0.06511602 | 0.97928959 | 59.46% | 6.46833982 | 78.57% |
| motion | analytical_available | 0.68987236 | 1.26941174 | 16.12234184 | 54.74% | 37.14443922 | 50.00% |
| motion | constant_velocity | 1.52165683 | 1.32998853 | 8.74087952 | 18.19% | 51.59422705 | 14.29% |
| motion | persistence | 1.19991051 | 1.31277210 | 8.31697326 | 8.04% | 56.25919821 | 14.29% |
| interaction | structured | 0.06930568 | 0.06111424 | 0.39076874 | 94.05% | 2.37794770 | 35.71% |
| interaction | matched | 0.04066502 | 0.05315167 | 0.32889433 | 95.72% | 2.28500790 | 35.71% |
| interaction | v3_1_frozen | 0.10018056 | 0.30289711 | 1.46262104 | 98.31% | 6.76233906 | 35.71% |
| interaction | analytical_contacts | 0.02774147 | 0.05139072 | 0.33068980 | 41.19% | 6.31280646 | 35.71% |
| interaction | analytical_available | 1.00044260 | 0.97283504 | 4.62136715 | 22.94% | 34.57465298 | 0.00% |
| interaction | constant_velocity | 0.98076029 | 0.97029531 | 4.59783816 | 22.27% | 34.30475086 | 0.00% |
| interaction | persistence | 0.25284810 | 0.31159100 | 3.99717901 | 15.07% | 27.32978048 | 0.00% |

| Family | Seed | Structured IID MSE | Structured OOD MSE | OOD success /14 | OOD coverage |
|---|---:|---:|---:|---:|---:|
| motion | 11 | 0.06353508 | 0.07514219 | 10/14 | 96.82% |
| motion | 23 | 0.06820282 | 0.07768343 | 11/14 | 97.74% |
| motion | 47 | 0.07442994 | 0.08430870 | 12/14 | 96.85% |
| interaction | 11 | 0.06610762 | 0.05628905 | 5/14 | 94.05% |
| interaction | 23 | 0.06850816 | 0.06956162 | 5/14 | 94.13% |
| interaction | 47 | 0.07330125 | 0.05749206 | 5/14 | 93.98% |

All method/seed/family/split/horizon measurements are in per-seed.json; per-scene aggregates,OOD strata,and confidence intervals are in summary.json.

| Original gate | Required | Motion observed | Interaction observed |
|---|---|---:|---:|
| ood_mse_ratio | ≤0.80 and CI upper<1 | 1.21390669 | 1.18920766 |
| iid_mse_ratio | ≤1.05 | 1.31652609 | 2.49826971 |
| velocity_mse_ratio | ≤1.00 | 1.24380158 | 1.18812851 |
| coverage | 0.85–0.99 | 0.97135417 | 0.94053819 |
| interval_score_ratio | ≤1.00 | 1.04058593 | 1.04067374 |
| contact_accuracy | ≥0.80 | 0.96205357 | 0.96875000 |
| floor_penetration | ≤0.05 | 0.00000000 | 0.00000000 |
| planning_gain | ≥0.10 and CI lower>0 | -0.02380952 | 0.00000000 |
| stress_mse | ≤100 | 0.15655174 | 0.23900464 |
| seed_structured_minus_matched_mse | all three<0 | +0.00338341/-0.00242996/+0.00254757 | +0.00758358/+0.00781111/+0.00849302 |

motion: OOD MSE-ratio95% paired scene-bootstrap CI **[0.9977816759589191, 2.330006389001899]**; planning-gain CI **[-0.26190476190476186, 0.04761904761904778]**. Passed:coverage,contact,floor,stress. Failed:OOD/IID MSE,velocity,interval score,planning,and all-seed matched superiority.

interaction: OOD MSE-ratio95% paired scene-bootstrap CI **[1.044259694458173, 3.19337955222952]**; planning-gain CI **[0.0, 0.0]**. Passed:coverage,contact,floor,stress. Failed:OOD/IID MSE,velocity,interval score,planning,and all-seed matched superiority.

Both the original frozen analyzer and a separate scene/bootstrap/gate implementation agree. Recomputing with promoted float64 bounds also leaves all gate booleans unchanged. No thresholds were relaxed.

## Interpretation and limits

The structured model is less accurate than approximate analytical contacts/springs and does not reliably beat the matched learned model. Motion OOD MSE is21.39% worse than the strongest control; interaction18.92% worse. Interval scores are4.06%/4.07% worse than matched despite acceptable marginal coverage. Motion planning trails matched by2.38 percentage points; interaction ties strongest controls. Internal structure has not demonstrated the predeclared superiority.

On this new matched panel, both new learners outperform the frozen V3.1 references, but that comparison changes architecture,training data and budget; it does not invalidate historical NO_GO results or establish which mechanism caused improvement. Stability/coverage engineering passes are distinct from scientific superiority.

Limitations: no predictions were regenerated from checkpoint weights,so that correspondence relies on authenticated native provenance. Original forecast target arrays were not separately archived; targets derive from twice-replayed authenticated scenes/pinned Pymunk rather than comparison with original target-array bytes.32 particles limit95% tail resolution;14 OOD scenes/family and2 per stratum limit bootstrap/stratum precision. Contact accuracy is prevalence-sensitive and does not alone establish collision modeling. Planning success is fixed-target simulated performance,not broad physical intelligence. Original campaign FAILED_CAMPAIGN is historical fact; this is a post-hoc recovery,not a clean native run.
