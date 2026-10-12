# V3.1 focused diagnostic review

**NO-GO for scaling or claiming a superior interaction mechanism. GO only for a small measurement/equivalence diagnostic.** The corrected pilot executed successfully and independently verified 780 horizon-conditioned forecasts,32 selected-action outcomes and 22 checkpoint indices. There are 26 independent development scenes; repeated seeds, variants and horizons are not independent samples. Sealed final trajectories were not decoded or evaluated. No further native run or full study has been launched.

Corrected run: `native-pilot-1791816435786858000`; preserved failed run: `native-pilot-1791815958222838000`. Native Apple MPS, float32, PyTorch 2.14.1; no CPU neural fallback. Original model/dataset/gates are unchanged.

## Every original gate

| Gate | Frozen threshold | Motion observed / result | Interaction observed / result |
| --- | --- | --- | --- |
| Finite outputs | All finite | All / PASS | All / PASS |
| Mean 32-step stress error | ≤10 | 2.8930 / PASS | 1.0058 / PASS |
| 32/8 error growth | ≤4 | 3.2570 / PASS | 1.0570 / PASS |
| Improvement versus matched | ≥10% | 1.52% / FAIL | -0.31% / FAIL |
| Nominal 95% coverage | ≥80% | 75.40% / FAIL | 68.62% / FAIL |
| Planning success difference | ≥0 | 0.00% / PASS | 0.00% / PASS |
| Pilot runtime | ≤600 seconds | 35.836 seconds overall / PASS | Same overall timing / PASS |

The gate runtime excludes process startup and independent replay. Launcher stage times were 36.463 seconds for the pilot and 0.801 seconds for verification. Passing finite/error-growth gates does not mean indefinite dynamical stability: the shared bounded increments prevent explosive feedback, but do not guarantee bounded trajectories as time tends to infinity.

## Per-seed matched comparison

Errors use the frozen fixed unit scales, not the old V3 training-standard-deviation normalization. Parentheses contain matched error. Each stress row has seven independent scenes, one per labeled condition; each IID row has six.

| Family | Seed | Panel |8 steps: candidate (matched) |16 steps |32 steps |
| --- | ---: | --- | ---: | ---: | ---: |
| motion | 11 | iid | 0.7985 (0.6041) | 0.7945 (0.7197) | 2.5061 (2.5393) |
| motion | 11 | stress | 0.8965 (0.8068) | 1.3109 (1.3134) | 2.9402 (2.9178) |
| motion | 23 | iid | 0.6337 (0.6848) | 0.7349 (0.7061) | 2.4580 (2.4951) |
| motion | 23 | stress | 0.8800 (0.9953) | 1.2709 (1.3946) | 2.8457 (2.9573) |
| interaction | 11 | iid | 0.4729 (0.4735) | 0.4261 (0.4282) | 0.4167 (0.4106) |
| interaction | 11 | stress | 0.9544 (0.9287) | 0.9584 (0.9186) | 0.9985 (0.9494) |
| interaction | 23 | iid | 0.5475 (0.5629) | 0.4768 (0.4976) | 0.4621 (0.5319) |
| interaction | 23 | stress | 0.9487 (0.9498) | 0.9359 (0.9432) | 1.0131 (1.0560) |

At 32 steps, candidate improvements flip sign across seeds: motion−0.77%/+3.78%, interaction−5.18%/+4.06% for seeds 11/23. Seed-averaged stress improvements are 1.52% and−0.31%; paired scene-bootstrap 95% intervals are−2.80% to 5.16% and−14.69% to 14.19%. Neither supports a reproducible 10% advantage.

A posthoc **nonneural constant-velocity prior control**, computed only from public initial states, scored 5.9446 motion and 2.4758 interaction at 32-step stress evaluation. Both trained models beat that prior. This supports learned contributions beyond the integration prior on these scenes, but does not identify which shared redesign caused them or establish final OOD generalization. This control does not replace any frozen gate.

## What explains the NO-GO

- **Divergence:** no catastrophic finite explosion through 32 steps. Both models meet the declared growth/error limits. Stabilization is shared and cannot be attributed to the candidate-only coordinate transform.
- **Object count:** the three-object configuration case is not catastrophic. Candidate versus matched 32-step error was 2.748/2.782 and 2.657/2.798 in motion,1.576/1.821 and 2.362/2.741 in interaction for seeds 11/23. Each family has only one such physical scene, so this is not population-level object-count generalization evidence.
- **Physical prediction:** candidate stress position MSE averages 0.293 motion/0.440 interaction, while velocity MSE is 5.493/1.572. Velocity errors dominate. Floor penetration is zero for the candidate, but total-momentum error against executed truth remains 24.420/7.079. These measure prediction accuracy, not conservation in a closed system; external forces/gravity may legitimately change momentum.
- **Planning:** candidate and matched each succeed 1/4 per seed/family—2/8 descriptive outcomes per family, reusing four scenes. They choose identical actions in 15/16 paired decisions. Planning equality passes the original gate but shows no advantage; the small sample cannot establish good policy quality.
- **Genuine primary failure:** candidate prediction accuracy does not improve enough over matched, and signs reverse with initialization. Correcting the uncertainty calculation cannot retroactively change that fact.

## History, concepts and memory

Changes below are component-minus-candidate 32-step stress MSE; negative is better.

| Component condition | Motion seed 11 /23 | Interaction seed 11 /23 | Interpretation |
| --- | ---: | ---: | --- |
| no_history | -0.01465 / +0.02547 | +0.11307 / +0.04495 | History helps interaction in both seeds; motion is mixed. |
| concepts | -0.05226 / -0.01617 | +0.01150 / -0.09637 | Mixed; no reproducible cross-family benefit. |
| memory | -0.06252 / -0.05428 | -0.00120 / -0.00709 | Same favorable direction in all four cells, but small and uncertain. |

Memory reduces mean stress error by 2.02% motion and 0.41% interaction, but paired 95% intervals include zero (−0.35% to 4.24%,−3.79% to 3.70%). No optional component is promoted as a validated improvement. Two raw frames remain a provisional partial-observation design, not proof of universal history benefit. Memory has zero extra gradient updates, approved training-only keys and no evaluation updates.

V3 validation-only diagnosis supports removing the old unbounded derived features:32-step MSE was 43.289 motion and 3.331 interaction; removing concepts reduced these to 16.492/0.919, removing history to 6.360/0.914. Removing memory gave 42.992/5.582; removing relationships 104.261/56.763. These are 6 old validation scenes per family, not old final test data. Ablations can alter optimization and effective capacity, so they do not prove a single causal explanation.

## Measurement and implementation issues

1. **Covariance propagation is misspecified.** The implemented position variance is a simple cumulative sum of per-step dt² velocity variance. Under independent velocity innovations, position variance at step k must instead include squared propagation weights(k−i+1)² and position/velocity covariance. At two steps with unit innovation variance, position variance is 5dt², not 2dt². Independent closed-form and Monte Carlo tests verify this discrepancy. The training likelihood uses the same approximation, so this is not merely a display issue.
2. **Calibration failure is coordinate-specific.** Candidate position coverage is about 30–67%, while velocity coverage is 98–100%. A posthoc linear-kinematic covariance diagnostic produces 98.3–99.8% overall coverage and wider intervals. That overcoverage is not calibration success, ignores the neural feedback Jacobian, and leaves original gates failed. Variance-head semantics and proper interval scores must be resolved before more training.
3. **Planner risk uses last-step variance**, not propagated terminal position covariance. Actual actions/outcomes were genuinely executed and independently replayed, so success counts remain valid. The advertised uncertainty penalty is an incomplete approximation.
4. **The architecture comparison is a reparameterization.** Both models already have shared object-centric pairwise messages. For relative columns J, candidate edge input is[f_self; f_neighbor−f_self]. Its first linear layer is exactly reproduced by matched weights W_self[:,J]←W_self[:,J]−W_neighbor[:,J]. Subsequent layers and integration are identical. Thus the function classes coincide; initialization, Adam coordinates and weight decay can differ. The observed comparison tests encoding/optimization, not distinct interaction architecture. No claim of superior object-centric reasoning is supported.
5. **Horizon labeling is not a distinct physical shift.** The generator replaces scene horizons with 32 for every pilot scene, including the case labeled horizon. Length generalization is the 8/16/32 prefix factor, not an additional exclusive OOD scene stratum. Future protocols must remove that misleading stratum label.
6. **Partial observability limits identification.** Hidden mass/materials are omitted, previous actions are not encoded, and spring context supplies strength without explicit attachment identity. Two state frames cannot guarantee unique hidden-parameter/topology inference. This is an information limitation, not a reason to supply hidden simulator answers.
7. **Import isolation is weaker than claimed.** Evaluation imports the sequence helper from training, which transitively imports the dataset/engine module. Forecast functions do not query that engine and receive no private targets, but the direct-import AST test does not prove process-level isolation. A future version should move tensor rollout into a predictor-only module and test transitive dependencies.
8. **Bootstrap limitations:** resampling the seven stress scenes changes the fixed stratum mixture; one scene per condition cannot estimate within-stratum variation. These pilot intervals are exploratory, not trustworthy final population intervals. A follow-up should preserve strata during resampling and avoid pooling frames/seeds.
9. **Engineering failure corrected:** the first run lacked a memory artifact directory. The V3.1 wrapper now creates it; a weights-only roundtrip regression passes. All 22 corrected checkpoints load safely; native 6+6 versus 12-step multi-step training had exactly identical parameters and optimizer state. No frozen V3 evidence, model settings, gates or sealed datasets changed.

## Smallest justified follow-up—proposal only

First resolve uncertainty semantics and confirm the linear reparameterization; do not train a larger model or repeat the full study. The fresh proposal in configs/v 3_1_physics/followup-proposal.json reuses frozen checkpoints, with **zero training updates**, two seeds and independent development seed 20261051. Six IID and twelve stress scenes per family (two per six physical shifts) yield 14432-step neural forecasts;8/16/32 prefixes are scored from the same trajectories. The current sealed final datasets remain unused.

Compare original versus explicitly declared linear-innovation covariance on identical means; report per-coordinate coverage, widths and proper interval score. Check a function-equivalent remapped matched checkpoint on MPS. Stop on any covariance fixture failure, nonfinite output, changed mean, failed numerical equivalence, checkpoint/source mismatch or final-dataset access. Wider intervals alone never authorize scaling. Retraining with corrected uncertainty semantics or a genuinely different interaction mechanism requires a separate subsequent protocol.

**No new native run is requested.** GO only for this narrow diagnostic proposal; NO-GO for scaling the current candidate, claiming final OOD improvement, retaining unverified concepts/memory, or claiming cross-domain transfer.

## Evidence and verification

Both native directories,128 frozen scenes and all raw/checkpoint evidence are archived in experiments/v 3_1_physics/evidence. The archive contains 1127 files,20,979,183 compressed bytes; fresh restoration verified every file hash. All 758 files tracked at the preserved V3 base commit are unchanged. Sealed files were hashed/copied as opaque integrity/archive bytes only; they were never deserialized into evaluation/training targets. Eleven new tests pass, including the targeted save regression and independent covariance checks. Native permutation, masked-parameter invariance and nonperfect-zero-weight probes also passed. No physical-world recording or cross-domain transfer experiment ran.
