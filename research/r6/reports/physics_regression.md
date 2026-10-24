# Historical physics regression diagnosis

Read-only checkpoint inference and autograd diagnostics; zero optimizer updates. Raw derivations, worst cases, reference values and verified checkpoint hashes are in `historical_physics_diagnosis.json`. All nine preserved cloud models were loaded with hash checks. Historical results and source were not modified.

## Matched cloud versus historical R5.7c

| Seed | Condition | CUDA MSE | CUDA minus MPS MSE | IID MSE | OOD MSE |
|---|---|---:|---:|---:|---:|
| 17 | graph-concept-on | 264.16429 | +0.0065778239 | 1.2973103 | 527.03127 |
| 17 | graph-concept-off | 271.77264 | -0.03144225 | 1.7052399 | 541.84003 |
| 17 | matched-gru | 261.73004 | -1.9446339e-05 | 1.4045835 | 522.0555 |
| 29 | graph-concept-on | 255.20359 | -0.0004489139 | 2.1226122 | 508.28456 |
| 29 | graph-concept-off | 255.51447 | -1.7478881e-05 | 2.095164 | 508.93378 |
| 29 | matched-gru | 240.86094 | -3.6344899e-05 | 1.9794026 | 479.74249 |
| 43 | graph-concept-on | 263.63419 | +9.2774203 | 1.6671442 | 525.60124 |
| 43 | graph-concept-off | 245.41168 | +0.52910369 | 0.8476015 | 489.97576 |
| 43 | matched-gru | 230.67707 | +0.010279523 | 1.2545946 | 460.09955 |

Positive deltas regress: seed 17 concept-on, and every seed 43 architecture. Seed 43 concept-on is the largest change (+9.2774202683 MSE). These paired descriptive differences are not significance tests. Only three exploratory seeds are available per condition; no significance test was performed.

## Scaling, units and distribution

The preserved label function is Δv = F Δt / m for each axis. Training and evaluator source use the same unscaled physical target, and the CPU evaluator independently recomputes it. The R5.7/R5.7c input encoder applies sign(x) log(1+|x|) separately to raw force, mass, time, velocity and impulse observations; it never supplies F Δt/m. This is input compression, not output normalization. The tanh hidden physics head has linear output; no output range constraint was changed during migration. Values are treated implicitly as SI, but historical records have no explicit unit declaration or unit-conversion evaluation. Unit correctness cannot be claimed from those records.

Each frozen training seed has 42 physics examples, masses 1–9 kg and times 0.1–1 s. Each final set has 48 fresh-IID cases (mass 1–9, time 0.1–1) and 48 low-mass/long-time OOD cases (mass 0.25–0.75, time 1.1–2). Final forces span ±12 versus training integer forces ±10. The OOD law needs much larger velocity changes; combining IID and OOD obscures the dominant failure. Input log compression and shallow shared representations are plausible difficulty contributors, not an established causal explanation of migration drift.

The source freeze and training records match, but Mac Torch 2.14.1/MPS and Linux Torch 2.11.0+cu128/CUDA differ. Initial seed-17 embedding differences up to 1.55e-6 and later optimization divergence prevent attributing the observed changes to CUDA alone. The completed same-weight CPU/CUDA inference equivalence check passed; small arithmetic differences during training can still produce different final parameters. There is no evidence of target-scale drift or checkpoint corruption. The established deficiency is poor extrapolation, including occasional reversed signs.

## Intervention and gradient evidence

| Seed | Condition | force-x derivative sign | force-y derivative sign | mass derivative sign | time derivative sign | head gradient norm |
|---|---|---:|---:|---:|---:|---:|
| 17 | graph-concept-on | 1.000 | 1.000 | 1.000 | 0.500 | 4.8843 |
| 17 | graph-concept-off | 1.000 | 1.000 | 0.917 | 0.500 | 3.46794 |
| 17 | matched-gru | 1.000 | 1.000 | 0.958 | 0.167 | 6.11033 |
| 29 | graph-concept-on | 1.000 | 1.000 | 0.500 | 0.583 | 1.79403 |
| 29 | graph-concept-off | 1.000 | 1.000 | 0.542 | 0.583 | 1.83418 |
| 29 | matched-gru | 1.000 | 1.000 | 1.000 | 0.500 | 5.2344 |
| 43 | graph-concept-on | 1.000 | 1.000 | 0.500 | 0.458 | 7.37564 |
| 43 | graph-concept-off | 1.000 | 1.000 | 0.625 | 0.583 | 7.86876 |
| 43 | matched-gru | 1.000 | 1.000 | 0.833 | 0.417 | 5.5035 |

The derivative diagnostic uses a fixed independent 12-case grid: masses {0.7,2.3,6.1}, times {0.2,0.8}, opposite force pairs ±(1.7,−2.1), velocity (0.3,−0.7). Forward finite differences with ε=0.01 are compared against analytic derivatives. Force sign agreement uses the perturbed axis; mass/time use both axes. Correct derivative sign is necessary but does not establish the correct magnitude or cross-axis invariance. Nonzero output-to-head autograd norms confirm gradient connectivity only; they do not establish learning. No optimizer or parameter modification occurs.

| Seed | Condition | doubled mass MSE | doubled time MSE | +1 force-x MSE | +1 force-y MSE |
|---|---|---:|---:|---:|---:|
| 17 | graph-concept-on | 60.9169 | 1103.48 | 266.427 | 273.745 |
| 17 | graph-concept-off | 64.1437 | 1121.65 | 275.693 | 280.89 |
| 17 | matched-gru | 60.3966 | 1102.36 | 263.608 | 270.909 |
| 29 | graph-concept-on | 56.7502 | 1083 | 258.253 | 257.319 |
| 29 | graph-concept-off | 56.896 | 1083.82 | 258.549 | 257.609 |
| 29 | matched-gru | 51.6961 | 1061.09 | 243.807 | 242.258 |
| 43 | graph-concept-on | 61.1041 | 1096.42 | 262.792 | 250.291 |
| 43 | graph-concept-off | 52.2217 | 1066.04 | 244.706 | 234.224 |
| 43 | matched-gru | 46.7308 | 1038.72 | 228.659 | 220.447 |

Each preserved intervention has 96 examples. Predictions changing in response to an intervention is not evidence that Newton’s law is respected. Doubled-time error is especially large. These intervention MSEs use the modified analytic reference, not a comparison with the original target.

## Concrete worst failure

Seed 43 concept-on, identity `bd7f52652f097a8d2187a4b939c47187900077a6911427f145dc855dec466c0c`: mass 0.29747237 kg, time 1.6301626 s, force [10.23931294537163, -10.224452858369588] N. Reference Δv = [56.11191728317411, -56.03048330638993]; prediction = [-1.4151248931884766, 0.5486043095588684]; MSE 3255.2769, both component signs wrong. This single case contributes about 33.9091 to aggregate MSE.

## R6 evidence-backed representation change

R6 introduces explicit SI unit metadata; nonzero initial position and velocity; ordered two-stage 2D force/time observations; a third-stage mask for structural OOD; fixed human-unit input scales; and four state outputs (vx,vy,x,y). All model features remain raw observed state, mass, force and time. Targets are produced by sequential kinematics and checked with independent exact rational impulse-moment integration. Train/dev/IID contexts use a seed-independent hash partition; final OOD adds a third stage and holds out force/mass/time ranges. Interventions double force, mass and time, and split a constant-force interval to check time-composition invariance. This directly exposes the missing behavior without hiding analytical corrections inside predictions.

This is a testable engineering change, not proof of improved physics. R6 must report state-component MAE/MSE, IID versus structural/range OOD separately, and intervention response error. Full CBSE mechanics or Class 12 competency is not established. Fixed scales avoid test-derived normalization; no fitting has run on this Mac.
