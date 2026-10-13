# R5.1 verified DATA_READY; learning pilot not authorized

Native run `results/r5_1/screen-2cac9d0fcef076df`, committed freeze `ed47341`. Independent read-only replay reproduced **81/81 outcomes** and all six original gates. **27 composition outcomes remain unexecuted.** No training or checkpoints were created.

## Native engineering evidence

Torch 2.14.1, MPS, float32. Actual model count **6,967 parameters**. Batch 2, padding 7, relation tensor `[2,7,7,3]`, action `[2,8]`.

| Domain | Raw entity tensor | Encoder width after mask concatenation | Effect tensor | Minimum recorded active parameter gradient | Action sensitivity | Relation sensitivity |
|---|---|---:|---|---:|---:|---:|
| Mathematics | [2,7,6] | 12 | [2,7,1] | .002406249987 | .005513191224 | .031608879566 |
| Coding | [2,7,10] | 20 | [2,7,1] | .001337935333 | .006195351481 | .041244849563 |
| Physics | [2,7,8] | 16 | [2,7,4] | .001444608439 | .021052256227 | .068258419633 |

Precondition shape `[2]` for all domains. Effect/precondition gradients and entity/relation/action input gradients are finite and nonzero; inactive-domain/head gradients are absent as expected. Recorded permutation errors are exactly 0 for all domains. The frozen native tests assert masked-input isolation and exact unchanged initial/final parameters, with zero optimizer steps. This independent audit checked recorded shapes/norms and the exact executed source; it did not regenerate native gradients. Underlying gradient tensors and initial model weights are not separately archived. This is engineering connectivity, not learning evidence.

## Exact qualification values

CE is the frozen macro changed-entity MSE. All development cohorts have **12 valid, 6 invalid, 12 changed-valid** tasks; all qualification cohorts have **6 valid, 3 invalid, 6 changed-valid**. Unchanged-entity persistence/local errors are zero in all six cohorts.

| Cohort | Tasks | Persistence CE | Local-action CE | Relative reduction | Whole-graph persistence MSE |
|---|---:|---:|---:|---:|---:|
| Math development | 18 | 4.862316666667 | 4.743033333333 | .024532201728 | 1.620772222222 |
| Coding development | 18 | .010000000000 | .009500000000 | .050000000000 | .004444444444 |
| Physics development | 18 | .025526355472 | .014959845882 | .413945093004 | .017017570315 |
| Math qualification | 9 | 3.727633333333 | 3.631533333333 | .025780432625 | 1.242544444444 |
| Coding qualification | 9 | .004500000000 | .004100000000 | .088888888889 | .002000000000 |
| Physics qualification | 9 | .033107832451 | .010218034171 | .691371092150 | .022071888301 |

Frozen thresholds: development >=12 valid, >=6 invalid, >=10 changed-valid; qualification >=6 valid, >=3 invalid, >=5 changed-valid. Both splits: persistence CE >=.0005; local-action CE >=.0002; local reduction >=.02. Coding operator balance is exactly 4 valid/2 invalid per add/sub/mul in development and 2 valid/1 invalid each in qualification. All checks pass. Exact full-precision values remain in `experiments/r5_1/native_v1/summary.json`.

The 168 native evidence files restored with identical hashes. Original archive SHA-256, sources/runtime hashes and preservation inventory are recorded under `experiments/r5_1/native_v1`. V5's original **DATA_NO_GO** and its 92 qualification files, architecture, commit `851815f` and PR #9 remain unchanged. Readiness says a small learning test is now feasible; it does not establish accuracy, rule learning or useful transfer.

## Separately frozen first learning pilot

`configs/r5_1/pilot_v1/{protocol,data-binding}.json` binds the exact qualified development identities, untouched coding composition inputs and original readiness archive. Status **FROZEN_AWAITING_EXPLICIT_AUTHORIZATION**. No pilot launcher, optimizer update or composition execution is issued in this turn.

The compact experiment tests math+physics -> coding only. Three seeds (11,23,47), three learned conditions (domain-only B, shared C, randomized-source R): **9 fits**, each **120 updates x16 graphs**, Adam .003. Maximum total: **1,080 updates /17,280 graph presentations**. First 60 updates are coding for B or alternating math/physics for C/R; final 60 are coding for all. All use the same 18 coding development labels; auxiliary domains provide 18 development labels each. B repeats target data more; this is disclosed rather than claiming identical target encoder updates. No qualification labels enter optimization or checkpoint selection.

Architecture is unchanged. Frozen supervised loss balances changed and unchanged active entity MSE, plus .1 precondition BCE. Masks derive only from training labels; inference never sees executed effects/validity. Randomized control permutes paired effect/validity labels within each source, with exact assignments preserved and no search for a favorable permutation.

**9 independent coding composition tasks**, shared across optimizer seeds, form evaluation. All nine final predictions from every fit must be persisted before executing any composition target; independent WASI/AST outcomes and exact replay follow. The 18 math/physics composition outcomes stay unexecuted in this single-target pilot. Persistence, local-action and the perfect procedural reference remain reported; no claim of superiority above an expert algorithm.

All pass criteria are frozen: >=10% mean changed-MSE gain over both B and R; strict positive gain versus both for every seed; >=20% reduction versus persistence and >=10% versus local-action; each shared seed validity balanced accuracy >=.70; shared whole-graph mean no worse than either control; shared mean unchanged-entity and invalid-noop MSE each <=.0001; finite gradients/predictions and all verification PASS. A zero-error control makes improvement impossible, rather than receiving an epsilon-assisted gain.

Model+Adam checkpoints at updates 60/120 preserve identities, hashes, losses and compute. Final update120 only enters comparison. Verification inspects CPU tensors without neural forwards, independent targets, all saved metrics/gates and archived-file hashes; native checkpoint forward replay would be separately identified. No retry/resume, new seed, budget expansion or bitwise continuation claim. Any failure stops NO_GO; every pass permits independent confirmation only. This pilot cannot establish all-domain benefit or assess negative transfer to math/physics targets.

**Stop here. Explicit human authorization is required before pilot training.** No website changes.
