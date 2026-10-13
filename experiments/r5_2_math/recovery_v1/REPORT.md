# R5.2 forensic recovery

**POST-HOC RECOVERY FROM FROZEN PREDICTIONS — NO_GO.** Original FAILED_CAMPAIGN remains unchanged. No neural forwards, optimizer steps, new tasks, or threshold changes.

Root cause: the inner digit/carry loop overwrote the outer condition variable. All 3,060 saved records say `carry`; the first recurrent/development group was empty and `good / len(rr)` raised ZeroDivisionError. Independently, original replay compared base verifier fields with an augmented dictionary containing context_diagnostics. Four nonneural regression tests reproduce both defects and reject incomplete/order-mismatched or tampered records.

Reconstruction: authenticated source dictionary order recurrent/reset_hidden/memory_only/zero_weight/zero_weight_memory; 204 manifest-ordered tasks per block and 1,020 records per seed. Every identity, split, family and seed matched. All three deterministic controls reproduced full traces and diagnostics independently. Neural blocks contain neural diagnostics; their attribution relies on frozen order, not intrinsic model identifiers. No accuracy was used for labeling.

Integrity: 25 run files and original archive match their preservation receipt. Six CPU-inspected checkpoints pass hashes, finite tensors, parameter counts, losses and optimizer states; all six progress logs contain exactly updates 1–800. Qualification independently reconstructs DATA_READY: add/sub/mul labels 3397/815/1726, signs 372/396, 38 withheld-context input exposures, 364/5497 OOD memory misses (6.6218%). 768 training and 204 reference traces independently reproduce.

| Seed | OOD /144 | Length /24 | Carry/borrow /24 | Sign /24 | Unseen /24 | Intermediate |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 12 (8.33%) | 1 | 2 | 3 | 2 | 83.02% |
| 23 | 18 (12.50%) | 4 | 9 | 3 | 0 | 83.37% |
| 47 | 6 (4.17%) | 3 | 1 | 2 | 0 | 78.84% |

| Control | OOD correct, seeds 11/23/47 (each /144) | Mean exact |
|---|---|---:|
| recurrent | 12/18/6 | 8.33% |
| reset_hidden | 35/35/38 | 25.00% |
| memory_only | 31/31/31 | 21.53% |
| zero_weight | 0/0/0 | 0.00% |
| zero_weight_memory | 31/31/31 | 21.53% |
| Procedural reference | 144/144/144 (same independent tasks) | 100% |

| Frozen gate | Observed | Required | Result |
|---|---:|---:|---|
| overall | 0.083333 | >= 0.65 | FAIL |
| length | 0.111111 | >= 0.80 | FAIL |
| carry_borrow | 0.166667 | >= 0.60 | FAIL |
| sign | 0.111111 | >= 0.80 | FAIL |
| unseen_context | 0.027778 | >= 0.50 | FAIL |
| intermediate | 0.817453 | >= 0.90 | FAIL |
| reset_gain | -0.166667 | >= 0.10 | FAIL |
| memory_controls | -0.131944 | >= 0.10 | FAIL |
| seed_reset_gains | -0.222222 | > 0.00 | FAIL |

Full family, development, strict-rollout, stage/sign, local seen/unseen metrics and per-seed Wilson intervals are in summary.json; index-to-label provenance is in label-map.json. Independent aggregate counts and separately computed gates match the frozen analyzer. There are 144 independent OOD tasks, 24 per family; repeated seeds do not increase this denominator.

Interpretation: numeric learning occurred relative to zero weights, but systematic generalization and useful recurrent-state benefit are not supported. Reset-hidden and demonstration controls outperform recurrent inference. Local correctness does not establish whole-task rule learning; signs and accumulated transition errors remain material. Symbolic correctness is excluded from neural metrics.

Limitations: no neural re-forward was allowed, so stored probability/write consistency and archive provenance do not independently regenerate predictions from checkpoint weights. Frozen loop order is the proof for neural condition attribution. This remains post-hoc recovery of a failed original campaign, not a clean successful execution. No retraining or further experiment is authorized by this recovery.
