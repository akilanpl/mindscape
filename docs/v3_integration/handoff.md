# Mindscape V3 three-domain integration — verified engineering milestone

Decision: GO for engineering integration; no learned transfer claim. Learned cross-domain transfer is NOT YET ESTABLISHED. This four-request demonstration is not a generalization benchmark.

Native evidence: `results/v3_integration/native-1791821373113687000`; preserved byte-for-byte copies and SHA-256 manifest in `experiments/v3_integration/native_v1`. Earlier failed `native-1791820719027451000` remains untouched. Historical V1/V2/V3/V3.1 sources, checkpoints and evaluations are unchanged; the branch adds the existing mathematics dependency and new integration files only.

| Frozen demonstration | Actual result | Task outcome | Direct vs kernel |
|---|---|---|---|
| 487 + 376 | 833 (expected 863) | FAIL | Identical |
| 73 - 29 | 43 (expected 44) | FAIL | Identical |
| Python inclusive threshold repair | `item > 4` repaired to `item >= 4`; visible 2/2, private 4/4 | PASS | Same repair/outcome |
| Eight-step motion planning | Goal distance 0.315639, gate <=0.75; prediction MSE 1.224645 | PASS | Identical |

## Mathematics diagnosis

Checkpoint SHA-256 matches the original training receipt and manifest identity. The original 109,843-parameter procedural model is loaded without changing weights. Local inputs, feature layout, output-head splitting, sign head, bounded consistency ranking, normalization and digit serialization match the preserved implementation. Saved neural claims replayed through the original `run_problem` scaffold reproduce both failures, with each local input state checked. Independent arithmetic scoring occurs only afterward.

Addition: the units claim (7+6) is correct, but the tens state (8,7,incoming=1) predicts raw=13 and next_digit=3 instead of raw=16 and next_digit=6. The remaining claims are correct. Subtraction: the units state (3,9,incoming=0) predicts raw=9, digit=9 and next_digit=3 instead of raw=-6 and next_digit=4; the tens state (7,2,incoming=1) writes the correct 4 but predicts inconsistent raw=5 instead of 4. This is learned local prediction/consistency underperformance, not an observed checkpoint, state conversion, execution or scoring defect. Signs are correct for these two positive answers. Exact logits and sign logits were not archived; no further neural diagnostic run was performed. No answers, weights or conditions were corrected.

## Verification and boundaries

Independent replay verified all four requests twice (direct and unified: mathematics 4, coding 2, physics 2), without neural calls. It checked complete frozen-request coverage, request identity, source/asset hashes, seven-stage order, policy selection/action binding, serialized evaluation, positive measured parameter/call counts and absence of training-memory/weight updates. Coding replay used real WASI calls for visible and private suites; the ACT event precedes private scoring. Generated code never executes in host Python. Public requests reject unknown/private fields and unsafe code. Coding generation payload contains only visible tests and approved training memories. Physics public observations use the preserved masked observation interface; hidden simulator state is used only by execution/scoring.

Native loaders fail closed if MPS is unavailable or fallback enabled, verify parameter placement, and construct all neural inputs on MPS. Qwen's actual generation receipt confirms MPS, float16, batch 1 and a fresh model response. Math and physics use float32. Parameter totals: mathematics 109,843; coding backbone plus proposal evaluator 1,543,717,986; physics 17,477. Kernel calls per request: 9, 7, 4, 8. Direct-plus-unified wall times (including relevant loading/execution): 0.173s, 0.051s, 33.279s, 0.178s. Preserved physics proposal metadata reports a legacy fixed 24 calls; the integration trace correctly records eight batched forward steps. This discrepancy is disclosed, not used as measured compute.

Engineering PASS means outcome equivalence and executable domain boundaries, not task correctness. Math/physics macro routing is a declared deterministic submit-proposal scaffold, not learned semantic selection. The common kernel does not pool knowledge between domains. Prior physics scientific NO-GO remains unchanged; this one successful action does not establish world-model accuracy.

## Validation and reproduction

108 tests and 4 subtests passed in 11.85s across integration/kernel/coding/math suites. Final focused check: 15 tests and 4 domain subtests. Ruff and Zsh launcher syntax pass. Regression covers the real Kernel bridge contract and serialized metadata for every domain; unknown/zero parameter counts fail closed. The original missing `BridgeBackend.parameter_count` was fixed by measuring loaded model parameters; backend identifier and per-instance counters were also completed. No new integration defect was found in the successful run.

Native replayable execution (existing local assets required; no download/training):
```zsh
scripts/v3_integration/native.command --domain all
```
Independent non-neural replay:
```zsh
PYTHONPATH=src:work/physics-deps/site-packages python scripts/v3_integration/verify.py --run results/v3_integration/native-1791821373113687000 --output work/integration-replay.json
```
Tests:
```zsh
PYTHONPATH=src:work/physics-deps/site-packages python -m pytest tests_integration_v3 tests/v3 tests_coding_v3 tests_math_v3 -q
```
No retraining, physics study rerun, benchmark expansion, sealed holdout access or website modification was performed. External model/runtime assets remain pinned local dependencies. Future mathematical improvement requires a separate authorized study; this milestone preserves the negative math evidence.
