# R5.4 scientific verdict: UNRUN

Engineering qualification is independently VERIFIED; mathematical learning is **UNRUN**. No scientific optimizer update or neural OOD prediction has executed. The 144 executed disposable engineering steps are distinct from the frozen 4,800-update scientific workload. Original v1 ENGINEERING_NO_GO remains unchanged.

## Source and evidence decision

The verified common development baseline is `development/r5.3-robust-physics` at `6c558caa3c3a1a09ec15e286a408cbb6d5076b03`. This preparation explicitly depends on unmerged PR #18, `research/r5.4-compositional-rule-learning` at `988657be83fc4582218a840150e5230f335ca930`. No gate was adopted by making this branch.

The frozen review launcher independently re-audited the original qualification directory on 2026-10-14. All 19 scientific inventory entries, 18 engineering inventory entries, integration inventory, original v1 NO_GO, qualification receipts and exact scientific AST remain intact. [Readiness evidence](r54-readiness.json) records exact locks, execution source, unchanged thresholds, runtime and workload. Qualification final receipt: `21ebe987767e541713d864156752751ae58d8cc7985c7f86bf649a1663bdd6a1`.

Six model/seed configurations qualify. Maximum projected fit: 214.8968172075044 seconds against 720; maximum process RSS: 458,686,464 bytes against 4 GiB. An actual 800-update fit and the 900-second runtime guard have not been exercised. Current sandbox PyTorch 2.14.1, arm64 and macOS platform match recorded versions, but MPS is unavailable here. Native runtime compatibility must pass on the authorized execution host; no CPU fallback is permitted.

## Concrete human decision package

1. Adopt or reject the existing separately frozen integration proposal in `integration/r5_4_gate_v2/REVIEW.md`, specifically its exact AST suffix reuse and explicit legacy `ENGINEERING_READY` schema mapping. That mapping records the native v2 status, preserved original v1 NO_GO and `v1_engineering_gate_executed=false`.
2. Separately authorize or reject scientific execution at the reviewed source commit, all three locks, exact qualification final receipt and one new output directory. Preserve six fits: Markov/recurrent × seeds 11/23/47, 800 updates per fit, batch 32, unchanged AdamW/objective schedule, final checkpoint only and no retries. Existing approval templates remain unapproved.

Neither decision has been issued. The documented future command in the frozen review is not executed. This preparation adds no training entry point and cannot bypass either decision.

## Additive nontraining safeguards and validation

The frozen checkpoint verifier uses `int(step.item()) == 800`, which can accept a fractional counter such as 800.5. It also does not verify all AdamW defaults or exact parameter-state routing. To preserve the scientific freeze, `research/alpha/r54_checkpoint_guard.py` supplies a separate storage-only audit. It rejects fractional/nonfinite/wrong-dtype step counters, incomplete or duplicated routing, wrong moment shape/dtype, nonfinite or negative second moments, altered epsilon/AMSGrad/maximize settings, checkpoint substitution and failed/incomplete campaign evidence. It constructs no model or optimizer and opens no outcome labels. This supplements the frozen auditor; it does not itself prove that a fit occurred.

Before any future scientific verdict, independently run the supplemental storage audit on the sealed completed campaign, then the frozen `integration/r5_4_gate_v2/audit_future_campaign.py` with the reviewed qualification evidence. The latter authenticates adoption/training provenance and independently recomputes the unchanged metric gates. Neural replay remains a separate authorized evaluation decision. No current campaign exists to audit.

Validation: 77 nontraining tests passed: 24 new synthetic checkpoint/storage/readiness tests, 33 frozen integration safeguards and 20 frozen engineering/diagnostic regressions. Synthetic tensors and campaign placeholders are explicitly fixtures, never scientific observations. Read-only source/evidence re-audit passed. Frozen files were not modified; no sealed outcomes were decoded for this preparation.

Reproduce the new tests with the existing runtime: `python -m pytest -q tests/alpha/test_r54_checkpoint_guard.py`. Reproduce readiness with `python research/alpha/r54_prepare.py --qualification <original-native-qualification-directory>`. The latter performs read-only verification and reports host compatibility; it exposes no training flag.

## Scientific claim boundary

All model/seed whole-answer accuracy, intermediate digit/carry correctness, predicted-carry feedback trajectories, held-context probes, memory gains and corrected paired tests remain unmeasured. Each frozen gate must pass for all three seeds of a model; partial-digit correctness is insufficient. A future R5.4 PASS would support only this frozen arithmetic study, not broader unseen-composition or cross-domain claims. Present status remains **UNRUN**, with separate human authorization and native runtime blockers.
