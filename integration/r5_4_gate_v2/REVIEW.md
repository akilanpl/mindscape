# R5.4 v2 gate integration — proposed, not adopted or authorized to train

Status target: **R54_V2_GATE_INTEGRATION_REVIEW_READY**. This is a design/implementation review state, not gate adoption, optimizer authorization or a learned-model PASS. Original v1 scientific source and lock, v1 ENGINEERING_NO_GO, v2 preparation, native diagnostic and completed v2 qualification remain distinct and unchanged. No qualification rerun, fit trial, training smoke test or neural OOD inference is part of this review.

## Verified original v2 evidence

Located the existing directory at `/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-r5.4/results/r5_4/engineering-v2-native`. The frozen independent v2 audit was executed read-only, inspecting saved initial/restored float32 tensors, clipped gradients and final zero optimizer state. It returns exactly `R54_V2_ENGINEERING_READY_TRAINING_NOT_AUTHORIZED`, independent audit `VERIFIED`, 144 disposable steps, 0 scientific updates, and all 6 architecture/seed configurations passing. Original 29 files are preserved byte-identically in an authenticated archive; the source directory is never edited.

| Configuration, original qualification order | Projection seconds | Peak RSS bytes |
| --- | ---: | ---: |
| Markov 11 | 208.49109078148857 | 457146368 |
| Recurrent 11 | 207.7556849174822 | 458194944 |
| Recurrent 23 | 207.49007321547313 | 458686464 |
| Markov 23 | 214.8968172075044 | 456966144 |
| Markov 47 | 205.29390021152238 | 456720384 |
| Recurrent 47 | 207.80473801500443 | 458014720 |

All projections are below 720 seconds; peak process RSS 458,686,464 bytes (437.4375 MiB) is below 4 GiB. Every observed worker wall duration is below 6 seconds, and the 90-second qualification timeout is below the frozen 900-second scientific runtime limit. No 800-update fit has run, so its actual runtime is untested; the original per-fit 900-second guard remains unchanged, including its existing limitation that a hung backend call can prevent the next in-process timeout check. V2's declared 720/900/4 GiB budgets are verified against frozen plan/source; no threshold is relaxed.

Exact original qualification receipt SHA256: `2dcd247d5827d3fcb61264d4d614192c0857ec09a82cea2cce1884e64ff3c0a8`.

Exact final receipt SHA256: `21ebe987767e541713d864156752751ae58d8cc7985c7f86bf649a1663bdd6a1`.

V1 lock: `a771f8ae619fa080b09f49c52ebbad6f8eb2dbb37f6b4b6748fbd86700f6a30a`.

V2 engineering lock: `ca6f9bf67ecf9fea9d3b0c8075cf3c0c0c4001f1a9049fef1cff28e5f7c585e5`.

Qualification source snapshot commit: `e47239c0f84ca8147091faa2eacd8180b94b7079`; every 18 engineering/reference/evidence hashes matches that commit. The original qualification START did not record its run HEAD. Current native-checkout HEAD was observed as `fab8339bc6c8116cd9d5a93f26e5c029b8526c4f`; that observation is not retroactive proof of run HEAD. The integration explicitly records the source snapshot compatibility and the missing run-commit provenance instead of inventing it. Native execution/human identity are not cryptographically attested; this audit establishes saved tensor/record/hash consistency.

## Original scientific path audit

| Location | Role |
| --- | --- |
| `__main__.py:13,33–37` | Legacy `--authorize-training` flag passes to original `campaign.run`; no v2 gate dependency injection exists. |
| `campaign.py:22–23` | Original direct human-authorized Boolean gate. |
| `campaign.py:24–27` | Original source/data lock, DATA_READY qualification and training-task load. |
| `campaign.py:28–30` | Original first-call v1 engineering check; this entry point remains unchanged. |
| `campaign.py:31–38` | Exclusive output creation, source commit/lock/engineering provenance, operands loaded. |
| `campaign.py:41–46` | Models and seed order, initialization, frozen fit invocation. |
| `training.py:40,43–59` | AdamW construction, exactly800 updates,900-second guard, schedule/objective/clipping and optimizer steps. |
| `campaign.py:49–59` | Checkpoint/optimizer/tensor metadata and hash storage, then free predictions. |
| `campaign.py:79–82` | Predictions written/fsynced and sealed before outcome scoring. |
| `campaign.py:84–96` | Separate teacher-forced/probe diagnostics, original audit, metrics, completion/failure receipts. |
| `audit.py:169–172` | Existing scientific audit requires generic `ENGINEERING_READY`; it does not understand v2 provenance. |

The entire 19-entry original scientific SHA256 inventory is verified, including models, objective/fitting, inference, scoring/audit, tests, dataset/outcomes and protocol. Original lock bytes and original NO_GO file are pinned separately. No existing scientific source is edited. No older research, specialist, physics, website, tags, main or deployment configuration is changed.

## Proposed execution boundary, explicitly requiring adoption review

The legacy `campaign.run` cannot accept a new engineering certificate; calling it would still perform the v1 check. Monkeypatching `check` or passing a made-up v1 result is excluded. A copied/reimplemented scientific loop is also excluded.

The new versioned launcher is outside the original inventory. It separately implements only the six original preamble statements: human authorization, source/data readiness, training-task load and engineering eligibility. Human authorization is strengthened into two distinct future decisions. It then reuses the **exact parsed scientific AST suffix** of the unchanged `campaign.py`, beginning at line 31 through 96, in a newly named three-argument function. The original file, original `run`, its globals and its gate are never mutated or invoked by this path. A private namespace copy retains the same frozen model/fit/inference/control/audit functions and package-relative imports.

Campaign file SHA256: `4131182eb5ed9595feed03a28753feb69e39a7ef8596d74023d8bb992e2aa21e`.

Scientific suffix AST SHA256: `0dff2e7d9244703599ed8e5554b7f170639a908ce00488c11dfd02613150ca13`.

The boundary is unique and fail-closed: any source or AST change invalidates the launcher. There is no patching of globals, `check`, `run`, bytecode or optimizer; no fallback into the old `python -m mindscape.r5_4 train` command. Tests inspect/compile the reused AST but never execute it. This is an explicit **versioned scientific execution-boundary proposal** requiring human adoption approval. AST/source equivalence is demonstrated; live numerical execution equivalence is untested because no fit or OOD forward is authorized.

Same scientific operations: Markov/recurrent architectures, seeds 11/23/47, exact 800-update fits, same AdamW/objective/exposure schedule/data permutations, final 800 checkpoint, dataset identities and ordering, inferred carry propagation, model/control conditions, checkpoint metadata/hashes, all free predictions sealed before outcomes, diagnostics/scoring/gates and completion/failure handling. The original source calls perform these operations directly. Input `tasks` comes from the same locked JSON after the same locked DATA_READY function. Every fit resets its seed; replacing the engineering preamble does not supply or alter a learned state.

## Fail-closed engineering certificate and provenance

1. Pin all 29 original qualification file hashes and both receipts in a new integration freeze. Reject missing directories; do not rerun qualification.
2. Reject symlinked root/parent/file paths and traversal. Open directory/file descriptors with `O_NOFOLLOW`; require regular files and exact reviewed inventory. Copy only exact pinned bytes into a private temporary snapshot. Tampered data or recomputed/substituted receipts cannot substitute another qualification.
3. Require complete base and final receipts and the stored independent audit. Status must exactly match approved v2 engineering readiness, `VERIFIED`, six complete configurations, 144 disposable steps and 0 scientific updates; engineering approval must remain explicitly not training approval.
4. Re-audit the snapshot using the frozen v2 tensor/metric auditor in a fresh storage-only process, avoiding cached/substituted Python audit modules. Compare the saved and recomputed verdict. Verify original source/evidence bytes again afterward.
5. Verify the full v1/v2 and integration source inventories, clean committed files and compatible campaign AST. Stale means a different reviewed receipt, source/data/plan digest, approved source commit, expired authorization or runtime; no unrecorded qualification timestamp/expiry is invented.
6. At any future authorized execution, check native MPS/fallback and match recorded torch version, arm64 machine and platform. The evidence did not store chip serial/model identity or Python version; do not claim stricter hardware attestation than the records support. An incompatible runtime needs another reviewed evidence/version decision, not an automatic rerun.

The new start record embeds qualification base/final/start/audit digests, v1/v2/integration lock digests, full compatible scientific inventory, approved current source commit, source-snapshot/missing-run-HEAD qualification provenance, engineering-only authorization provenance and both future human approval records/digests. Generic `ENGINEERING_READY` appears only as a **declared schema compatibility label** required by the unchanged scientific verifier. The same record explicitly states v2's exact native status, `original_v1_status=ENGINEERING_NO_GO`, and `v1_engineering_gate_executed=false`. This is not a v1 PASS or overwrite. Review must approve that mapping explicitly.

A proposed post-campaign gate auditor verifies this richer provenance and historical source-commit locks/authorization-at-start, then calls the unchanged scientific metric auditor with neural replay disabled. It adds a gate-specific audit layer; the original scientific audit cannot by itself authenticate the new gate fields. This tool is unrun on scientific artifacts because no scientific campaign exists.

## Future human authorization requirements

Gate adoption and scientific training require different approval files, distinct decision IDs, explicit statements, approved-by attribution, current reviewed execution commit, every lock and exact final qualification receipt. Engineering START approval cannot satisfy either. The training record binds exactly two models, three fixed seeds, six fits, 800 updates each/4800 total and a new output directory. Local UTC authorization windows must be explicitly future-issued/valid at start and at most 24 hours; these are integration approval safeguards, not changes to scientific accuracy/resource gates.

Templates are unapproved and contain null future fields. No live authorization file is generated. These are human-approved local attestations, not signed identity credentials or an OS security boundary. Future authorizations must come from an explicit human instruction; the agent must not fill `approved=true` from engineering readiness or this implementation request.

Default launcher invocation does only read-only review (or returns BLOCKED if evidence is absent), with 0 scientific optimizer updates and 0 neural OOD predictions. `--execute-scientific-training` alone cannot execute. Both adoption and scientific approval are validated before importing the scientific campaign or crossing its body boundary. The body boundary revalidates approval/source bindings defensively. Destination creation remains exclusive inside the original scientific body; no resume, retries, extra seeds or changed workload.

## Tests, preservation and unrun stages

Synthetic fixtures/dry-run checks cover missing, failed, partial, malformed, symlinked, tampered, stale and substituted evidence; repaired receipt hashes hiding invalid statuses/counts; source tampering; engineering-vs-scientific scope confusion; stale/expired approval commits/receipts/destinations; separate decisions; default nontraining invocation; no old-command fallthrough/monkeypatch/optimizer calls; exact AST suffix and checkpoint/prediction/scoring order; and incomplete scientific audit rejection. Every test is nontraining; no scientific optimizer or neural predictor is constructed. Source compilation and inventory checks complement the test suite. Frozen inventory comparison and raw evidence preservation results are separate artifacts.

Unrun: gate adoption, live training authorization, native execution of the reused body, numerical campaign equivalence, every scientific fit, neural OOD predictions and future post-campaign gate audit. No model-learning PASS is reported. Original v1 NO_GO remains historical evidence; v2 engineering readiness is distinct.

## Proposed future native command

**NOT AUTHORIZED — DO NOT EXECUTE.** After separate human approval of this execution boundary/schema mapping and separate approval of the six-fit scientific campaign, update the native checkout to the approved commit, create the two explicitly authorized approval records, then use the proposed versioned command:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python integration/r5_4_gate_v2/launcher.py --execute-scientific-training --qualification results/r5_4/engineering-v2-native --gate-adoption authorizations/r54-v2-gate-adoption.json --scientific-authorization authorizations/r54-scientific-training.json --output results/r5_4/scientific-v2-campaign
```

No adoption is made by documenting this command, presenting review material, merging unrelated work or running a dry review. No draft PR merge, main update, deployment or Alpha release is authorized.

**Decision requested:** approve or reject this separately frozen gate-integration execution-boundary design, including exact-AST reuse and declared legacy readiness-schema mapping. Scientific training requires a later, separate explicit human authorization. Current scope ends at review readiness.
