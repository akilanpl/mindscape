# R5.4 Decision A review — PENDING_REVIEW

This is an unissued review package. Gate adoption is false; training authorization is false; scientific status is UNRUN. Recommendation: approve the frozen gate design for adoption after human review of the explicit AST boundary and schema mapping; withhold Decision B until native preflight succeeds and an independent validator accepts it. This recommendation issues neither decision.

Reviewed source: PR #20 base `950a17519f43fbc887e41f29ce2e24d4b6036ead`, stacked on PR #18 `988657be83fc4582218a840150e5230f335ca930`. The review is additive; no frozen scientific, engineering, integration or original evidence file is changed. The reviewer participated in prior preparation; root review and Agent B runtime observation supply separate checks, not a claim of fully independent authorship.

## Object bindings

- integration_lock_sha256: `35be4554909e822458f818be5dec845c10c2cf2062582151d9dc73e244fe03c7`
- v1_lock_sha256: `a771f8ae619fa080b09f49c52ebbad6f8eb2dbb37f6b4b6748fbd86700f6a30a`
- v2_lock_sha256: `ca6f9bf67ecf9fea9d3b0c8075cf3c0c0c4001f1a9049fef1cff28e5f7c585e5`
- Qualification final receipt: `21ebe987767e541713d864156752751ae58d8cc7985c7f86bf649a1663bdd6a1`
- Protocol: `8d94b1b6f495c5e4c2f4c57f31f8504ea2835d530a58779230cbc2b098090891`
- campaign_source_sha256: `4131182eb5ed9595feed03a28753feb69e39a7ef8596d74023d8bb992e2aa21e`
- scientific_suffix_ast_sha256: `0dff2e7d9244703599ed8e5554b7f170639a908ce00488c11dfd02613150ca13`

The read-only frozen verifier validated all 19 scientific and 18 engineering inventory entries, every integration inventory entry, the pinned original ENGINEERING_NO_GO, both receipts and all 29 original qualification artifacts. The original saved tensor auditor independently recomputed VERIFIED, 144 disposable steps, zero scientific updates, and six configurations. These assertions concern original saved evidence, not a new qualification run. Any binding mismatch blocks use pending reviewed version approval.

## Execution boundary and benefits

Original `campaign.py` has six preamble statements: authorization conditional, verify_lock(), DATA_READY conditional, locked training-task load, v1 check(tasks), and readiness conditional. The new launcher replaces only these with two separate approval validations, the same data/source checks, and pinned v2 eligibility. The parsed suffix begins at line 31 and ends at line 96; its exact normalized AST digest is bound above. Compilation preserves each suffix statement. No original run invocation, monkeypatch, fallback to the v1 gate, rewritten optimizer, fit, inference or scoring loop occurs. This preserves original workload and prediction-before-scoring order while making the engineering evidence provenance explicit.

The unchanged historical scientific verifier requires ENGINEERING_READY. The adapter emits that generic label only after validating both future approvals and exact v2 readiness. Its envelope explicitly records R54_V2_ENGINEERING_READY_TRAINING_NOT_AUTHORIZED, original_v1_status=ENGINEERING_NO_GO, and v1_engineering_gate_executed=false. Approving Decision A must explicitly approve this compatibility mapping. The generic label cannot mean original v1 qualification passed.

The storage-only supplemental checkpoint audit covers fractional/nonfinite/wrong-dtype Adam counters, missing/duplicated parameter routing, wrong moments, nonfinite or negative second moments, changed optimizer defaults, checkpoint substitution and failed/incomplete receipts. It constructs no model/optimizer and proves stored-state consistency, not actual training execution.

## Risks and limits

The original qualification recorded source-snapshot compatibility with e47239c0f84ca8147091faa2eacd8180b94b7079, but omitted original run HEAD. Do not retrospectively substitute a later checkout HEAD. Local approval records and original native/device assertions are not cryptographic identity attestations. AST equivalence proves source structure; live numerical execution equivalence and an actual 800-update duration remain untested. The frozen in-process timeout cannot interrupt a hung backend call. Original projections and memory observations meet frozen limits without establishing future runtime success.

Observed current runtime is Darwin arm64, torch 2.14.1, macOS-27.0.1-arm64-arm-64bit; MPS availability is false. Agent B's separately observed native-runtime.json agrees. Native execution readiness is BLOCKED; CPU results cannot supply a native PASS. No protected fit or neural OOD prediction occurred.

## Validation and reproducibility

77 permitted nontraining tests passed in 0.74 seconds (24 storage/readiness, 33 integration, 20 engineering/diagnostic). Full logs remain outside the repository in work/sprint-evidence/a. Commands below ran from the exact reviewed checkout; PY is the existing final-venv Python path documented in the frozen REVIEW.md.

```sh
$PY -m pytest -q tests/alpha/test_r54_checkpoint_guard.py integration/r5_4_gate_v2/test_safeguards.py engineering/r5_4_v2/test_amendment.py diagnostics/r5_4_mps_v1/test_diagnostic.py
$PY research/alpha/r54_prepare.py --qualification /Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-r5.4/results/r5_4/engineering-v2-native
```

Exact evidence SHA256:

- `readiness.json`: `558a730f8a359de2b9e9b17e8446f95fad083ea137b59234c2520971a1ff23a7`
- `readiness.stderr`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `tests77.log`: `1dc5ad4ad317c990d396cb42861a90c18d46fbbcc08d8e213a0a4bb24c0f2ab6`

## Unissued human records and future Decision B scope

R54_APPROVAL_TEMPLATE_UNISSUED.json is an exact schema-compatible copy of the frozen adoption template: approved=false, identity, decision ID, issuance/expiry and source bindings null. PENDING_REVIEW is the memo status; adding a status field to the authorization JSON would violate its exact-key schema. The template intentionally fails live authorization. Only an explicit future human decision may supply approved=true and attribution. Bind any future approval to the actual reviewed execution commit and all object hashes above, with the frozen valid UTC window of at most 24 hours; never invent identity, signature or dates.

Decision B, drafted only: separately authorize Markov/recurrent × seeds 11/23/47, exactly six fits × 800 updates = 4,800 scientific updates, batch 32, original AdamW/objective/schedule, fixed 1,024 training, 64 development, 192 OOD tasks and 200 context probes, all existing model/seed accuracy, confidence, corrected paired-test and resource gates, final checkpoint only, new exclusive output, no resume, retries, extra seeds or tuning. It requires actual Decision A adoption, exact native preflight success at matching reviewed source/host, independent validator acceptance, and a distinct explicit scientific authorization record bound to the destination. No Decision B record is issued here.

After an authorized completed campaign, require supplemental checkpoint storage audit plus the frozen future gate-provenance/scientific auditor with neural replay disabled. Current source/evidence review cannot replace these future checks. No scientific PASS, adoption, merge, deployment or Alpha release is implied.
