# Native architecture correction — engineering review

This replaces PR27's external reasoning architecture on the separate branch
`development/mind1-native-r56`, from exact head
`3cefed2a2cc4e6a5867602822fcd35bc8d19bc96`. PR27 remains unmerged.

| Gate | Status |
|---|---|
| ARCHITECTURE IMPLEMENTED | Implemented bounded original graph transition model, trainable concept embeddings, relation processing, action conditioning, bounded memory, typed domain adapters and separate verifiers |
| NATIVE INFERENCE TESTED | Real CPU random-weight predictions travel through browser/API/native runtime/model/verifier; wrong predictions preserved |
| LEARNED MODEL QUALIFIED | NO — no scientific training campaign or qualified checkpoint |
| COMPOSITIONAL GENERALIZATION VERIFIED | NO — evaluation preparation and engineering tests only |
| CROSS-DOMAIN TRANSFER VERIFIED | NO — shared parameters and core-transfer preparation are not transfer evidence |
| PRODUCT RELEASE READY | NO — bounded engineering candidate; broad native language and scientific qualification remain unresolved |

R5.6 EXPERIMENTAL — SCIENTIFIC GENERALIZATION UNVERIFIED

No third-party inference account is required. Hosted/pretrained inference removed;
normal chat does not use an external provider or hidden fallback. Deterministic
parsing, supplied concepts/schedules, neural proposals, reference verification and
scientific tools have separate attribution. StageB language encoder is original
but untrained; synthetic paraphrase/intent/span learning infrastructure is prepared.

Native demo observed 27+18 -> original prediction166, verifier expected45 false.
This is intentionally recorded as failure of the random prediction and successful
engineering evidence separation, never as a learned success. Real WASI execution
and symbolic checks remain tools, not native neural intelligence.

The protected inventory audit checks 1,275 historical baseline files and two tags.
Old foundation README and research-v2 test-collection CI failures predate PR27;
exact hashes/logs are documented in docs/native/CI_AUDIT.md. Old checks and files
are not disabled or rewritten. New native CI contains only engineering checks;
local passing tests do not establish a successful hosted CI run.

Training infrastructure is implemented but optimizer loops are unexecuted here.
Frozen candidate budgets32/128/512 and seeds17/29/43 are retained. Eight neural
controls imply72 candidate fits/144000 updates, pending review and authorization.
Matched baseline parameter counts differ by less than0.5%; compute is measured,
not presumed matched. Distinct task counts and repeated exposures are reported.
Source/data/evaluation locks and external independent qualification plus human
scientific authorization are required before optimization. Unissued templates
are not authorization. Historical study approval never authorizes R5.6.

Unresolved: independent scientific qualification, approved training, real heldout
learning curves, learned language integration qualification, broad task coverage,
and hosted CI verification. No merge, tag, release or public deployment occurred.
