# Validation and review ledger

All validation was nontraining. The lead reviewer independently reproduced **123 checks**: R5.4 77, OOD 24, transfer 9, generic/independent audit 13. These are test counts, not scientific denominators. Historical campaigns and original qualification were not rerun. Zero scientific optimizer updates and zero neural OOD predictions executed. Synthetic CPU forward/backward architecture checks in the OOD suite use invented inputs and no optimizer.

| Branch / draft review | Final experimental commit | Lead validation |
|---|---|---|
| `research/alpha-r54`, PR #20 stacked on #18 | `950a17519f43fbc887e41f29ce2e24d4b6036ead` | 24 new checkpoint tests +33 frozen integration tests +20 original engineering/diagnostic tests =77 |
| `research/alpha-ood`, PR #19 | `d457428540b34a96e16c31fc4ffe3d13e8cba3fa` | 24 synthetic benchmark/architecture/replay/workflow tests; 14-file source inventory verifies |
| `research/alpha-transfer`, PR #21 | `03fc481192dca47410dd809de1303658c2f556cb` | 9 synthetic manifest/receipt/routing/controller test methods |
| `research/alpha-audit` | Root primitives `2ee1aa8a3b780e73509d0493a2e2039e2ecc12b9`; independent report version recorded by Git history | Root8 + independently authored D5 adversarial methods =13 |

Reproduction uses the already available final-venv runtime (no package/model downloads). From the respective worktree:

- A: `PYTHONPATH=src <python> -m pytest -q tests/alpha/test_r54_checkpoint_guard.py integration/r5_4_gate_v2/test_safeguards.py engineering/r5_4_v2/test_amendment.py diagnostics/r5_4_mps_v1/test_diagnostic.py`
- B: `PYTHONPATH=src <python> -m pytest tests/alpha_ood -q`
- C: `PYTHONPATH=src <python> -m unittest discover -s tests_alpha_transfer -v`
- D/root: `PYTHONPATH=src <python> -m unittest discover -s tests_alpha_audit -v`

The lead independently invoked the frozen R5.4 launcher in its default read-only mode against original qualification evidence; it returned `R54_V2_GATE_INTEGRATION_REVIEW_READY`, qualification independent audit VERIFIED, 144 saved disposable steps and zero scientific updates. The raw JSON is preserved in the exported handoff. No execution flag or approval artifact was supplied.

D's independent review has its own narrower reproduction scope described in `INDEPENDENT_VERIFICATION.md`; the aggregate123 is the lead's reproduced suite, not a claim that D executed every architecture test. D verified all1,192 original baseline blobs unchanged, reviewed experimental-source additions, challenged guard gaps, and independently checked fixes. The baseline path/hash inventory SHA256 is `4738d19587561ba0cdba9bc4ef2a659540d238b09fc6c94d9d6632ace149c5e0`.

All three scientific classifications remain UNRUN. Overall numerical execution readiness is UNVERIFIED. No scientific PASS, transfer benefit, neural generalization or Alpha release is established by these tests.

The exported `evidence-ledger.json` records exact final heads, review bases, PR URLs and SHA256 hashes of every additive committed file. This avoids embedding a commit's own hash in its contents. No branches have been merged, historical evidence rewritten, tags changed or frontend/public release modified.
