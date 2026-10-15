# Preserved Candidate B baseline

Remote baseline commit: `78c5771490491a2d1dc58e7baebdaf2ee5efd22a`; immutable commit on branch `baseline/hybrid-candidate-b-20261015`; PR https://github.com/akilanpl/mindscape/pull/31 . Baseline was pushed and fetched back before any solver improvement.

Original delivered B source hashes match its historical freeze. Its historical acceptance remains **31/36 hybrid, 16/36 pretrained-only, 4/4 safe refusals**, warm p95 3.456s; 13 accepted topics. Coulomb and natural-language SQL were excluded from accepted topic coverage. These numbers come from the preserved historical campaign, not the new hard benchmark; subtracting them from new C results would not be a matched comparison.

Fresh baseline checks: 147 unchanged hybrid tests plus two boundary checks passed; original native unittest discovery ran 92 tests (one skipped); pytest 146 passed / 22 skipped, 6 subtests passed. The first broad pytest attempt failed on missing optional Wasmtime and sparse-checkout v3 fixtures; these original dependencies/fixtures were restored and the suite passed without source changes. Skipped tests retain their original optional-runtime conditions; this does not certify unexecuted GPU/cloud paths.

Actual CPU model-backed HTTP smoke verified differentiation with pinned Qwen3-0.6B advisory inference and tool provenance. Native no-site-packages smoke independently verified 19*3=57 without torch, transformers, SymPy or hybrid modules initialized. No optimizer updates occurred.

Portable baseline changes remove machine-specific launcher fallbacks and replace copied private R6 assets with optional external mounts. Original B implementation identities are recorded under `candidate-b-source-identity.json`. No model weights, offline bundle, checkpoints or evaluation secrets were pushed.
