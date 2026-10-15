# GitHub integration

Repository: https://github.com/akilanpl/mindscape

Feature branch: https://github.com/akilanpl/mindscape/tree/feature/detachable-hybrid-assistant

Pull request: https://github.com/akilanpl/mindscape/pull/31 — targets existing `main`; remains unmerged.

Known-good B branch: https://github.com/akilanpl/mindscape/tree/baseline/hybrid-candidate-b-20261015

Immutable B commit: `78c5771490491a2d1dc58e7baebdaf2ee5efd22a`.

Original default HEAD: `f9955c7b8043a00ebea36fffd41c646d1d54f2bd`. Core files, root dependencies, original CI, research blobs and checkpoint formats are unchanged. An independent sparse clone preserved all unrelated local checkouts. The baseline was tested, committed, pushed, fetched back and checked against remote before auditing/improving.

## Separately reviewable commits

```text
78c5771490491a2d1dc58e7baebdaf2ee5efd22a feat(hybrid): add optional isolated Candidate B assistant
a973acfad24cb70575346f4dafdee6cfe8da6fc5 feat(hybrid): add bounded verified math physics and inline CSV tools
86b0def0613477143bfe18a009f8e90b72eea412 fix(hybrid): ground electrostatic force and literal SQL table routing
4802bc5d9c52b8560309be11d7cf6f94bd36de42 fix(hybrid): preserve integral domains and reject unknown coding tasks
57286adf1b2fc794f1b20b649da7bd1536954dc4 test(hybrid): cover domain unit sandbox and routing regressions
5d0ab46e72ff726aa1be6218f3276d0d7f3dc5b8 test(hybrid): add reproducible 36 question diagnostic and independent grader
a23954a5db5b9d9690af5c030edf622334ba4520 fix(hybrid): parenthesize rational probabilities in worked Bayes equality
e00f832b92c595932be28737a1960c5541285f7d test(hybrid): package calibrated pacing and exact numeric benchmark grader
ccc9874e9738c030bd6f4c79a7dd14a8484bf35d docs(hybrid): publish isolation matched results and offline rollback guides
```

The B tag initially added for convenience conflicted with V3's exact archived release-tag set. Only the newly created tag was removed; original tags were restored exactly, and B uses its immutable SHA plus a dedicated baseline branch. Preservation code was not changed. The original baseline foundation CI failures are retained as evidence; corrected-tag runs/final-head checks are recorded separately. CPU validation and mathematics jobs passed on the baseline. Do not describe a pending final check as passing.

## Publication boundaries

All changes are confined to `integrations/hybrid/`. No offline ZIP, pretrained weights, virtual environment, private dataset/checkpoint, Kaggle credential, API token, absolute Mac path, raw protected evaluation content or Git LFS upload is included. Model revision, dependency locks and nine-file SHA256 manifest allow explicit downloads. GitHub receives lightweight source, public diagnostic questions and aggregate reports; complete protected evidence stays local.

See HYBRID_ISOLATION_REPORT.md for exact changed files/removal proof and HYBRID_HARD_CBSE_BENCHMARK.md for matched results and failures. Final branch push and PR update are verified after the documentation commit; the local PUSH_VERIFICATION.json records remote SHA, PR head/base/merge state and final CI snapshot.
