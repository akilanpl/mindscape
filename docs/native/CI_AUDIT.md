# Native correction: preservation and historical CI audit

This audit is additive. It does not modify historical workflows, dependency locks,
research sources, archived evidence, tags, preservation scripts or research gates.
It does not report legacy red CI as green. No cloud job or training campaign was
started by this audit.

## Exact references and README evidence

| Reference | Full commit | README Git blob |
|---|---|---|
| Frozen pre-V3 declaration | `33c5b5657b2d19741b7b0ac84388d4f7997f9582` | `bac8aa0ed64a86c476d5953d8c94c88750d6214c` |
| Pre-product PR27 base | `8c141d8f1b237251d8fa9e5b4ccd87efbf3b3c96` | `7d47768a22505341e7768da5dd1b3c9bd5558eca` |
| PR27 product head | `3cefed2a2cc4e6a5867602822fcd35bc8d19bc96` | `7d47768a22505341e7768da5dd1b3c9bd5558eca` |

The README differs from the historical frozen declaration **already at the
pre-product base**. Base and PR27 head are exactly equal. The observed foundation
job failed at the unmodified `scripts/v3/preservation.py` with
`ValueError: Historical file changed: README.md`. Therefore that specific mismatch
was not introduced by PR27 or the native correction. This attribution does not
waive the original historical check or retrospectively repair its baseline.

## Observed research-v2 collection failure

Downloaded CI evidence for artifact `research-v2-cpu-b5b6d31b1d80f1f8d40b9d698c3418afd2b6a20b`
records integrity and statistics passing, followed by pytest collection exit 2:

1. `tests/r5_3_physics/test_public.py` imports the physics environment, which imports
   `pymunk`; the environment reports `ModuleNotFoundError: No module named 'pymunk'`.
   The historical research-v2 CPU dependency lock does not list pymunk.
2. Both `tests/r5_2_recovery/test_recovery.py` and
   `tests/r5_3_recovery/test_recovery.py` are collected with the same bare module
   name. Pytest reports an import-file mismatch between those two files. The
   historical runner collects the entire `tests` directory; its pytest settings
   do not select importlib mode.

These are observed collection/dependency failures, **not evidence that the old
scientific tests pass or fail**. The historical workflow, dependency lock, runner,
pytest settings and cited test files have identical Git blobs at the pre-product
base and PR27 head. Their relevant source conditions predate the product branch.
No historical dependency or test setting was altered to hide these failures.

## Native preservation boundary

Run `python scripts/native/ci_preservation.py --output work/native-preservation.json`.
The baseline is exactly PR27 head above. Every existing tracked baseline blob is
protected, except existing product surfaces:

- `src/mindscape/mind1/`
- `mind1_web/`
- `docs/mind1/`
- `tests_mind1/`

Two exact existing product files are also mutable: `.env.mind1.example` and
`requirements-mind1.txt`. They were introduced for PR27 product configuration and
dependencies, and are not historical research locks. These exceptions permit
removal of obsolete external-provider tests/configuration and native Torch
dependencies without weakening protection of historical research files.

That protects 1,275 baseline files, including complete historical research/config/
dataset/archive/script/test/workflow trees, root README and package metadata, and
original `src/mindscape/r56/` research sources and `docs/r56/` report. Content and
executable/symlink modes are checked against Git blob IDs. Missing files fail.
The two historical tag references declared in the preserved tag manifest must
also match exactly; missing or changed declared tags fail.
New additive native modules, tests, docs, configuration and a dedicated workflow
are outside the old tracked-blob inventory; this does not authorize edits to old
files. Existing product-prefix changes remain reviewable through the Git diff;
this audit does not claim those mutable surfaces are unchanged.

`tests_native/test_preservation.py` checks the exact README evidence, the complete
protected inventory, old CI source equality across base/head, and a temporary Git
fixture that proves frozen-source deletion and lock mutation are detected while
allowed product edits (including product-test deletion, product dependency and
environment changes) remain allowed. An independent AST scanner additionally
rejects cloud SDK/provider imports, old provider calls and known external inference
endpoint literals across production Mind1 modules, and requires the provider
tombstone to declare external inference disabled. This is static evidence, not a
claim to prove arbitrary dynamic/transitive code cannot use a network. The native
runtime audit root can be selected with `MIND1_AUDIT_ROOT` for pre-merge review. These are read-only audits against the real
checkout; only the test fixture creates temporary commits.

A dedicated native workflow must run a narrowly selected, nontraining contract
suite and this additive preservation audit. Running the repository's older push
workflows can invoke the historical V3 coding training entry point. No training
is authorized by adding a native workflow, and no green-check claim is justified
until that particular workflow actually runs successfully. Historical red jobs
remain reported separately. This audit does not weaken or replace any old check.
