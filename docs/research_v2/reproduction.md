# Research v2 reproduction baseline

Canonical public repository: https://github.com/akilanpl/mindscape. Default branch `mindscape-cloud-migration-2026-10-08`; frozen Researchv1 tag `mindscape-research-complete-v1` points to `2862f5ae54443b231abda4255bdd6c652eca2343`. Earlier releases and failed studies remain preserved.

Clone, then restore the checksum-bound release into an empty destination:

```sh
git clone https://github.com/akilanpl/mindscape.git
cd mindscape
python scripts/coding/migration_checkpoint.py restore --root research_checkpoint/research_release_2026-10-09 --destination work/research_v1_restore
```

Source/tests/protocols/reports and archiveindexes are tracked. Large evidence/adapters live in immutable Git-versioned32MiB parts, with SHA256 in `ARCHIVE.json` and `SNAPSHOT_MANIFEST.json`. Foundation weights are pinned external HuggingFace revisions in the v1 manifest and bootstrap, not embedded credentials or copied virtual environments.

Stage1 clean Git clone and complete archive restoration passed 130tests plus6subtests in the existing pinned Mac environment. This does not establish a clean Linux dependency installation: that is Stage2. The old unittest-only CI failed on missing numpy/pytest and is preserved in Git history. See `experiments/research_v2/stage1_canonical_repository.json` for exact scope.
