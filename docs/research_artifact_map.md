# Scientific artifact map

| Collection | Location | Verification / limits |
| --- | --- | --- |
| Historical migration snapshot | research_checkpoint/2026-10-08 | ARCHIVE.json, SNAPSHOT_MANIFEST.json, split parts |
| Completed historical release | research_checkpoint/research_release_2026-10-09 | Same part and per-file SHA-256 verification |
| Local memory recovery | research_checkpoint/local_memory_recovery_2026-10-09 | Inspect its own manifest; do not assume identical schema |
| Research v2 gap closure | research_checkpoint/research_v2_gap_closure_2026-10-09 | Verify its own complete inventory |
| Research v2 numeric | research_checkpoint/research_v2_numeric_2026-10-09 | Verify its own complete inventory |
| Later ignored results/work | Original local workspaces | BLOCKED until exact content is mapped to verified remote bytes |
| r6/r6.1 checkpoints and delivery evidence | Original research/delivery workspaces | Separate from historical archive coverage; preserve independently |
| Foundation models | Pinned upstream model repositories | Revision, license, filename and checksum required; no implicit latest revision |

Run `python scripts/verify_research_archives.py PATH...` for collections using
ARCHIVE.json/SNAPSHOT_MANIFEST.json. It validates the part hashes, manifest hash,
every streamed archive member hash, and inventory completeness without extracting
or modifying the archives. Historical restore commands remain branch-specific.
Do not treat successful historical restoration as proof that newer local evidence
is backed up. Preserve original failed and negative experiments as well as successes.
