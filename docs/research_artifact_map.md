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
| Supplementary scientific preservation | GitHub release preservation-evidence-20261024 | 30,887 deduplicated objects; see preservation_evidence_20261024.json |

Run `python scripts/verify_research_archives.py PATH...` for collections using
ARCHIVE.json/SNAPSHOT_MANIFEST.json. It validates the part hashes, manifest hash,
every streamed archive member hash, and inventory completeness without extracting
or modifying the archives. Historical restore commands remain branch-specific.
Do not treat successful historical restoration as proof that newer local evidence
is backed up. Preserve original failed and negative experiments as well as successes.

## Supplementary release recovery

Download into a new directory:

```sh
mkdir supplementary-recovery
cd supplementary-recovery
curl -fL -o scientific-evidence-20261024.tar.gz \
  https://github.com/akilanpl/mindscape/releases/download/preservation-evidence-20261024/scientific-evidence-20261024.tar.gz
cd ..
python scripts/restore_preserved_evidence.py \
  supplementary-recovery/scientific-evidence-20261024.tar.gz \
  --sha256 26c1ecb0a14dffebc247fdc74e8830d28a33f4a8ece7618a3d53b90cf839d8dc \
  --destination supplementary-restored
```

The destination must not exist. Omitting `--destination` verifies without extracting.
Files are restored under `origins/` with workspace-relative historical origins.
Select the intended research branch and map its required artifact paths explicitly;
the archive is a preservation collection, not an automatic integration of r6/r6.1
and older architectures. Private/machine-specific files remain excluded and blocked
from deletion until separately preserved. See the local preservation report for the
actual upload/download verification result; the existence of these instructions
alone is not proof of successful off-device preservation.

The same release includes `workspace-source-snapshots-20261024.tar.gz` (459
additional historical source objects). Its SHA-256 is
`5dc5b6c9ddcffefa856f63dd4951e4b264820f4d87fb0aed97c377ee190bec53`.
Use the same verifier with a different new destination. Both assets were uploaded,
independently downloaded from the published release, and restored with every
object hash checked during preservation. See `workspace_source_evidence_20261024.json`.
