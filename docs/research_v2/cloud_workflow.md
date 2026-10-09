# CPU cloud validation

Stage1 canonical release verification precedes this workflow. `.github/workflows/tests.yml` runs on GitHub-hosted Ubuntu24.04 with Python3.11.17, pinned action commit hashes, CPU PyTorch2.8.0+cpu and direct dependency pins. Install reports record resolved artifact URLs/hashes; `linux-resolved.txt` captures every installed version. The complete real Linux resolution is now pinned in `configs/research_v2/linux-cpu-lock.txt`. Run37939153417 installed successfully but failed18tests on an incompatible bundled ARM execution cache; failed evidence is preserved. The immutable WASM checksum is verified before preserving and rebuilding that cache for the actual host. Frozen source/data/outcomes are unchanged.

No benchmark inference, GPU allocation or paid reference API is invoked. LoRA unit tests use a tiny synthetic random model. The full export is stream-verified (all23842 hashes); only the immutable6242-file package is written and clean-restored, avoiding duplicated historical snapshots on limited disks.

```sh
python scripts/research_v2/restore_cpu_baseline.py --archive research_checkpoint/research_release_2026-10-09 --destination work/cloud_research_v1
python scripts/research_v2/cpu_validate.py --evidence-root work/cloud_research_v1 --output results/research_v2/cloud_cpu_validation
# Repeat the identical command to validate resumability.
```

Each output namespace holds an exclusive lock. Code/config/rawdata/environment fingerprints reject changed inputs; completed steps are reused only after their output hashes match. Original experiment rows are never appended or rerun. Commands, logs, exit codes, host/backend/packages, resource use and hashes are uploaded as run artifacts (30day retention). Durable scientific receipts and the full Linux lock are versioned after independent retrieval; CI artifacts alone are not permanent publication storage.

Cloud success requires real Linux installation, complete export verification, all gates/tests/fairness/frontend checks, unchanged raw evidence and verified repeat invocation. Local success or active CI status is insufficient. Later inference/ablations/reference-model experiments need separately frozen protocols and outputs; historical CPU/MPS evidence remains unchanged.
