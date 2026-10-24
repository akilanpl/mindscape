# Portable CPU development and recovery

Use Python 3.11 or newer (the preservation check uses Python 3.12). The core
reasoning engine has no runtime dependencies. NumPy is needed for learning tests.
CUDA, Apple MPS, PyTorch, and downloaded language models are optional and are not
required for the minimal development path. This is a smoke/development path,
not a reproduction of multi-hour research experiments.

Clone the public repository and select the exact desired commit. The reviewed
default implementation at preservation start is
`5ad2de4313aba6a7468638ed5709876e0c406668`. Historical experiments remain independent
branches; local r6/r6.1 and delivery work must not be silently substituted for main.

```sh
git clone https://github.com/akilanpl/mindscape.git
cd mindscape
git switch preservation/portable-recovery-20261024
python3.12 scripts/bootstrap_portable.py
.venv-portable/bin/python scripts/verify_portable.py
.venv-portable/bin/python -m pytest -q tests/test_foundation.py tests/test_benchmark.py
.venv-portable/bin/python scripts/run_episode.py 19 3
```

On Windows use the environment's `Scripts/python.exe`. The bootstrap refuses to
overwrite any existing environment. It installs from PyPI without reusing pip
caches. `requirements-cpu-dev.lock.txt` pins the three top-level CPU development
packages; the preservation receipt also records the complete resolved environment.
The lock is not a cross-platform hash-locked research environment. Preserve the
receipt and create platform-specific locks for full neural research.
The complete minimal macOS resolver snapshot is also recorded in
`requirements-cpu-resolved-20261024.txt`; it contains no machine paths. A fresh
Ubuntu 24.04 runner passed the minimal CPU path and all five historical archive
checks at commit `c109c1fd7f65dbbd437a89575873ec932be45d06`:
https://github.com/akilanpl/mindscape/actions/runs/40098362319.

The same-device fresh model smoke used torch 2.8.0, transformers 4.57.6 and
safetensors 0.8.0 on CPU. The pinned SmolLM2 revision and downloaded file digests
are in `docs/pinned_model_download_20261024.json`. The three Qwen revision/LFS
manifests are in `docs/pinned_qwen_upstream_20261024.json`; these were checked
against upstream metadata, but full fresh Qwen downloads/inference were not run.

For the versioned historical evidence, first verify without extracting:

```sh
python scripts/verify_research_archives.py research_checkpoint/*
python scripts/coding/migration_checkpoint.py restore \
  --root research_checkpoint/research_release_2026-10-09 \
  --destination work/research_v1_restore
```

Restoration requires new, empty destinations. Do not overwrite current Git source
with an older restored source snapshot. Keep evaluation inputs and frozen evidence
immutable. The full checkout contains large Git-versioned archive parts; inspect
their sizes before downloading and allow additional free space for extraction.

For neural/model-backed research, follow the selected branch's bootstrap and
model manifest, pin the exact Hugging Face repository revision, and validate each
downloaded weight with its recorded SHA-256. Do not replace a pinned revision with
`main`. Download size depends on the model: historical Qwen Coder caches total
about 4.10 GB; the Qwen3-0.6B delivery weight is about 1.503 GB; the SmolLM2 cache
is about 0.273 GB. These are observed local sizes, not guaranteed transfer sizes.
Offline operation requires previously verified model files and package wheels.
Minimal core reasoning needs neither. No claim of full model recovery is made
until a separate remote-download and inference receipt passes.

## Continuing the preserved r6/r6.1 research state

The independent working-state branch
`preservation/working-20261024/mindscape-r61-82c6cf37f9` records commit
`82c6cf37f9a5bac5c1e3604910a12edbc9e0fa3d`. A fresh remote clone of that branch
passed 124 tests under `research/r61/tests` and `research/r6/tests` on macOS CPU
with Python 3.12, NumPy 2.2.6, pytest 8.4.2 and torch 2.8.0. Install that clone
editable into a new environment before testing it. This validates its code path,
not every external campaign, private artifact or training continuation.

```sh
git clone --branch preservation/working-20261024/mindscape-r61-82c6cf37f9 \
  --single-branch https://github.com/akilanpl/mindscape.git mindscape-r61
cd mindscape-r61
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.2.6 pytest==8.4.2 torch==2.8.0
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q research/r61/tests research/r6/tests
```

On Linux, select the official CPU-only PyTorch distribution if no accelerator is
needed. Preserve campaign checkpoints and map supplementary archive origins to
the branch's expected relative artifact paths before resuming research. No
historical architecture has been merged automatically into the default branch.
