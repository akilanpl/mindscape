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
