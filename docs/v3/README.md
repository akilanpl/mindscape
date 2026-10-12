# Mindscape V3

Start with [handoff](handoff.md), [architecture](architecture.md),
[smoke protocol](protocol.md) and [baselines/compute](baselines-and-compute.md).

Python 3.11+ is supported by the source; this run used Python 3.12.14 on macOS ARM64.
V3 itself requires only the standard library. From repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests/v3 -v
python scripts/v3/preservation.py --output work/v3-preservation.json
PYTHONPATH=src python -m mindscape.v3.experiment --manifest configs/v3/smoke_manifest.json --output work/v3-smoke
PYTHONPATH=src python -m mindscape.v3.experiment --manifest configs/v3/smoke_manifest.json --output work/v3-smoke
```

The manifest is already frozen; do not overwrite it. A new protocol needs a new
manifest path and a new run namespace. Resume requires an identical source and
runtime identity. Optional legacy tests need NumPy/Pytest and preserved WASI assets;
optional historical neural adapters need locally approved Transformers/PyTorch
checkpoints. Neither is needed for the V3 controls. CI adds an independent V3 job
without replacing the existing V2 workflow.
