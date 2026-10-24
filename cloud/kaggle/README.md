# Mindscape private Kaggle execution

Migration infrastructure for unchanged R5.7c source at d54ff8fe64f30c53b9662ddba90691338aad05f6. The Mac prepares, verifies, submits and downloads. Fitting requires actual Kaggle Linux CUDA. Use the delivered CLOUD_EXECUTION_COMMANDS.md and actual manifests for execution status; source alone is not execution evidence.

- `scripts/workflow.py`: account/quota verification, private offline packaging, one-shot submission, status and download.
- `scripts/cuda_probe.py`: five actual CUDA updates, hardware, reload and hashes.
- `training/worker.py`: exactly 2,000 Adam calls per seed/condition, losses, checkpoints and failure receipts.
- `scripts/campaign.py`: guarded sequential nine-fit campaign; no automatic optimizer retry.
- `reproducibility.py`: immutable training package and export verification.
- `scripts/collect_evidence.py`: fresh authenticated trainer source pulls and exact package/source/data/checkpoint binding.
- `scripts/prepare_evaluation.py` and `scripts/evaluate_campaign.py`: two-phase separate CPU evaluation with frozen runtime isolation, actual provider source checks, read-only checkpoint checks and verified downloads.
- `scripts/prepare_equivalence.py`: zero-update CPU/CUDA inference comparison with pinned tolerances.
- `scripts/comparison_report.py`: raw-prediction verification, historical distribution matching, descriptive scores and regressions.

The delivered campaign contains nine completed verified CUDA fits (18,000 native updates), plus a separate five-update successful probe. Two earlier old-account probes stopped before any optimizer call. Preserve every reservation and failed receipt. No expansion of the native budget, retraining to repair scores, public publication or paid service is authorized.

The evaluator protects finalized runtime inputs under a trusted account-owner controller; it does not claim separately owned accounts or an adversarial-owner boundary. Historically reused final cases support a matched comparison, not a new untouched benchmark. See `evaluation/README.md` and the actual downloaded audit.

Run development checks without fitting using Python >=3.11, PyTorch and pytest:

```sh
PYTHONPATH=.:src python -m pytest -q cloud/kaggle/tests tests/test_training_portability.py tests/test_infrastructure.py tests/test_cloud_comparison_report.py tests/test_prepare_equivalence.py
```

The offline cloud worker uses preinstalled dependencies. Hardware/software versions come from cloud receipts; no local CPU substitution is accepted.
