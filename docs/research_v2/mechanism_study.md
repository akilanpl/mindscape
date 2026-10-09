# Frozen numeric mechanism study

This study tests the existing 52-input, 64-hidden-unit NumPy trajectory policy at fixed trained weights. It does not evaluate the coding foundation model, establish general intelligence, or compare GPT/Gemini.

The protocol and novel evaluation data were committed before execution (`e059e8e`). Eighty task identities are disjoint from all preserved generated dataset identities and each checkpoint's training IDs: 40 unsigned 2×2 IID tasks and 40 signed 4×3 OOD tasks. This identity check cannot establish absence of every possible semantic overlap or outside pretraining exposure.

Three historical budget-50 checkpoints (seeds 0, 1, 2) each execute all eight conditions of legal-action masking, state-feature readout, and previous-action-feature readout. Removing a readout means zeroing its declared input slice, not removing the entire memory architecture. Operands, model weights, tasks, environment, independent scorer and 256-action budget stay fixed. Verification and goals are controls; planning, dreaming and recovery are absent from this backend.

Each of the 1,920 planned executions stores actual neural inputs, logits, derived uncalibrated probabilities, direct action-selection observations, memory changes, result and independent verification. Hidden activations are reconstructed and labelled; they are not causal explanations. Executed failures remain unsuccessful outcomes. Missing executions never become fabricated zeros. Instrumented runtime includes observation overhead and is not a dedicated latency measurement.

Three primary comparisons use the all-enabled condition against each single feature removal. Exact paired McNemar tests are calculated per seed and split with Holm correction across 18 predeclared tests. Task-cluster bootstrap confidence intervals concern task sampling conditional on these three trained seeds. Wilson intervals accompany per-cell rates. These intervals do not imply generalization to arbitrary domains or model seeds.

Cloud execution commands (after checksum-verified archive restoration):

```sh
PYTHONPATH=src python scripts/research_v2/mechanism_study.py --evidence-root work/cloud_research_v1 --output results/research_v2/cloud_cpu_validation/mechanism_study
python scripts/research_v2/analyze_mechanisms.py --study results/research_v2/cloud_cpu_validation/mechanism_study
```

Run identity binds source, protocol, tasks, checkpoint hashes and runtime. An exclusive lock prevents concurrent writers. Every case is saved before its fsynced canonical row; same-identity orphan traces can be recovered. Repeating the study verifies completed trace checksums and skips completed keys. Different identity in the same namespace is rejected.

Results must be interpreted within this explicit software intervention design. Ordinary-test/OOD differences, failed cases and any weak or null mechanism effects must be retained. Broad foundation-model telemetry and historical-model comparisons remain outstanding. Exact GPT/Gemini releases, access and authorized spending are required before those comparisons can run.

The actual run and findings are in [the scientific report](scientific_report.md). Restore every saved cloud trace and result from a fresh clone without inference:

```sh
python scripts/research_v2/evidence_archive.py --verify-existing --archive research_checkpoint/research_v2_numeric_2026-10-09 --restore work/numeric_study_restore
```

The checksum-bound dashboard is `mechanism_study/dashboard.html` inside the restored evidence. The same report and raw statistics are versioned under `experiments/research_v2/cloud_runs/37946027443/mechanism_study`.
