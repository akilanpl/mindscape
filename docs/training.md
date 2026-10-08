# Local reproducible learning

Install optional `.[learning]` dependencies (NumPy); `.[plots]` adds ReportLab for
standalone SVG charts. Existing deterministic core remains dependency-free.

```sh
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python scripts/train.py --dataset datasets/generated/development_v1 \
  --config configs/experiments/learning.toml --kind baseline --budget 50 --seed 0 \
  --output results/my_baseline
python scripts/train.py --dataset datasets/generated/development_v1 \
  --config configs/experiments/learning.toml --kind mindscape --budget 50 --seed 0 \
  --output results/my_policy
python scripts/evaluate.py --dataset datasets/generated/development_v1 \
  --checkpoint results/my_baseline/checkpoint --regime answer_only --split test
python scripts/evaluate.py --dataset datasets/generated/development_v1 \
  --checkpoint results/my_policy/checkpoint --regime trajectory_supervised --split ood_test
python scripts/evaluate.py --dataset datasets/generated/development_v1 \
  --checkpoint results/my_policy/checkpoint --regime trajectory_supervised --split test --unmasked
python scripts/run_learning_experiment.py --dataset datasets/generated/development_v1 \
  --output results/my_comparison
python scripts/plot_learning_curves.py results/my_comparison
```

Output paths must be new. Evaluation verifies checkpoint dataset hash, training
budget and regime against the same dataset manifest. Training reads only the recorded
nested training subset and validation labels. The held-out sets are validated for
integrity/leakage but never passed to fitting or checkpoint selection. Seeds change
weight initialization and mini-batch sampling; the dataset/subset IDs remain fixed.

Fixed preliminary config: 400 Adam updates, batch 64, learning rate .003, validation
loss checks every 50 updates. Choose the minimum validation-loss checkpoint. Both
systems use identical optimizer steps, batch size, seed policy and training problem
budgets; the policy receives multiple supervised transition labels per problem.
The minibatch units therefore differ; label counts, selected step and runtime are
recorded. No hyperparameter tuning on IID/OOD scores occurs in the comparison script.

Training outputs: config.yaml (JSON syntax, valid YAML 1.2) and matching config.json with complete dataset manifest, metadata.json,
metrics.json, training.jsonl and checkpoint/{weights.npz,backend.json,model.json}.
Evaluation uses the existing runner and saves full configs, metrics and per-example
predictions with action/transition diagnostics. Comparison outputs retain fairness
conditions, raw curves, failure counts and the full 19×3 learned demonstration.
All run directories refuse overwrite. Reported training time measures optimization
and validation only, excluding data loading/adaptation/checkpoint I/O. Existing
inference time includes model execution and per-example assessment, not loading or
artifact writes. Model calls count one per baseline example or per policy decision.

Checkpoint reload is tested and used before benchmark evaluation. Resume-training
optimizer state is not saved; restarting a training run uses the recorded config/seed
rather than continuing Adam moments. Reproduction is exact on the tested NumPy/CPU
setup for weight arrays and predictions. Different NumPy/BLAS/CPU versions can cause
floating-point or RNG differences; versions and Git revision are recorded. Timing,
UUIDs and timestamps naturally differ. NumPy's random Generator has no cross-version
compatibility guarantee: https://numpy.org/doc/stable/reference/random/generator.html

TensorBoard is unavailable; plain machine-readable logs provide tracking. No external
tracking service or paid API is used. The SVG plots are labelled preliminary and retain
raw data; 50-example points average three seeds, larger points use one seed. Final
statistical studies and confidence intervals remain future work.

## Controlled numerical-claim revision

The revised study is documented in research_protocol.md and milestone_04.md.
Prior singleton/oracle-assisted results are exploratory, not current evidence.
Primary numerical-claim execution writes model proposals without correction;
Regime C uses training-only scalar feedback and positive replay. All four conditions
share capacity and the evaluator; remaining procedural/supervision/compute differences
are explicit. All reported thresholds are censored; no data-efficiency advantage
has been established. Use the research-specific CLI/config files for this study.
