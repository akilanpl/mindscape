# Phase 2 milestone: dataset and benchmark infrastructure

## Implemented
General benchmark schemas and backend registry; deterministic configurable generation;
identity-aware splits, OOD isolation, checksum/target/leakage validation; three separate
data forms; nested budgets; measurable difficulty filters; independent metrics/N*/DER;
abstract model interface, failure records, runner, immutable artifacts and four CLI operations.

## Tested and verified
54 tests passed on Python 3.12.14, including all 21 original tests. Tests cover generation,
seeds/byte reproducibility, JSON schema roundtrips, targets, digit ranges, signed inputs,
difficulty bounds, split identities, swapped OOD/operands, trace contamination, target-free
views, subsets, metric denominators, missing/malformed/foreign/incomplete trajectories,
N*/DER, exception records, pre-call validation, no overwrite and repeat-run predictions.
A non-arithmetic test-only backend verifies generic pipeline reuse.

Development dataset: train 1000; validation 250; IID test 500; OOD test 500.
All targets and leakage checks passed. Nested budgets: 10,25,50,100,250,500,1000.

| Split | Examples | Accuracy | Grounded rate | Trajectory validity | Goal success | Artifact |
|---|---:|---:|---:|---:|---:|---|
| test | 500 | 1.0 | 1.0 | 1.0 | 1.0 | results/20261008T005539Z_26822e7d16a44754af415c71c728034d |
| ood_test | 500 | 1.0 | 1.0 | 1.0 | 1.0 | results/20261008T005539Z_d7e71c7dd32e46e5864e15af0d53a552 |

These are **deterministic reference infrastructure results**, not learned model results.
No DER or neural data-efficiency evidence exists. Training time is null; no training occurred.

## Not implemented
Neural training/downloads, learned baselines, experimental comparisons, interactive
experiential learning, confidence intervals, UI, dream engine or published historical scores.

## Known limitations
Rejection sampling is unbalanced and exhausts the finite 1x1 family in train; IID
evaluation has only 2x1/2x2 examples. Full datasets/traces are in memory. Directory
creation prevents overwrite but filesystem-write failures can leave partial directories.
Future backends must implement identity/structure/validation/evaluation correctly.
Stored schema checks are strict for core traces; benchmark metadata uses backend validation.
Python model views are API isolation, not an adversarial filesystem sandbox. model_calls
relies on model reporting; timeout records require the model to raise TimeoutError, not
a runner-enforced wall-clock timeout. Training budget is metadata/subset identity only:
future training must consume that subset. Ruff/MyPy/pytest and hosted CI were not run.

## Relevant repository tree
```text
configs/experiments/{iid,ood,data_efficiency}.toml
src/mindscape/core/serialization.py
src/mindscape/data/{schemas,backends,generation,splits}.py
src/mindscape/evaluation/{schemas,metrics,evaluator,runner}.py
src/mindscape/models/{base,reference}.py
scripts/{generate_data,dataset,evaluate}.py
datasets/generated/development_v1/{manifest.json,*.jsonl}
results/<unique_id>/{config.json,metrics.json,predictions.jsonl,summary.md}
docs/{benchmark,experimental_protocol,historical_references,milestone_02}.md
tests/{test_foundation,test_benchmark}.py
```

## Git
History preserved on main. Phase-2 commits add infrastructure, verification tests,
and documented development artifacts. Results remain available locally but are
ignored by Git. Final working-tree status and latest commit checked after committing.
