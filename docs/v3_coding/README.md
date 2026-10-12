# Reproduce and hand off V3 Python repair

Read `scientific-report.md` for observed results and limitations. The experimental source/benchmark froze at `38a8494`; the throughput-only runner froze at `267a4ad`. Original V1/V2 data, Qwen adapters, sandbox source, release tags and historical lockbox remain immutable.

## Setup

Use Python3.12 with an isolated environment and the observed runtime versions in `configs/v3_coding/requirements-local.txt`. An existing pinned Qwen1.5B snapshot and an already prepared **copy** of the pinned WASI runtime are required. No model download or paid API is automatic. Do not run the old bootstrap casually: it downloads both models. The two measured runtime inputs are:

- Qwen2.5-Coder-1.5B-Instruct revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- WASI Python SHA256 `d24bd98d3071af6b17d51d53a08700b9acef59172a0afcb6adb733645c2a1715`, wasmtime49.0.0. Preserve the old runtime; compile a fresh copy for a different host/architecture.

```sh
export PYTHONPATH=src
coding_python=work/final-venv/bin/python
coding_model=/absolute/path/to/existing/Qwen/snapshot
$coding_python -m unittest discover -s tests_coding_v3 -v
$coding_python -m mindscape.v3.coding_learning.train --manifest configs/v3_coding/benchmark_v1 --output work/coding-training-new --runtime work/coding/runtime
$coding_python -m mindscape.v3.coding_learning.experiment --manifest configs/v3_coding/benchmark_v1 --training work/coding-training-new --model "$coding_model" --runtime work/coding/runtime --output work/coding-scalar-new --seconds 1800
$coding_python scripts/v3_coding/batch_resource_probe.py --model "$coding_model" --output work/coding-batch-probe.json
$coding_python scripts/v3_coding/batched_study.py --manifest configs/v3_coding/benchmark_v1 --training work/coding-training-new --model "$coding_model" --runtime work/coding/runtime --output work/coding-batched-new --seed 11 --batch-size 4 --seconds 1800
# Repeat an identical study command to resume only unfinished task/condition keys.
$coding_python -m mindscape.v3.coding_learning.analysis --run work/coding-batched-new --output work/coding-analysis.json
$coding_python -m mindscape.v3.coding_learning.verify --run work/coding-batched-new --manifest configs/v3_coding/benchmark_v1 --runtime work/coding/runtime --output work/coding-replay.json
```

Run evaluator seeds23/37 with separate output directories and the same generation cache. Cached deterministic Qwen outputs are shared actual evidence, not fresh independent samples. Inspect all `invocation-*.json` receipts, including incomplete caps and completed no-op resumes, to total actual compute. Do not sum cache-hit episode wall time or overlapping batch request durations as exclusive generation work. The full 18 held-out tasks × eight conditions is144 episodes per evaluator seed; a shorter receipt is incomplete, never silently reported as a full benchmark.

## Data and security

`public.json` is the only learner task source. `private.json` contains terminal scorer cases and training teacher repositories. `validation.json` retains all192 actual generation-validation executions. Never put the private file or validation archives in model context. These files are auditable researcher artifacts rather than a cryptographically sealed new lockbox. The **historical lockbox** is not opened or re-evaluated in this study; its hashes are checked only.

All persistent memories require approved training identities and real successful visible + training-private execution. Evaluation passes cannot update them. Hash checks alone are not a trust mechanism; memory loading re-executes approved training repairs. No model-generated Python is run on the host. Static parsing is permitted; all function execution goes through the original WASI worker. Unsupported capabilities/imports fail closed.

## Diagnostics and future work

`historical_audit.py` reads saved historical result traces only. `legacy_bridge.py` reruns original B/C adapters on two new public tasks under their original eight-action loop; it is an exploratory bridge with unequal training history/budgets, not a direct rate comparison to historical48%.

`path_schema.py` is a post-freeze transport remediation, validated on **training tasks only**. It removes the fictitious filename example from the instruction and enumerates actual allowed paths. It never rewrites proposed code. Its training-only observations must not be substituted for the frozen V1 structural-OOD study. A new frozen benchmark is required before treating a revised prompt as confirmatory generalization evidence.

Reference-source upper bounds, teacher examples and learned repairs are explicitly distinguished. New Qwen LoRA training is not performed; retained checkpoints train only the small proposal evaluator. V1 memories/adapters are never overwritten. Raw logs, failed outputs, per-attempt predictions, actual test traces, information ledgers, checkpoint hashes, resource receipts and preservation receipts belong in the evidence export.
