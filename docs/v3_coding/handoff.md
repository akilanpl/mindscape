# V3 coding handoff

## Entry points

- `src/mindscape/v3/coding_learning`: typed tasks, AST extraction, conservative capabilities, WASI execution, verified memory, learned evaluator, bounded agent, training, evaluation, analysis and independent replay.
- `configs/v3_coding/benchmark_v1`, `benchmark_v2`: versioned public tasks, independent private scorer records, hashes and actual defect/reference validation receipts.
- `configs/v3_coding/protocol_v2.json`: final frozen follow-up protocol and exact execution-helper hashes.
- `scripts/v3_coding/transport_v2_study.py`: primary hash-checked CPU study.
- `scripts/v3_coding/complete_study.py`: repeated capped invocations to complete coverage and test no-op resume; does not resample saved failures.
- `scripts/v3_coding/audit_decisions.py`: bind raw model text to edits, neural predictions and selected actions without Qwen inference.
- `scripts/v3_coding/representation_collision.py`: executed training-only feature expressivity falsification.
- `scripts/v3_coding/software_controls.py`: unchanged/constant-zero negatives and explicit teacher upper bound.
- `scripts/v3_coding/evidence.py`: deterministic multipart export, hash verification and safe restore; active run locks block export.
- `tests_coding_v3`, `.github/workflows/v3-coding.yml`: isolated coding/security/scientific controls with no LM downloads in CI.

## Commands actually used locally

Working directory was `/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3`.

```sh
export PYTHONPATH=src
coding_python=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python
coding_model=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a
$coding_python -m mindscape.v3.coding_learning.train --manifest configs/v3_coding/benchmark_v1 --output results/v3_coding/training_v1 --runtime work/coding/runtime
$coding_python scripts/v3_coding/transport_v2_study.py --model "$coding_model" --runtime work/coding/runtime --training results/v3_coding/training_v1 --output results/v3_coding/transport_v2_s11 --seconds 1800
$coding_python scripts/v3_coding/complete_study.py --model "$coding_model" --runtime work/coding/runtime --training results/v3_coding/training_v1
$coding_python scripts/v3_coding/summarize_study.py --root results/v3_coding
$coding_python -m pytest tests tests_coding_v3 -q
$coding_python -m ruff check src/mindscape/v3/coding_learning scripts/v3_coding tests_coding_v3
$coding_python scripts/v3/preservation.py --output results/v3_coding/preservation-delivery.json
$coding_python scripts/v3_coding/preserve_inputs.py --output results/v3_coding/immutable-inputs.json --verify
```

`complete_study.py` records subprocess commands, exit codes, coverage and full invocation durations. Each study directory records protocol/source/backbone/checkpoint identity, raw traces and separate per-invocation fresh compute. Seeds23/37 use their trained evaluator weights but reuse identical deterministic Qwen responses. Source/parser/weights/task identities cannot change on resume. Do not “fix” a failed benchmark repair and resume the same identity.

## Inspect and replay delivered evidence

```sh
$coding_python scripts/v3_coding/evidence.py restore --package experiments/v3_coding/release_v1 --destination work/restored-coding
$coding_python -m mindscape.v3.coding_learning.verify --run work/restored-coding/transport_v2_s11 --manifest configs/v3_coding/benchmark_v2 --runtime work/coding/runtime --output work/replay-s11.json
$coding_python scripts/v3_coding/audit_decisions.py --run work/restored-coding/transport_v2_s11 --manifest configs/v3_coding/benchmark_v2 --training work/restored-coding/training_v1 --seed 11 --output work/actions-s11.json
$coding_python scripts/v3_coding/retrain_evaluator.py --training work/restored-coding/training_v1 --seed 11 --output work/retrain-s11.json
```

Repeat replay/audit for seeds23/37. Replay requires the pinned WASI runtime, not Qwen weights. Retraining from frozen labels requires small CPU Torch, not Qwen. Full inference requires the separately supplied existing local backbone. Runtime/backbone/adapters and old lockbox contents are excluded from the new evidence export. Training-evidence timestamps/runtime labels are part of the frozen numerical input: recollecting timings is a fresh training run, not identical reproduction.

## What to read first

Read `scientific-report.md`, then `baseline-matrix.md` and `architecture.md`. `ablation-results.csv` exposes all cells; `study-summary.json` includes per-family counts, paired intervals, all failure identities, candidate counts and fresh compute. Raw episode files retain source proposals and executed tests. No observed failure is dropped. Original failed V1 transport pilots and their partial/resource receipts remain in the archive.

## Explicit incomplete work

No new Qwen LoRA training, broad real-repository benchmark, fully equal-information trained baseline, full semantic/alias analysis, robust adversarial scorer proof, learned long-horizon policy or optimizer-state training resume is delivered. The optional historical-adapter bridge is not part of the matched primary study and must not be used for a direct new-vs-old percentage comparison. The old benchmark is never reopened. Published new researcher-private JSON is not a sealed future test set; freeze new unseen tasks before evaluating a revised mechanism.

The next hypothesis should address the executed literal-feature collision and realistic training candidate distribution before larger domain training. Preserve this release and create a new dataset/protocol/model version for that work. Do not tune V2 until it passes an aspirational threshold.
