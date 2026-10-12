# Mathematical learning v1: reproduce and inspect

Read [scientific-report.md](scientific-report.md) before interpreting accuracy. This is a completed negative-result study: the length-family target failed, and finite memory can outperform the trained mechanism with zero neural weights. The mathematical environment and learner remain useful research infrastructure; they do not establish abstract rule learning.

## Runtime and commands

Python 3.12; actual local versions are pinned in `configs/v3_math/requirements-local.txt`. Use an isolated environment. CPU only is sufficient. Transformers and the cached MiniLM snapshot are optional; no download occurs automatically.

```sh
python -m venv work/math-venv
work/math-venv/bin/pip install -r configs/v3_math/requirements-local.txt
export PYTHONPATH=src
PY=work/math-venv/bin/python
$PY -m unittest discover -s tests_math_v3 -v
$PY -m mindscape.v3.math_learning.experiment --manifest configs/v3_math --output work/math-replication --budgets 10 25 50 100 250 500 1000 --seeds 0 1 2 --updates 600 --modes procedural concepts no_memory no_planning answer_only
# Repeat the identical command for hash-checked resume.
$PY -m mindscape.v3.math_learning.verify --run work/math-replication --output work/math-verification.json
$PY -m mindscape.v3.math_learning.analysis --help
$PY scripts/v3_math/full_mechanism_controls.py --help
$PY scripts/v3_math/infer.py --help
```

The secondary adapter uses the same experiment command with `--pretrained-path PATH_TO_EXISTING_MINILM_SNAPSHOT` and omits `answer_only` from modes. Check `--help` for exact parser options. The run rejects changes in source, manifest, backbone files or resume identity. Use a new output directory for changed protocols. Actual runs live under `results/v3_math/{compact,minilm-adapter}` before archival.

## Exported evidence

`experiments/v3_math/release_v1/manifest.json` binds multipart compressed evidence, including all 63 checkpoints, raw traces, predictions, information ledgers, aggregate analyses, failure analysis, test logs and checkpoint/preservation audits. No pretrained weights or lockbox key are exported.

```sh
python scripts/v3_math/evidence.py restore --package experiments/v3_math/release_v1 --destination work/math-evidence-restored
PYTHONPATH=src python -m mindscape.v3.math_learning.verify --run work/math-evidence-restored/compact --output work/compact-replay.json
PYTHONPATH=src python -m mindscape.v3.math_learning.verify --run work/math-evidence-restored/minilm-adapter --output work/adapter-replay.json
```

Restoration checks each part and every file before writing. The verifier replays actual numerical claims without calling a neural model. Checkpoints contain optimizer/RNG state; their hashes are checked before loading with Torch's restricted `weights_only` loader.

## Handoff and limits

- Frozen source/protocol commit: `b43d778`. Development branch: `development/v3-mathematical-learning`, stacked on the validated V3 foundation.
- The production lockbox remains encrypted. Its operator key is retained only at ignored `work/math-lockbox-key.bin`; do not publish it or decrypt it for development. Synthetic encryption tests do not open this lockbox.
- Twenty math tests live in `tests_math_v3`, outside preserved V2 discovery, to keep optional encryption/model dependencies isolated. Full local validation explicitly included both directories: 188 tests and six subtests, no skips.
- New CI performs a small 20-update replication, resume and independent replay; it is a diagnostic control, not a replication of the 600-update primary curves.
- Baseline matrix, paired intervals, information accounting and measured compute are in the scientific report. No alternate-base experiment or peak-memory measurement was performed. Pretrained answer-only training was not run.
- Next research should freeze a new protocol with balanced elementary-context demonstrations, equally informed numeric/relational baselines, and mandatory memory-only controls. Do not tune this study's lockbox or reclassify an easier family as the target.
