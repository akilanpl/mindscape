# Mindscape

**Mindscape: A Multimodal Cognitive Architecture for Grounded Event Representation
and Structured Goal-Directed Reasoning**

This research project tests whether explicit state-transition representations can
improve data efficiency, structural generalization, and groundedness relative to
answer-oriented learning. No superiority or learned reasoning result is established.

Observation → state → action → event/result → state update → verification → goal.

## Final research build

The final controlled study uses one pinned, free Apache2.0 SmolLM2-135M-Instruct
frozen backbone with identical 337,618-parameter trainable heads for answer-only,
structured, trajectory-supervised B and experiential C. It evaluates seven nested
problem budgets 10–1000 with three seeds, separate IID/structural OOD, 100 numerical
action candidates and no inference-time arithmetic correction. Historical MLPs,
negative studies and Git history remain intact.

See [final results](docs/final_results.md), [research conclusions](docs/research_conclusions.md),
[protocol](docs/final_experimental_protocol.md), [architecture](docs/final_architecture.md),
[limitations](docs/limitations.md) and [exact reproduction commands](docs/final_reproducibility.md).
The static [evidence demo](demo/index.html) displays frozen public execution records,
including wrong actions, with an answer-only/Mindscape toggle.

The architecture implements typed state/action/event/result/goal trajectories,
a generic environment interface, independent replay verification, working/episodic/
concept memory and bounded hypothetical simulation. Decimal cursor/decomposition
is manually designed. No independent architecture, compute-efficiency or universal
reasoning advantage is implied by the implementation.

Phases1–4 are historical milestones: deterministic decimal execution, dataset/leakage
infrastructure, NumPy learned prototypes and the controlled claims study. Their
schemas, commands and results remain available. Early singleton-mask success was
confounded and is not a final learned result. The 92 passing tests cover both legacy
and final paths.

## Installation and execution

Python 3.11+; the core has no runtime dependencies and requires no paid service.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
make test
make demo
PYTHONPATH=src python scripts/run_episode.py -123 45
```

Offline execution without installation:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_episode.py 19 3
```

The demo prints initial state, actions, events, actual results, every before/after
state, final answer, independent verification, and goal status as JSON.

## Benchmark and historical references

The completed benchmark measures exact accuracy, structural OOD, data efficiency,
groundedness, trajectory validity, goal success and compute. Final protocols,
results and controls are documented in `docs/final_results.md` and
`docs/final_experimental_protocol.md`. Historical GPT-4/Gemini sources and
comparability limits are retained in `docs/historical_references.md`.
Their GSM8K scores are unsuitable for direct comparison to this multiplication task.

## Remaining research directions

The final multiplication build and validation are complete. Full pretrained-model
fine-tuning, independently reserved carry-only holdouts, learned cursor planning,
matched query budgets, a second learned environment and multimodal observations
remain future work. None is implied by directory placeholders or by the existing
architecture. See `docs/research_conclusions.md` and `docs/second_vertical.md`.

## Reproducibility

The environment is deterministic, frozen records prevent mutation, and verification
replays the authoritative transition logic from the original observation.
`scripts/run_episode.py --output PATH` exclusively creates a JSON artifact and
refuses to overwrite an existing file. Tests require no network. Future experiments
must record seeds, configurations, model/data versions, splits, hyperparameters,
checkpoints, software, hardware, runtime and raw predictions. See
`docs/experimental_protocol.md`.

## Dataset and benchmark commands

From the repository root, with Python 3.11+ and PYTHONPATH=src:

```sh
export PYTHONPATH=src
python scripts/generate_data.py --config configs/experiments/data_efficiency.toml --output datasets/generated/my_run
python scripts/dataset.py validate datasets/generated/my_run
python scripts/dataset.py inspect datasets/generated/my_run
python scripts/dataset.py export datasets/generated/my_run --form answer_only --split train --output answer_train.jsonl
python scripts/evaluate.py --dataset datasets/generated/my_run --split test --training-size 1000
python scripts/evaluate.py --dataset datasets/generated/my_run --split ood_test --training-size 1000
```

Use a new output path for generation/export. Config examples: iid.toml, ood.toml and
data_efficiency.toml. Development v1 contains 1000 train, 250 validation, 500 IID test
and 500 OOD test examples; its nested budgets are 10,25,50,100,250,500,1000.
The CLI evaluator uses a deterministic procedural reference only. Models later plug
into models/base.py and evaluation/runner.py. JSON results live under results/ and
are never overwritten. Run IDs/times vary; datasets, predictions and scores reproduce.
See docs/milestone_02.md for actual infrastructure checks and limitations.

## Learned training and demonstration

Install `.[learning]` for NumPy and optionally `.[plots]` for ReportLab charts. No pretrained
model download is required; this prototype is a small task classifier/decoder, not an LLM.

```sh
PYTHONPATH=src python scripts/run_learning_experiment.py --dataset datasets/generated/development_v1 --output results/new_learning_run
PYTHONPATH=src python scripts/run_learned_episode.py 19 3 --checkpoint results/new_learning_run/mindscape_n50_seed0/checkpoint --unmasked
PYTHONPATH=src python scripts/plot_learning_curves.py results/new_learning_run
```

See `docs/training.md` for individual baseline/policy training and shared evaluation
commands; `docs/modeling.md` documents features, architecture, memory and confounds.
The unmasked learned policy succeeds on the saved 19×3 demonstration. The frozen
preliminary protocol uses three seeds at 50 examples and one at 100/250. Raw measured
curves and standalone SVGs are retained in `experiments/first_learned/`. Full checkpoints,
logs and per-example predictions remain in `results/learned_development_v1/` locally.
Regime C is explicitly unimplemented; no experiential learning claim is made.

## Historical phase-4 controlled research study

The numerical-claim study supersedes prior oracle-assisted performance interpretation.
Each policy must predict its numerical result; primary execution never corrects it.
All four conditions use identical 8,722-parameter models. Seven budgets, three seeds:
90 trained models (including interventions), 204 evaluations, 86 tests passed.
At 1000 examples, Mindscape B IID accuracy is 6.333% ± 1.443%; C is 6.000% ± 1.732%.
Every primary OOD run has 0% measured accuracy; N*/DER targets are not reached.
These results do not reproduce the earlier exploratory 82% OOD score or establish
architecture superiority. See `docs/milestone_04.md` and `docs/research_protocol.md`.

```sh
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
python scripts/generate_data.py --config configs/experiments/research_dataset.toml --output datasets/generated/new_research_set
python scripts/run_research_study.py --dataset datasets/generated/new_research_set --output results/new_research_study
python scripts/analyze_research_study.py results/new_research_study
python scripts/audit_research_study.py results/new_research_study --dataset datasets/generated/new_research_set
python scripts/run_dream_demo.py --depth 3 --goal 3
```

Individual training/evaluation commands are `scripts/train_research.py` and
`scripts/evaluate_research.py` (`--help`). Regime C is self-generated scalar-feedback
learning plus positive replay, not autonomous rule discovery. Pretrained encoder/head
backend is optional and offline by default (`.[pretrained]`); it was not used in these
experiments. Publication SVGs, raw summaries, confidence intervals, failures and fairness
checks are committed under `experiments/research_claims_v1/`; full local records and
checkpoints remain under `results/research_claims_v1/`. No UI was built.

Cloud continuation and verified raw-artifact restoration: [migration instructions](docs/cloud_migration.md). Historical migration snapshot: primary evidence 400/400; learning 1,122/1,920 was partial. The completed release is described below.

<!-- coding-completion-release -->

## Completed coding research package

The frozen Qwen2.5-Coder-1.5B study contains 400/400 locked condition/task cases (A100, structured100, B100, C100), 1,920/1,920 learning cases, independent replay of all 400 locked trajectories and 16 dedicated latency cases. See [actual metrics](docs/final_metrics.md), [claims](docs/final_claims.md), [limitations](docs/final_limitations.md), [package verification](docs/final_package_verification.md) and [recorded A/C playback](demo/coding/index.html). Completion describes experimental coverage, not a positive scientific result. Historical negative studies are preserved.

## Research v2

Canonical v1 preservation and clean restoration are verified. A real Linux CPU cloud validation and safe resume passed. See [reproduction](docs/research_v2/reproduction.md), [cloud workflow](docs/research_v2/cloud_workflow.md), [observed computation traces](docs/research_v2/computation_tracing.md), and [frozen numeric mechanism protocol](docs/research_v2/mechanism_study.md).

The numeric-core study is a bounded intervention experiment, separate from coding-foundation telemetry and historical GPT/Gemini comparisons. Neither broad intelligence nor foundation-model superiority is established. Consult versioned stage receipts under `experiments/research_v2` for actual execution state.
