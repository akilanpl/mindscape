# Mindscape

**Mindscape: A Multimodal Cognitive Architecture for Grounded Event Representation
and Structured Goal-Directed Reasoning**

This research project tests whether explicit state-transition representations can
improve data efficiency, structural generalization, and groundedness relative to
answer-oriented learning. No superiority or learned reasoning result is established.

Observation → state → action → event/result → state update → verification → goal.

## Current implementation

First learned prototype now available: local NumPy answer-only MLP and trajectory-supervised
action policy, checkpoint training/evaluation, working memory and SQLite episodic API.
69 tests pass. See `docs/milestone_03.md` for measured development results. Singleton
action masks force success; arithmetic-tool access differs, so no superiority is established.


Phases 0–3: immutable typed schemas, a deterministic integer multiplication
environment, ordered trajectories, replay verification and a command-line demonstration.
The environment performs decimal digit/carry operations for arbitrary operand lengths,
including zero and negative operands. Python multiplication is used for individual digit
operations and as an independent final-answer oracle, never as the whole procedure.
The policy chooses the single legal next procedural action; this is a hand-authored
algorithm, not learned intelligence. Evidence categories remain explicit.

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

The planned benchmark measures exact accuracy, OOD accuracy, data efficiency,
grounded answer rate, transition validity, goal success, and compute.
Dataset and benchmark infrastructure is implemented; preliminary learned development evaluations now exist; no valid data-efficiency claim exists. `docs/benchmark.md`
defines operational metrics and their denominators. Historical GPT/Gemini
results will be documented with exact versions, datasets, dates, sources and
protocols in a later phase. Multiplication scores cannot be compared directly
with GSM8K, MATH or MMLU scores. No historical numbers are asserted here.

## Roadmap

Seeded datasets, checked structural splits and reference evaluation are implemented.
Regimes A/B and local task MLPs are implemented. Next: regime C, matched-tool
controls, meaningful action choices, fair data-efficiency studies, ablations and historical
references. Bounded hypothetical simulation, a second vertical, UI and multimodal
observations follow only after the core experiments. Directory placeholders do
not imply implementations.

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
