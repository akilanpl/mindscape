# Mindscape

**Mindscape: A Multimodal Cognitive Architecture for Grounded Event Representation
and Structured Goal-Directed Reasoning**

This research project tests whether explicit state-transition representations can
improve data efficiency, structural generalization, and groundedness relative to
answer-oriented learning. No superiority or learned reasoning result is established.

Observation → state → action → event/result → state update → verification → goal.

## Current implementation

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
No learned benchmark or data-efficiency curve exists yet. `docs/benchmark.md`
defines acceptance targets; they are not measured results. Historical GPT/Gemini
results will be documented with exact versions, datasets, dates, sources and
protocols in a later phase. Multiplication scores cannot be compared directly
with GSM8K, MATH or MMLU scores. No historical numbers are asserted here.

## Roadmap

Next: seeded datasets and contamination-free structural splits; then local model
baselines, regimes A/B/C, fair data-efficiency studies, ablations and historical
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
