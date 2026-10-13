# Mindscape

Mindscape is an experimental AI research system for mathematics, Python coding,
and simulated physics. It investigates explicit state transitions, structured
representations, and shared neural learning through frozen experiments and
independent verification.

The latest implementation is a **versioned three-domain research kernel**.
Engineering integration is working; learned cross-domain transfer and neural
superiority have **not** been demonstrated. This repository is not a public
Mind1.0 release or Research Alpha launch.

## Current status

| Component or study | Verified status | Interpretation |
|---|---|---|
| V3 unified runtime | Engineering integration verified | Specialists can execute through a common runtime; routing is not learned transfer. |
| V4 shared candidate scoring | **NO_GO** | Semantic transfer was not demonstrated. The fresh-data qualification screen also stopped at **DATA_NO_GO**. |
| V5 structured transition prototype | **DATA_NO_GO** | Original qualification failed; its architecture remains available for research. |
| R5.1 learning readiness | **DATA_READY** | 81 tasks executed and independently replayed before the learning pilot. Readiness was not a learning result. |
| R5.1 shared neural learning | **NO_GO** | Sharing did not meet matched-control gates; negative transfer and a target-exposure confound remain documented. |
| R5.2 mathematical generalization | **NO_GO — post-hoc recovery** | Measurements recovered from frozen predictions; the original **FAILED_CAMPAIGN** remains preserved. |
| R5.3 physics superiority | **NO_GO — post-hoc recovery** | Measurements recovered from frozen evidence; the original **FAILED_CAMPAIGN** remains preserved. |
| Current R5 kernel | **CPU tool integration PASS** | 15 focused tests and six direct/kernel comparisons pass. Native MPS/shared-core and Qwen repair integration smoke checks are **NOT_RUN**. |

Successful symbolic computation, simulator execution, and specialist integration
are distinct from learned reasoning. None reverses the scientific NO_GO findings.

## Available capabilities

| Domain | Default pathway | Experimental pathway and limits |
|---|---|---|
| Mathematics | Verified symbolic/procedural integer arithmetic, division, fractions, elementary algebra and equations, matrix products, coordinate geometry, polynomial calculus, probability and statistics | Fixed R5.2 recurrent arithmetic checkpoint. Neural outputs remain uncorrected and are independently checked. Symbolic support is bounded, not comprehensive mathematical mastery. |
| Python coding | Existing isolated WASI execution; preserved Qwen coding repair specialist, evaluator and verified training memory | Existing Qwen/WASI behavior is unchanged. Passing visible tests does not establish general program correctness. Local model/runtime assets are required. |
| Simulated physics | Analytical contact/spring forecasts, with separate independent Pymunk execution when a verifier scene is provided | Fixed R5.3 neural forecasts remain experimental. The analytical baseline had lower reported primary OOD forecast error, but its uncertainty intervals were undercovered and remain labeled uncalibrated. |
| Shared transitions | Explicit experimental mode only | Original 6,967-parameter V5 model with domain-specific encoders/heads and a genuinely shared trainable core. Supports bounded register, AST-edit and impulse graphs—not a universal ontology or demonstrated transfer. |

The runtime loads neural assets sequentially through one model slot to limit memory
use on the Apple Silicon development machine. Default symbolic, WASI execution,
and analytical physics paths do not require neural model loading.

See the [capability/evidence report](experiments/r5_kernel/REPORT.md) for selection
rationale, checkpoint provenance, negative results and limitations.

## Unified interface

The stable contract identifier is **`mindscape.kernel/1`**:

```text
observation → state representation → action → prediction
            → execution / verification → trace
```

Requests specify `version`, `domain`, `operation`, `mode`, `observation` and
`action`. Supported modes are `validated`, `experimental_neural` and
`experimental_shared`. Here, `validated` identifies the selected tool pathway;
it is not a claim of perfect predictions or calibrated uncertainty.

Responses preserve the original prediction, execution outcome, verification,
model/tool provenance and ordered trace hashes. They distinguish neural inference,
symbolic computation, procedural prediction and actual execution. Private verifier
context is supplied separately from public observations. Unsupported requests are
rejected; a verifier never silently replaces a neural prediction with a reference
answer. Specialist routing is always labeled `learned_transfer=false`.

See the [interface and operation guide](configs/r5_kernel/README.md).

## Run examples

Run from the repository root using the existing project Python environment.
The examples below perform inference or tool execution, not training.

```sh
export PYTHONPATH=src:work/physics-deps/site-packages
python -m mindscape.r5_kernel --capabilities
python -m mindscape.r5_kernel --request configs/r5_kernel/example-math.json
python -m mindscape.r5_kernel --request configs/r5_kernel/example-coding.json
python -m mindscape.r5_kernel --request configs/r5_kernel/example-physics.json --verifier configs/r5_kernel/example-physics-verifier.json
```

The committed CPU smoke evidence includes integer composition `−432`, fraction
addition `3/2`, a verified matrix product, polynomial integral `14`, actual WASI
execution returning `15`, and an approximate physics forecast followed by
independent action replay. Forecast error remains visible rather than being
replaced with simulator truth.

### Environment and assets

Python 3.11+ is required. The historical base package has no runtime dependencies;
the current kernel's domain paths require additional packages. The focused CPU
test workflow pins NumPy 2.3.5, SymPy 1.14.0 and Pymunk 7.2.0. Coding and neural
paths additionally need the existing WASI/PyTorch/Transformers environment and
locally preserved assets.

For a basic editable installation and public CPU tests:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . numpy==2.3.5 sympy==1.14.0 pymunk==7.2.0
PYTHONPATH=src python -m unittest discover -s tests/r5_kernel -v
```

A fresh clone does not include every large model or local artifact. Asset-dependent
WASI/checkpoint tests skip when those assets are absent; this is not equivalent to
the fully populated local run. The [asset registry](configs/r5_kernel/assets.json)
records expected hashes and local paths. The native launcher targets the existing
macOS development environment and is not a portable installer.

### Native engineering smoke

On the configured Apple Silicon machine:

```sh
./scripts/r5_kernel/native.command
```

This bounded, nontraining smoke checks MPS gradient connectivity, action/relation
sensitivity, shared parameter identity, unchanged weights, experimental model
loading and preserved Qwen repair parity. It refuses an existing output directory
and does not automatically rerun. These native checks remain pending; the launcher
is not evidence that they passed.

Completed smoke receipts can be checked without another neural forward:

```sh
PYTHONPATH=src python scripts/r5_kernel/verify_smoke.py results/r5_kernel/native-smoke
```

## Verification and scientific boundaries

- [15 focused regression tests](experiments/r5_kernel/tests.txt): contract behavior,
  bounded mathematics, numerical precision, real WASI isolation, Pymunk replay,
  privacy boundaries, failed-prediction preservation, checkpoint integrity and
  resource lifetime.
- [CPU example receipts](experiments/r5_kernel/cpu-smoke-final/summary.json) and
  [independent audit](experiments/r5_kernel/cpu-verification.json): six direct/kernel
  comparisons, zero optimizer steps and 1,726 historical files preserved by hash.
- [Frozen shared-learning proposal](configs/r5_kernel/shared-study-proposal.json):
  proposed matched target exposure and baseline-headroom requirements only. No new
  tasks, holdout inspection or optimizer execution is authorized by integration.

The integration commit skips automatic CI because existing unconditional historical
workflows run neural training and experiments. Local test receipts are available;
remote CI is not claimed. The separate kernel-only workflow contains no training.

Future learning requires explicit authorization, frozen data and gates, matched
controls and independent verification. Repeated seeds are not independent tasks.
Historical holdouts, raw predictions, thresholds and negative results remain intact.

## Research evidence and history

The current kernel supplements earlier work; it does not rewrite its conclusions.

- [V3 integration](docs/v3_integration/README.md), [mathematics](docs/v3_math/README.md)
  and [coding](docs/v3_coding/README.md).
- [V4 scientific report](docs/v4/scientific-report.md) and
  [V5 scientific report](docs/v5/scientific-report.md).
- [R5.1 verified readiness](docs/r5_1/verified-readiness.md),
  [learning-pilot handoff](docs/r5_1_pilot/handoff.md) and
  [current interpretation](experiments/r5_kernel/REPORT.md).
- [R5.2 post-hoc recovery](experiments/r5_2_math/recovery_v1/REPORT.md) and
  [R5.3 post-hoc recovery](experiments/r5_3_physics/recovery_v1/REPORT.md).
- Earlier [controlled-study results](docs/final_results.md),
  [research conclusions](docs/research_conclusions.md),
  [coding metrics](docs/final_metrics.md) and
  [Research v2 reproduction](docs/research_v2/reproduction.md).

Historical training commands and protocols remain in their versioned documentation.
They are not prerequisites for using the unified kernel and should not be rerun as
part of an integration smoke check.
