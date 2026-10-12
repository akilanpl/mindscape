# V3 handoff — 12 October 2026

Implemented a runnable, isolated V3 foundation with real CPU controls, independent
execution/evaluation, raw predictions, frozen task manifests, replay, safe resume,
partitioned memory and additive CI. No domain training or reasoning breakthrough
is claimed. This is a bounded foundation, with the explicit limitations below.

Repository base: `33c5b5657b2d19741b7b0ac84388d4f7997f9582`.
Development branch: `development/v3-research-foundation`.
Protocol/code freeze: `61687e8`; forward-accounting correction freeze: `99c09ee`. The manifest SHA-256 is
`edcc596f8a297562ff5d32ca03158da38b0948e51873f8d54a47121b9d7f4435`.
The current GitHub HEAD was checked by `git ls-remote`. The network clone was slow
because of archived evidence; implementation used a clean independent local clone
of the matching commit. No uncommitted source was copied into the branch.

## Inspection and preservation

Read the V1 numerical-claim protocol, final architecture/results/claims, V2 scientific
report/mechanism protocol/gap closure, coding architecture/audit, original typed
records, NumPy learned model, memory stores, planner, multiplication environment,
WASI sandbox and CI. V1's final numerical-claim results were weak (B IID 41.5%, OOD
6.11% at the largest tested budget), with unsupported memory/dream claims. These
must not be replaced by V2 masked-controller ceilings or V3 smoke outcomes.

V2 records 19,448 observed singleton legal-action selections. Zero-weight masked
controllers completed all 24 unique control tasks while permissive selection failed
all 24. V3 adds a direct software regression on three historical multiplication
cases: singleton-masked all-zero selection completes each; unrestricted all-zero
tie-break selection fails each. Historical source and checkpoint bytes are unchanged.
This demonstrates the specific controller confound, not a universal result.

`preservation.py` checked all 574 original tracked blobs against the base Git tree
and verified both existing release-tag refs. Receipt: `experiments/v3` evidence archive,
`preservation.json`. No old dataset, model, archive, protocol, source file, result,
website or release tag was modified. No new completion/release tag was created.

## Delivered code

- `src/mindscape/v3/schema.py`: typed JSON records and provenance for the full cycle.
- `kernel.py`: bounded proposal selection, separate neural hypothetical transitions,
  actual execution, independent verification, train-only episodic update and replay.
- `models.py`: standard-library MLP and optional approved local historical models.
- `memory.py`: structured partitioned retrieval, concepts, rules and durable JSON.
- `domains.py`: separate math, bounded Python repair and dynamics adapters.
- `experiment.py`: frozen splits, canonical identity checks, seeds, hashed checkpoints,
  exclusive writers, atomic traces, strict resume, raw predictions and Wilson intervals.
- `baselines.py`: same-backbone/information-budget comparison interface and matrix.
- `tests/v3`: schema, integration, numerical confound, leakage, sandbox-subset,
  memory, checkpoint, manifest, replay, resume and archive-security controls.
- `scripts/v3`: old-file preservation and safe checksum-verified evidence restoration.
- `.github/workflows/v3-foundation.yml`: additive standard-library CPU CI.

## Commands and observed validation

V3 works with standard-library Python. The observed smoke interpreter was
`/opt/homebrew/bin/python3.12` (3.12.14); full legacy tests used the existing local
`work/final-venv/bin/python` from the preserved checkout, which supplies pytest,
NumPy and Wasmtime. Resolved packages are saved in the evidence.

```sh
PYTHONPATH=src python -m unittest discover -s tests/v3 -v
python -m pytest -q -rs
python -m ruff check src/mindscape/v3 tests/v3 scripts/v3
python scripts/v3/preservation.py --output work/v3-preservation.json
PYTHONPATH=src python -m mindscape.v3.experiment --manifest configs/v3/smoke_manifest.json --output work/v3-smoke
PYTHONPATH=src python -m mindscape.v3.experiment --manifest configs/v3/smoke_manifest.json --output work/v3-smoke
PYTHONPATH=src python scripts/v3/restore_evidence.py --destination work/v3-restored
```

The first integrated suite passed 145 tests plus six subtests, with 20 optional-asset
skips. Existing local WASI/source dataset assets were then copied into ignored local
work/results directories. The WASM hash was checked against the preserved runtime
pin (`d24bd98d3071af6b17d51d53a08700b9acef59172a0afcb6adb733645c2a1715`).
No guest source executed on host Python. The subsequent suite passed 165 tests plus
six subtests with no skips; archive controls were then added; the final suite passed **168 tests plus six
subtests with zero skips/failures**, and all **29 V3 tests** passed. Scoped lint passes. A first collection attempt
failed due to a V1/V3 test filename collision; that was corrected without changing V1.
The standalone historical regression initially imported an optional NumPy module;
it now reads the frozen literal action vocabulary through safe AST parsing.

The initial saved smoke completed all 450 planned task/seed/condition keys, and the
second invocation resumed all 450. A separate test replaces neural forward with an
exception during resume and confirms no forward occurs. Independent restoration
hash-verifies every evidence member and replays all 450 executions with zero forwards.
Archive-corruption and traversal tests reject before writing a destination.

| Control | Mathematics | Python subset repair | Discrete dynamics |
|---|---:|---:|---:|
| Random weights, pooled descriptive counts | 0/75 | 0/75 | 1/75 |
| Zero weights, pooled descriptive counts | 0/75 | 0/75 | 0/75 |

These aggregate counts summarize executions; they are not independent-task statistical
samples. The 54 per-seed/domain/split cells and Wilson 95% intervals are retained in
`summary.json`. There are only 75 distinct evaluated tasks, reused across conditions
and seeds. Zero copies have identical weights despite seed metadata. Every action had
65 candidates. Incorrect numerical/repair proposals remained incorrect; no verifier
corrected a proposal or offered an inference retry. Predictions and actual transitions
are separate in every trace. These controls reject controller-supplied perfect success
for this implementation, not every possible source of leakage in future adapters.

Final episode wall latency sums to 0.176977 seconds; episode CPU sums to 0.176970 seconds.
They exclude setup and artifact writes. Cost paid: USD 0. No large model download,
training or paid inference occurred. Historical-model experiments remain unrun.

## Evidence and reproduction

`experiments/v3/cpu-smoke.tar.gz` includes the final 450 raw traces and six JSON checkpoints,
run identity, complete summaries, dependency/runtime evidence, preservation receipt
and test logs. It also preserves the initial 450-case run under `initial-cpu-smoke`.
The initial telemetry counted only the policy forward; final traces correctly count
one policy forward plus one neural transition forward (two total). The source fix
was committed before the final run. No weights or task outcomes changed.
`evidence_manifest.json` binds every member and the compressed archive.
`restore_evidence.py` rejects unsafe members, mismatched hashes, duplicate completed
keys and disagreement between raw outcomes and summary rows. No original research
archive was edited. The protocol commit predates the saved smoke run. Development
tests used generated task families before freeze; this is not an unseen scientific
lockbox or a preregistered confirmatory performance study.

## Incomplete or deliberately deferred

The kernel's initial adapters use one decision per episode. Partial observations have
typed representation but no partial-observation benchmark. Python is a capability-
bounded affine-repair subset, not arbitrary repository repair; the preserved WASI
system was regression-tested but is not a new general V3 adapter. Dynamics is discrete
and one-step, not a continuous simulation suite. Concepts/rules have storage and
provenance but no learned rule-induction algorithm. The MLP's transition head is
untrained; predictive accuracy is not established. Memory retrieval is logged, not
consumed by the smoke predictor, and its benefit is untested.

Training algorithms, optimizer-state resume and trained answer-only/structured/memory
baseline implementations are not part of this pretraining foundation. The fair budget
interface is implemented; declaring a future baseline is not an executed comparison.
Historical GPT-Neo/Pythia loading paths are prepared and fail-closed checks are tested;
no actual model was loaded, so tokenizer/generation compatibility and performance
remain unvalidated. Other historical model families require explicit approval and
implementation. Confidence is uncalibrated. Peak process memory, energy, GPU cost,
paired cluster-bootstrap comparisons and independent cross-domain replication were
not measured. CI is configured; remote execution status must be checked separately.

Exact/canonical split separation does not prove semantic independence or absence of
historical pretraining contamination. In-process domain plugins and caller-supplied
memory judgments are trusted code; these interfaces do not isolate hostile plugins.
Interrupted locks/temp files require inspection rather than unsafe automatic recovery.

## GO / NO-GO

**GO for bounded foundation integration and further engineering experiments.** The
information boundary, independent execution/evaluation, confound regressions, leakage
gates, local controls and replay/resume have executable evidence. All historical
artifacts are preserved.

**NO-GO for broad V3 completion claims, competitive reasoning claims or substantive
domain training yet.** First freeze a domain-specific training protocol and budgets,
implement the matched trained baselines and optimizer lifecycle, extend adapters to
the intended multi-step tasks, and independently review target/feedback boundaries.
No reported zero/random control licenses a learned-reasoning claim. No domain training
was started in this assignment.
