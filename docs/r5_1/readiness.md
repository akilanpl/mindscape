# R5.1 — original pre-execution readiness handoff

**Completed:** native run returned verified DATA_READY. See `verified-readiness.md`. Do not rerun the command below. The separately frozen learning pilot awaits authorization.

Status: frozen preparation, **native engineering and qualification NOT RUN**. V5 `851815f`, PR #9, original .002 gate and DATA_NO_GO remain unchanged.

Existing-evidence diagnosis: coding qualification has 6 valid changed tasks and 2 invalid no-ops, 3 active entities/task, 2 changed entities/valid task. Only 12/24 entity outputs change. Persistence MSE is .0004 across the graph versus .0008 over changed entities; .0004 is diluted by unchanged entities and invalid tasks. This is a post-hoc V5 diagnosis, not a changed V5 verdict. The factor of two follows the preserved outputs; scale 100 also makes ±2/±4 changes small. Recorded diagnosis: `experiments/r5_1/v5-metric-diagnosis.json`.

R5.1 defines changed-entity error from independently executed deltas: select an entity if any true normalized component exceeds 1e-8; score all its output components, average within task, then equally across tasks with changed entities. Preserve whole-graph macro and scalar-pooled MSE, unchanged-entity error, and valid/invalid counts. No-op tasks receive no invented zero changed score; their error remains in whole/unchanged metrics. The mask is post-execution evaluation metadata only, never inference input. A future supervised loss must be frozen separately before training. Reporting only changed-entity error would hide collateral changes; retaining other metrics is mandatory.

Baseline gates require headroom over both persistence and a public local-action heuristic, plus that heuristic improves persistence by >=2%. The heuristics are unit-source register updates, literal-only AST updates with no subtree propagation, and unit-mass velocity impulses with no position update. They require no private or future values. They are deliberately simple controls, not the strongest domain procedures; actual procedural execution is perfect and remains an explicit ceiling. DATA_READY would establish prerequisites for a learned test, not superiority or semantic transfer.

The new deterministic generator forms coding operator × valid/invalid Cartesian cells, then shuffles once with seed 20261016051. Development has 4 valid/2 invalid for every add/sub/mul; qualification has 2 valid/1 invalid each. New math values 51–89, AST leaves 11–19 with |x|>=7, and physics masses 1.4–2.0 are declared before execution. They define a new numerical-support study, not a filtered or enlarged V5 run. No task is selected by outcome or model performance; canonical overlap with the public V5 manifest is rejected without resampling. There is no historical sealed-data read.

Complete protocol and public dataset/scorer specification are committed before execution: `configs/r5_1/{protocol,tasks,source-freeze}.json`. The architecture, V5 feature widths (6/10/8), domain effect widths (1/1/4), padding 7, action width 8, and relation width 3 are unchanged. Encoder concatenation doubles raw feature widths to 12/20/16. Expected parameter total 6,967 is tested against actual native construction, not represented as already measured.

The single native command first tests actual MPS input/output shapes, effect and precondition backward paths, all expected active-domain/core gradients, absence of inactive-domain/head gradients, input/action/relation connectivity, exact masked-input isolation, action/relation sensitivity, entity-order equivariance and unchanged parameters. Synthetic probe losses perform backward only: no optimizer or parameter updates. Public composition graphs are shape/connectivity probes; no composition action or label is executed/read. These engineering values do not measure held-out performance and cannot qualify data.

After engineering PASS, execute at most 81 tasks (18 development +9 qualification per domain). First failed completed cohort stops before another task. Development requires 12 valid/6 invalid/>=10 changed-valid; qualification requires 6 valid/3 invalid/>=5 changed-valid. Both splits require changed-entity persistence MSE >=.0005, local-action MSE >=.0002 and relative local improvement >=.02. Coding balance and exact independent effect replay also must pass. Twenty-seven held-out compositions stay unexecuted. No replacement task, second seed, training, checkpoint or resume path exists.

Run **once**, in normal macOS Terminal:

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/r5_1/native.command
```

Output: fixed `results/r5_1/screen-<manifest hash>`, full gradients/shape receipts, executed labels and effects, all observed gate values, and a verified evidence archive. ENGINEERING_NO_GO stops without data execution; DATA_NO_GO stops without retries/training; DATA_READY stops and requires explicit authorization for any domain-only/shared/randomized-auxiliary learning pilot. The existing native MPS boundary prevents execution inside the sandbox.

Eight focused nonneural tests pass; source/metric and generator checks run without a neural forward or new execution target. The original V5/V4 source/evidence is preserved. A read-only post-run audit replays completed effects and exact gate calculations; it does not rerun neural tests or execute compositions. Native claims and final readiness decision remain pending.
