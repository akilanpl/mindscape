# V5 prototype implemented; DATA_NO_GO before training

V5 implements a new action-conditioned graph transition generator: separate domain entity encoders/heads, a shared two-round relation/message/effect core, and a precondition predictor. It predicts numeric effects, rather than selecting a V4 candidate. This implementation is an untrained prototype. Native neural behavior, learning, generalization and transfer are **not measured**.

The complete model, comparison protocol and 132-task manifest were frozen in commit `118e2ca` before qualification execution. Three seeds and domain-only/shared/randomized-auxiliary controls were specified in advance, with fixed compute and success gates. Nine focused nonneural tests passed, including actual WASI execution, independent physics replay, graph masking, freshness/identity checks and gate rejection. No neural workload, large model training, Qwen generation, historical sealed holdout access or website changes occurred.

## Qualification result

**DATA_NO_GO.** Execution stopped on the coding qualification cohort after **88/132 tasks**. The remaining **8 physics qualification and 36 composition tasks were not executed**. No composition targets were generated or used for feasibility selection. Original protocol/gates are unchanged; no tasks were regenerated or thresholds revised.

| Cohort | Gate sample | Positive valid | Negative invalid | Changed valid | Persistence normalized MSE | Result |
|---|---:|---:|---:|---:|---:|---|
| Mathematics first training cohort | 12 | 8 | 4 | 8 | .1148833333 | PASS |
| Coding first training cohort | 12 | 8 | 4 | 8 | .0003777778 | PASS |
| Physics first training cohort | 12 | 8 | 4 | 8 | .0423141460 | PASS |
| Mathematics qualification | 8 | 6 | 2 | 6 | .1510562500 | PASS |
| Coding qualification | 8 | 6 | 2 | 6 | **.0004000000** | **FAIL** |
| Physics qualification | 0 | unobserved | unobserved | unobserved | unobserved | not tested |

The first 12 training tasks in each domain must have >=6 valid, >=3 invalid and >=6 changed-valid actions. Qualification requires >=4 valid, >=2 invalid and >=4 changed-valid actions plus **persistence normalized MSE >=.002**. Coding passes all count/change gates but fails the fixed MSE-headroom gate by a factor of five.

All 24 training tasks/domain were executed: each has 16 valid and 8 invalid. The final qualification cohorts add 6 valid/2 invalid each for mathematics and coding. Across the 88 executed tasks: **60 valid, 28 invalid**. These are action-precondition labels, not repair-candidate quality labels. Target effects are preserved per task/entity.

Coding literal increments of ±2/±4 yield small normalized effects after dividing by 100, and MSE averages unchanged leaves alongside changed literal/subtree values. The observed .0004 is limited headroom under this declared metric; it does **not** imply these tasks have no headroom under every correctness metric. The gate deliberately precedes learning; no learned performance was used to select tasks.

An additional deterministic design confound was identified post hoc: `invalid = i % 3 == 2` and coding operator selection `i % 3` associate every multiplication task with an invalid edit. Thus valid coding training/qualification transitions contain addition/subtraction only. This is not corrected retrospectively. It further limits any future interpretation of this dataset and is another reason not to train it. No revised dataset, extra seed or alternative cutoff was run.

## Verification and evidence

The independent review script re-executes the exact completed prefix using the register interpreter, actual WASI before/after subtree outputs, and pinned Pymunk impulses. It checks task bindings, public graph snapshots, target effects, exact gate arithmetic and first-failure stop. All 88 outcomes reproduce. The initial reviewer rejected Pymunk provenance tuples compared to JSON lists; canonical JSON comparison repaired this serialization check without changing any numerical effect or target. That initial reviewer failure is retained. Qualification files and protocol are preserved byte-for-byte. Original V3/V4 evidence and sources remain untouched.

`experiments/v5/qualification_v1`: all frozen inputs, public graph snapshots, executed targets, five observed gates and empty composition index. `experiments/v5/review_v1`: independent replay receipt and per-task valid/invalid/effect labels. `experiments/v5/release_v1`: original qualification archive, original source archive and restoration/hash manifest. The miniature demo prints actual saved graph/effect records and explicitly says `learned_prediction: null`.

## Executability and limits

Nonneural demonstration:

```zsh
PYTHONPATH=src /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python -m mindscape.v5.demo
```

The native MPS training launcher is blocked by DATA_NO_GO; **do not launch it**. No source-freeze/qualified training release was issued because Phase 2 failed. Native engineering tests for gradients, entity-order equivariance and action sensitivity are implemented but unexecuted. Neural checkpoint generation, native replay, parameter-transfer evidence and bitwise continuation are not claimed. There is no automatic retry/resume path.

The graph abstraction is computational dependency, not an assertion of equal semantics across domains. The physics microdomain is fully observed, one-step and contact-free; object-count composition does not establish learned interaction physics. The procedural executor is perfect by construction and provides an expert ceiling; qualification seeks headroom over persistence only. This scope is much narrower than V3 specialist tasks and is explicitly synthetic. Mathematical targets are register-state changes, not full arithmetic rule induction or digit/carry reasoning. Coding targets are executed subtree effects after a supplied edit, not autonomous program repair.

**Verdict:** engineering prototype implemented, data qualification failed, learning/transfer untested. Preserve DATA_NO_GO and stop this frozen experiment. No native training command is needed. A separate future experiment would require a new predeclared dataset/protocol and authorization; it cannot silently replace this result.
