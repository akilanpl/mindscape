# R5.2 Mission 1 — mathematical generalization, prepared native campaign

Engineering implemented; neural qualification/training/results **NOT RUN**. Historical V1–V5/R5.1 sources/evidence and Python coding are unchanged. This additive namespace is `mindscape.r5_2_math`.

## V3 diagnosis and new mechanism

V3's three-seed study trained63 checkpoints and replayed60,900 executions. At n1000 concepts26.89/31.33% OOD versus procedural17.56/22.44%; zero-weight network retaining demonstrations32.67%. That result cannot establish induced arithmetic rules. V3 training exposed373 exact local contexts; memory reuse and a supplied signed-magnitude/place-value scaffold accounted for substantial success. Unseen carry/borrow contexts, signs and length remained weak; its sign input truncated original numbers to8 digits. Integration preserved failed487+376 and73-29 outputs. Its top-two digit/carry alternatives could omit the correct local answer entirely.

R5.2 uses learned decimal embeddings, observable relational features, a shared48-unit GRU transition state, full10-way digit/carry heads and a separate learned sign head. There is no arithmetic sum/product input feature. Recurrent state resets at each explicitly scheduled arithmetic row; predicted incoming carry is executed uncorrected at inference. Full-vocabulary probabilities make missing correct labels visible via post-hoc ranks, without target-based masks or proposal repair. Sign inputs use all integer signs/magnitude comparisons/zeros rather than truncated digit slots.

The place-value cursor, signed-magnitude ordering, addition/subtraction primitive selection, grade-school multiplication rows and serialization remain supplied inductive biases. This is not discovery of the entire arithmetic algorithm. Matched reset-hidden control uses identical parameters, features, examples, loss and update counts but no retained temporal state; its access to previous inputs intentionally differs. Neural results never receive demonstration retrieval. Memory-only and zero-weight-plus-memory use exact training keys only; misses emit uncorrected zeros. No numerical rule synthesis is credited to retrieval.

## Frozen campaign

768 fresh signed1..3-digit training tasks,60 development tasks and144 OOD tasks (24/family): length4..8, carry/borrow chains4..11 digits, signs, compositions, larger operand ranges8..10 digits, and explicit held local labels. Six fits: recurrent/reset-hidden x seeds11/23/47,800 updates,batch32,Adam.003,final checkpoint only. No checkpoint selection,tuning or extra model training. Fixed signed add/sub/mul scheduling supports40 serialized output digits and4096 local writes. Division is a separate symbolic Euclidean pathway, not a neural achievement.

The held local add/sub/mul context (8,7,carry) receives no supervised loss or memory entry. Its public input and next-state teacher-forcing indirect exposure are counted/disclosed; this is a supervision holdout, not complete input blindness. OOD family labels can involve multiple shifts (length/digits/sign/carry), so they are not isolated causal factors. Inputs and context identities are freshly generated and checked against public historical task identities and opaque lockbox IDs; no historical lockbox is opened. Other historical datasets are not used as training.

Before training, the committed manifest is checked, all training/reference traces must reproduce exact arithmetic, each operation has>=200 supervised labels and both sign classes>=32. OOD reference local-context memory-miss fraction must be>=5%, so the benchmark cannot be entirely exact demonstration retrieval. This predeclared reference-based feasibility screen neither observes neural performance nor filters evaluation tasks. Any failure stops DATA_NO_GO. No historical holdout task is read to establish success.

Frozen neural GO requires mean exact OOD>=65%,length>=80%,carry/borrow>=60%,sign>=80%,held-local-context family>=50%,intermediate next-state accuracy>=90%,>=10 percentage points over reset and both memory controls,and positive reset gains for every seed. These are demanding pilot targets; failing any yields NO_GO. All three seeds reuse204 distinct evaluation tasks, not612 independent tasks. Symbolic cases never enter these metrics.

Records preserve probabilities, every local write, sign decisions, independent reference annotations, per-family/seed exact and strict accuracy, local failures, known/unknown training context diagnostics, losses, model+Adam states, checkpoint hashes and time. Saved claims independently replay through a fresh write-only rollout; verifier never repairs values. Numerical results and checkpoint tensor audits must be reviewed after native execution. No bitwise continuation or full general mathematical coverage is claimed.

## Modular capability contract

`Request(version='R5.2-math-1',family,inputs)` rejects private fields and oversized/malformed data. `solve` returns explicit `method`: `neural_with_declared_scaffold` for integer inference; `symbolic_tool` for bounded fractions, polynomial evaluation, nondegenerate linear equations,<=4x4 matrix multiplication,2D distance,polynomial differentiation/definite integration,population mean/variance,event-count probability,and Euclidean integer division. SymPy is already installed; no downloads. Fraction/reference arithmetic,residuals,coordinate/derivative/integral identities and transparent tool traces verify results. Tool correctness is not learned reasoning. This is a modular foundation for later kernel integration,not complete algebra/calculus/statistics or a drop-in replacement for the old kernel.

Twenty focused nonneural tests exercise signed reference rollouts,carry/borrow chains,multiplication,malformed/private inputs,no verifier corrections,held-label memory exclusion,deterministic split identities,and all nine symbolic families. The current neural architecture, qualification and MPS behavior remain untested until the single native command below. Existing V3 checkpoint65e822bd... is present and hash-bound for preservation,not retrained or counted as a new trained result.

Run once in normal macOS Terminal:

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/r5_2_math/native.command
```

The launcher requires MPS and forbids fallback/downloads. It performs a single bounded qualification,then at most6 fits,then recorded evaluation/independent replay and archive verification. Output is `results/r5_2_math/native-one-campaign`; existing directory blocks rerun. DATA_NO_GO/NO_GO/FAILED_CAMPAIGN ends this experiment. No additional training or public claim follows automatically. Return the final receipt for independent audit and a scientific report/PR completion.
