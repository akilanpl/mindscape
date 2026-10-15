# Agent B — exploratory failure diagnosis

Status: PARTIAL mechanism; no official scientific qualification or production claim.
Frozen R5.6 sources/checkpoints/evaluators were not modified. Only public exploratory
training and old exploratory validation files were read. New final data was generated
before candidate scoring and placed under evaluator-only; it was not used to tune.

## Data evidence

`r57.data.prepare(output, legacy_root)` copies the original128 training records per
seed byte-for-byte. Math exposures43 have distinct identities36/35/38 for17/29/43;
coding43, physics42. New normalized IDs ignore concept/meta fields and canonicalize
incoming/carry aliases. All original train and old exploratory validation identities
are excluded from new development and final sets; final also excludes development.

New development per seed:240 rows,80/domain; math8 of each digit,40 each carry,
40 each operation. Coding has20 each correct candidate position with cyclic
permutation families and split-specific identifier names. Final has80/domain,
math40 each operation/carry and6–9 per digit (finite remaining feasible contexts),
coding20 each position. Exact balance in every joint math stratum is constrained by
the400 finite operand/operation/incoming states; no inflated unique counts are claimed.
Math unseen combinations are represented; development/final coding include several
AST templates, fresh names and candidate permutations. These are narrow synthetic
rename-selection tasks, not broad programming semantics.

Manifest: `outputs/r57-neural-improvement-v1/data-v1/manifest.json`, identity
`08b7814d78fb61247164b578318fe09cb6a4b30ce19b48136f1bc425fa787546`.
Final datasets are evaluator-only by convention, not a claimed OS isolation boundary.

## Actual original seed17 initial/final audit

All128 train examples received individual loss/backward measurements, no optimizer
updates in the audit. Gradients are cleared per example. Margin is true digit logit
minus highest rival (math) or requested candidate logit minus highest rival (coding).

| Domain | Mean loss initial→trained | Mean gradient L2 initial→trained | Mean target margin initial→trained |
|---|---|---|---|
| Math |3.11357→2.57265|5.96463→6.04448|−.52447→−.54140|
| Coding |1.41467→1.34776|4.53160→3.52287|−.17461→−.18773|
| Physics |1.63764→.82737|5.50890→3.14063|not a classification margin|

Per-example raw margins, correctness flags, parameter gradient norms and identities
are in `failure-audit-seed17.json`; hard/easy grouping can be reproduced from the
correctness flag. This does not support an initial physics-gradient-domination
explanation: math mean gradient is larger. Heterogeneous loss conditioning remains
possible, but raw loss magnitude alone is insufficient causal evidence.

Changing only requested candidate on one actual training problem gives:
encoded feature deltaL2=.95109; trained node states=.80375; pooled=.03097;
coding logits=.002730. Initial coding logits delta=.003081. The requested signal
exists at input but is attenuated through pooled/head stages. The fixed-index
head competes without candidate-conditioned scoring; combined with position
imbalance, constant candidate predictions are an accessible solution. This
supports testing a learned query/candidate pointer and equivariant representation;
it does not prove mean pooling is the only cause across all examples.

## Explicit bounded CPU optimization probe

Two strategies, same seed17 original128 records,120 updates each: **240 total CPU
engineering optimizer updates**, separate from native campaign fits. No protected
or new final evaluation was read. The balanced-normalized variant cycles domains
and divides loss by frozen initial eight-example per-domain mean. First12 train
examples, selected before running, give:

| Strategy | Math loss final | Coding loss final | Physics loss final |
|---|---:|---:|---:|
| Legacy order |2.4454|1.5997|.0942|
| Balanced normalized |2.4512|1.5366|.1200|

Initial losses on these12 were3.1623/1.4003/.2640. Normalization modestly reduces
coding degradation versus legacy order but worsens physics and slightly math;
**REJECT as an established across-domain improvement**. This tiny train-only
probe is not a generalization comparison and is not hyperparameter selection.

## Recommendations delivered early to model agent

Retain exact128 matched-data baseline. Test learned operand-sensitive math heads
and operation-aware representations, no analytic digit outputs. Test candidate
query scoring with permutation/alpha-renaming invariance, exact equality baseline
separate. Request signal attenuation supports this change more directly than a
claim of broken labels or absent gradients. Audit final improvements on identical
new development examples before untouched final evaluation. No significance or
cross-domain transfer conclusion follows from these diagnostics.

Reproduce tests: `PYTHONPATH=src ../py313/bin/python -m pytest tests_r57/test_data.py tests_r57/test_failure_audit.py -q`.
Three tests passed. Audit functions have zero optimizer updates; the explicit
sampling probe reports its240 updates independently. Historical9 fits remain unchanged.
