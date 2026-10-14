# R5.6 architecture and evaluation preparation

**EXPERIMENTAL — GENERALIZATION NOT YET DEMONSTRATED**

R5.6 is the requested next architecture name. This checkout contains R5.4 modules;
we have not established a genuine independently qualified R5.5 release. The name
is a project designation and does not imply an intervening scientific milestone.
Historical studies, R5.3 NO_GO evidence, websites and experiment workloads are unchanged.

Implemented interfaces: typed scalar state and action schemas, immutable ordered
state encoding, concept registry, preconditions, separate postcondition verifier,
budgeted composition, original prediction provenance, optional **untrained**
PyTorch shared transition candidate, prediction summarizer, content-addressed
experiment protocol preparation. Imports and deterministic paths need only Python
3.11+. PyTorch construction is optional and has not been tested in this environment.

Working adapters are deliberately bounded:

| Domain | Transition | Independent check | Limit |
|---|---|---|---|
| Math | radix carry decomposition | reconstruction and remainder bounds | integer total 0–1,000,000, base 2–36 |
| Coding | rename Names in one arithmetic assignment | inverse rename AST equivalence | no functions, scopes, calls, execution, or general refactoring claim |
| Physics | one-dimensional impulse | momentum difference | finite scalar SI inputs; fixed positive mass, no collisions or simulator claim |

Shared abstract labels describe structural relationships; they do not identify
carry, Python binding, and momentum as equivalent domain operations. Units for
physics are a documented SI contract, not a dimensional type system. The state
encoder uses explicitly ordered numeric fields; source text needs a separately
designed encoder. Candidate neural initialization has no trained weights or
performance claim. Registry and typed transitions are deterministic scaffolding.

## Falsifiable limited-data comparison

Hypothesis: concept guidance improves held-out transition accuracy with matched
training exposure and compute. Null: improvement is absent or attributable to
exposure, parameters, retrieval, or deterministic correction.

Before any protected run, freeze real dataset SHA256 identities, per-example split
identities, hypotheses, seeds, train sizes, optimizer and batch budgets, model
parameter counts, hardware, metric implementation, evaluation code revision,
bootstrap implementation and multiple-comparison family. `Protocol.freeze()`
creates a content-addressed **preparation manifest**, not scientific authorization
or a complete run-ready preregistration. No actual dataset was synthesized or
training run approved by this implementation.

Use paired examples and equal exposure, optimizer updates, parameter/compute
budgets and selection opportunities for neural-without-concepts and guided
neural models. Include deterministic rules, shuffled labels, removed concepts,
and matched-exposure controls. R5.4 and V5 shared graph comparisons require
explicit compatibility analysis; skip incompatible tasks with reasons instead of
inventing comparable scores. Freeze untouched value holdouts, composition depths,
and domain holdouts before training. Fit normalizers and retrieval only on train.
Keep private answers/verifier context out of predictors and external providers.

Report raw original prediction accuracy separately from verified/accepted
fraction, abstention and coverage. Never replace predictions with rule outputs
before scoring. Report composition success and error by depth, transfer,
robustness under frozen perturbations, samples to a preregistered target,
wall time and peak memory. Aggregate across frozen seeds; paired bootstrap 95%
intervals and Holm correction are the proposed tests. Define resampling unit
(problem family for dependent samples) and power analysis before approval.
Negative results remain evidence. Deterministic verification is no substitute for
an independent neural comparison. The original R5.4/OOD/cross-domain full workloads
and approval obligations remain in force.

## Reproduction and current evidence

Run `python -m pytest tests_r56 -q` in an environment with pytest and Python >=3.11.
The tests include 3,500 bounded carry cases, mixed-sign impulse composition,
AST rename/capture rejection, malformed predictions, typed rejection, domain and
budget boundaries, provenance retention and manifest identity checks.

Current verification establishes only deterministic bounded interface behavior.
It does not demonstrate data efficiency, unseen composition generalization,
transfer, expert proof validation or trained model correctness. `evaluate_predictions`
reports verifier acceptance only and explicitly carries this limitation.
