# OOD test ledger

Status: **NONTRAINING_TESTS_PASS**. Scientific status **UNRUN**. Numerical specification **PROPOSED_REQUIRES_REVIEW**; data **DATA_UNQUALIFIED**.

Validation command: existing final-venv Python, `PYTHONPATH=src python -m pytest tests/alpha_ood -q`. Result: **58 passed**, including34 new numerical/custody tests and24 preserved OOD tests. Complete output is archived by the orchestrator under `work/sprint-evidence/c/pytest.log`.

New tests exercise actual CPU forward/backward for both learned architectures, finite connected gradients for every model parameter, unchanged parameter bytes after backward, common initialization parity, exact800×32 deterministic task schedules, train-only packaging, absence of reference access from inference, signature/content/purpose failure cases, no authorization at fit entry, full ordered seal coverage and overwrite rejection, synthetic predicted-feedback replay and corruption, miniature metric replay with independent6-task denominators, release sequence failure and raw seal substitution, public dataset count rejection and strict production216-task boundary. Optimizer behavior is inspected statically; no test calls `optimizer.step`. Tiny fabricated training fixtures and ephemeral test signing keys never touch protected benchmark data or historical locks.

The executable candidate includes future Adam/final-checkpoint/campaign paths. Those native paths were **not executed**, because neither human scientific authorization nor external qualification exists. MPS placement and sustained numerical execution are not established by these tests. Static source inventory and preservation verification are supplied to the independent orchestrator for reproduction and review. This ledger is preparation evidence, not scientific results or an approval receipt.

Four additional nontraining tests check mandatory source omission/changed hashes, source traversal/symlink rejection, exact runtime version mismatch/native guard rejection and static ordering of source/runtime validation before protected construction. The committed runtime snapshot is not native qualification or authorization.

Fourteen additional synthetic audit-schema cases reject empty/partial/truthy gates, inconsistent scientific status/counts, missing condition/seed/family/comparisons, wrong denominators, nonfinite metrics and corrupt/reordered raw report hashes. A synthetic monkeypatched signature/boundary diagnostic tests complete pass/fail verdict logic without issuing real approvals or native work.

A fabricated committed-but-incorrect label trace is rejected against the independently executed postrelease procedural interpreter; no actual OOD labels or public tasks were generated or scored.
