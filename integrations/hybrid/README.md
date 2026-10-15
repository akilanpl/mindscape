# Optional detachable hybrid assistant — Candidate C

This standalone service is disabled unless explicitly launched. Native Mindscape imports, dependencies, architectures, training and evaluation are unchanged. It uses Qwen3-0.6B plus bounded tools; it is not the native architecture.

```sh
cd integrations/hybrid
./setup.sh  # independent venv and pinned official model download; needs network once
./start.sh  # local files only thereafter; CPU default, loopback port 8765
# Optional existing environment / model:
MINDSCAPE_PYTHON=/path/to/python MINDSCAPE_MODEL_PATH=/path/to/Qwen3-0.6B ./start.sh
# Deterministic backup:
MINDSCAPE_TOOLS_ONLY=1 ./start.sh
.venv/bin/python -m pytest -q tests
```

The official model revision is pinned in model-lock.json. Download script records SHA256 hashes. Weights, local environments, private checkpoints and evaluation secrets are excluded from Git. Preserve the model license from its official download. External R6 research source and artifacts may optionally be supplied using MINDSCAPE_R6_REPOSITORY and MINDSCAPE_R6_ARTIFACTS; absent mounts return unavailable. No native fitting occurs.

Historical B report: hybrid 31/36; pretrained 16/36; safe refusal 4/4; warm p95 3.46s; 147 tests; 13 accepted topics. Coulomb and natural-language SQL excluded. These are historical results, not a fresh benchmark or full Class 12 mastery. Original pre-adaptation source hashes are in candidate-b-source-identity.json. Portable changes only remove machine layout assumptions and private checkpoint copies.

Rollback: check out the recorded baseline commit in a separate worktree. Removal: delete integrations/hybrid; no native entrypoint or configuration needs changes. See tests/test_boundary.py for automated dependency and native import checks. Independent dependencies never alter the repository root pyproject.toml.

## Candidate C and preserved B

Baseline B: `78c5771490491a2d1dc58e7baebdaf2ee5efd22a`, remote branch `baseline/hybrid-candidate-b-20261015`. C improvements are separate commits; no native code changes. Historical B31/36 is not a matched comparison with new C cases. Frozen hard36: B14/36 vs C36/36 answers (C35/36 consistent worked responses before a separately corrected fraction formatter). Corrected release fresh12: B3/12 vs C12/12, with C12/12 consistent worked responses. These are small, targeted structured tasks, not full Class12 mastery. See docs for limits, timings, raw-evidence location and no-training provenance.

New bounded JSON contracts accepted directly in the question field:

- Math `simultaneous`: variables plus equal-count linear equations; `probability`: operation conditional (intersection/given) or bayes (priors/likelihoods/index); `linear_ode`: variable, constant coefficient, polynomial forcing, initial_x, initial_y. Expressions use restricted arithmetic/functions.
- Physics `field_vector`: 2D point and charges with q/position; `dc_network`: fixed_voltages, passive resistors with a/b/resistance, requested unknown node; `induction`: integer turns, flux_initial/flux_final per turn, duration; `photoelectric`: frequency and work_function, returns stopping potential above threshold. Quantities require `{value,unit}`. Supported conversions include uC/nC, cm/mm, ms/mWb/eV. Cartesian signs and the supplied flux normal are explicit.
- Coding `csv_summary`: inline text, numeric column, sum/count/mean; no host file access. Python remains a bounded AST subset; only named ZeroDivisionError/IndexError/ValueError/TypeError handlers without aliases are accepted. Unknown coding tasks are refused.
- SQL accepts literal SELECT plus explicit JSON tables before or after the query. Joins, aggregates/HAVING and read-only subqueries run in isolated SQLite with an authorizer and progress limits.

The inherited capabilities endpoint lists the original B families; this README/docs define the additional explicit-schema C scope. The neural advisory prompt/weights are unchanged. Displayed tool answers and worked text cannot be rewritten by a second model call. Some symbolic checks share the computation engine rather than constituting independent proof.

For automated callers, pause 150 ms after each response: the inherited loopback handler can briefly retain its request slot after sending the response. There are no automatic retries in the reported hard benchmark. No perf optimization is claimed; C adds tool work.

Model file verification, after the pinned download:

```sh
(cd models/Qwen3-0.6B && shasum -a 256 -c ../../model.SHA256SUMS)
```

Public diagnostic reproduction (not a fresh holdout):

```sh
.venv/bin/python benchmarks/evaluate.py benchmarks/diagnostic-36.json diagnostic-results.jsonl 8765 hybrid
```

The original grader is conservative for free-form pretrained prose; final metrics include separate full-response reference review. The calibrated numeric grader accepts exact finite fractions under the declared tolerance. Raw protected final questions stay outside the repository. Baseline and C results, demo, rollback and removal instructions are under docs/.
