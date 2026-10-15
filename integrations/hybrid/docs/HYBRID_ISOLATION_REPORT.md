# Architecture isolation proof

Upstream: `main`, inspected HEAD `f9955c7b8043a00ebea36fffd41c646d1d54f2bd`. Native package lives under `src/mindscape`: core schemas and transitions, environments, reasoning, memory, verification, models, training and evaluation; later research components live under `v3`, with dedicated scripts, configs and archived evidence. Public native interfaces include `MultiplicationEnvironment`, `verify`, and `scripts/run_episode.py`. Root dependencies are empty, with optional research extras.

The extension is standalone `integrations/hybrid`, preserving its own package namespace `mindscape_delivery`, dependency manifests, configuration, launcher, API and frontend. No root dependency, native import, architecture, checkpoint format, training protocol, evaluation entrypoint or CI change is required. Optional external R6 demonstration imports flow from extension to explicitly mounted research source only; absent mounts return unavailable. It never supplies a CBSE answer attributed to native R6.

## Executed evidence

- Native unittest: 92 tests, one skip.
- Native pytest: 146 passed, 22 skipped, 6 subtests passed. Original optional/runtime skips are retained.
- Native `-S` test disables site packages, verifies 19*3=57 and asserts no hybrid, torch, transformers or SymPy module loaded.
- B: 147 original tests + 2 new import-boundary checks pass.
- C release: 173 tests pass, including integral domain preservation, unsafe-code rejection, task validation, singular circuits, physical units and routing regressions.
- Real enabled API smoke used the official pinned model plus tools, local CPU, zero optimizer updates.
- Disposable clean worktree: deleted only `integrations/hybrid`, ran the original pytest suite (146 passed / 22 skipped / 6 subtests), and original no-site-packages native entrypoint (verified 57). Primary development checkout remained intact.
- Automated AST scan finds no native `src`/`scripts` imports of the hybrid package. Git diff verifies all changed paths are confined to `integrations/hybrid/`; all tracked core blobs and research assets remain identical to upstream.

The checkout omits large archived research/dataset directories for transfer efficiency; v3 evidence fixtures required by tests were restored byte-for-byte from Git. No archive was removed from the repository. GPU/cloud paths requiring unavailable optional runtimes were not exercised.

## Changed files

- `integrations/hybrid/.gitignore`
- `integrations/hybrid/ARCHITECTURE.md`
- `integrations/hybrid/DEMO_5_MINUTES.md`
- `integrations/hybrid/README.md`
- `integrations/hybrid/benchmarks/diagnostic-36.json`
- `integrations/hybrid/benchmarks/evaluate.py`
- `integrations/hybrid/candidate-b-source-identity.json`
- `integrations/hybrid/docs/BASELINE.md`
- `integrations/hybrid/docs/HYBRID_BASELINE_RESULTS.md`
- `integrations/hybrid/docs/HYBRID_CANDIDATE_C_RESULTS.md`
- `integrations/hybrid/docs/HYBRID_FAILURE_ANALYSIS.md`
- `integrations/hybrid/docs/HYBRID_GITHUB_PUSH_REPORT.md`
- `integrations/hybrid/docs/HYBRID_HARD_CBSE_BENCHMARK.md`
- `integrations/hybrid/docs/HYBRID_ISOLATION_REPORT.md`
- `integrations/hybrid/docs/HYBRID_OCT21_DEMO.md`
- `integrations/hybrid/docs/HYBRID_REMOVAL_GUIDE.md`
- `integrations/hybrid/mindscape_delivery/__init__.py`
- `integrations/hybrid/mindscape_delivery/acceptance.py`
- `integrations/hybrid/mindscape_delivery/advanced.py`
- `integrations/hybrid/mindscape_delivery/assistant.py`
- `integrations/hybrid/mindscape_delivery/native_demo.py`
- `integrations/hybrid/mindscape_delivery/neural.py`
- `integrations/hybrid/mindscape_delivery/parsing.py`
- `integrations/hybrid/mindscape_delivery/server.py`
- `integrations/hybrid/mindscape_delivery/solvers.py`
- `integrations/hybrid/mindscape_delivery/tool_worker.py`
- `integrations/hybrid/model-lock.json`
- `integrations/hybrid/model.SHA256SUMS`
- `integrations/hybrid/requirements-dev.txt`
- `integrations/hybrid/requirements.lock.txt`
- `integrations/hybrid/requirements.txt`
- `integrations/hybrid/scripts/benchmark_neural.py`
- `integrations/hybrid/scripts/download_model.py`
- `integrations/hybrid/scripts/rehearse.py`
- `integrations/hybrid/setup.sh`
- `integrations/hybrid/start.sh`
- `integrations/hybrid/tests/test_acceptance.py`
- `integrations/hybrid/tests/test_application.py`
- `integrations/hybrid/tests/test_boundary.py`
- `integrations/hybrid/tests/test_candidate_c.py`
- `integrations/hybrid/tests/test_extended_contract.py`
- `integrations/hybrid/tests/test_native_demo.py`
- `integrations/hybrid/tests/test_neural_contract.py`
- `integrations/hybrid/tests/test_server.py`
- `integrations/hybrid/tests/test_solvers.py`
- `integrations/hybrid/web/app.js`
- `integrations/hybrid/web/index.html`
- `integrations/hybrid/web/math.js`
- `integrations/hybrid/web/science.js`
- `integrations/hybrid/web/style.css`

Final release removal at runtime commit a23954a: 146 passed, 22 skipped, 6 subtests; native no-site-packages entrypoint verified. Original two annotated release-tag objects match the archived values exactly. Runtime module hashes match the protected release freeze. The separately packaged benchmark harness is byte-identical to the calibrated harness used for hard/release predictions.
