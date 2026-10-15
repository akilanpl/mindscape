# Optional Candidate B hybrid assistant

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
