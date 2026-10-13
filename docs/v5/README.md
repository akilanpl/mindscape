# V5 development handoff

**DATA_NO_GO — no training.** See [scientific-report.md](scientific-report.md).

The action-conditioned graph model and matched experiment are implemented and frozen. Coding qualification failed the original persistence-headroom gate (.0004 < .002); the run stopped after 88 tasks. Physics qualification and all 36 composition tasks remain unexecuted. No training release or new dataset is authorized by this failed protocol.

- `architecture.md`: abstraction, model and scope.
- `configs/v5/{protocol,tasks}.json`: original committed comparison/data freeze.
- `experiments/v5/qualification_v1`: original inputs, actual execution targets and DATA_NO_GO.
- `experiments/v5/review_v1`: independent replay, labels and explicit nonneural demonstration.
- `experiments/v5/release_v1`: qualification/source archives and verified per-file preservation.
- `scripts/v5/review_qualification.py`: read-only repeat verification, using existing WASI and Pymunk.

The native launcher rejects DATA_NO_GO before touching MPS or constructing a model. Native gradient/equivariance checks and all learned results remain untested. Nine CPU nonneural tests, Ruff, Python compilation and launcher syntax pass. A reviewer list/tuple serialization mismatch was fixed using canonical JSON equality; exact numeric outcomes were unchanged, and the initial reviewer failure is preserved.

Do not run training or generate replacement tasks. No website changes were made. Historical V3/V4 verdicts and sources are preserved.
