# Limitations

The multiplication scaffold, cursor progression, terminal-state construction and digit/carry mechanics are hand defined. Models learn numerical claims within that scaffold. They do not discover long multiplication or demonstrate general-purpose reasoning. The core interface is reusable, but the current state schema contains multiplication fields; no second learned-domain transfer is established.

The final transformer is frozen. Its indexed numerical feature text differs from ordinary instruction prompting, and categorical features accompany its hidden representation. Improvements over the historical MLP cannot isolate pretraining, representation, capacity, loss masking or update budget. Its135M parameters make the final model larger, not necessarily a stronger arithmetic model; measured results must decide. All final conditions share its capacity.

Answer labels, trajectory labels and scalar experiential feedback carry different amounts of information. Each B problem can contribute several supervised rows; C uses up to64 feedback attempts/problem and may not complete an episode. Dataset budget is not label, token or compute budget. The matched local supervised policy controls information for a wrapper comparison. Data-efficiency ratios against answer-only learners cannot alone establish architecture efficiency.

Groundedness and goal-success metrics require emitted verified trajectories. Answer-only outputs therefore score0 for those metrics even when numerically correct; this is a protocol constraint, not evidence of intrinsic unreliability. Trajectory validity is null for absent trajectories. No-environment recurrence loses real evidence by definition and is not an isolated arithmetic capability effect. The no-goal intervention removes a constant model-visible flag; external evaluator correctness remains mandatory.

No-episodic-memory control uses the same accepted rows, order and update count in memory rather than retrieving SQLite replay. It tests this storage/retrieval mechanism, not all possible benefits of episodic reuse. Concept descriptions are stored but no measured concept-learning contribution is claimed. Dream uses proposed-state dynamics and confidence, not a trained world model. A small generic goal demo demonstrates simulation interfaces only.

Nonzero1x1 canonical pairs are exhausted in training; IID holdouts therefore comprise larger seen structures. Structural OOD holds out3x2/3x3/4x2; carry/structure novelty is analyzed posthoc, not every possible carry pattern pre-reserved. Eight output digits and six input digits bound this adapter. The signed/zero-capable environment is broader than the final nonzero positive training distribution.

Only three training seeds, one generated dataset and one domain are studied. Mean±sample SD measures seed variation; Wilson intervals describe conditional per-run example uncertainty and do not remove shared-test dependence. One-seed McNemar tests are exploratory, with Holm adjustment, not cross-seed proof. Absolute and practical performance matter.

CPU inference uses a target-free shared frozen-feature cache. Reported latencies are warm/cached research measurements; first misses depend on run order. Shared transformer work and actual tokens are reported separately. They do not establish cold latency or end-to-end compute superiority. Full fine-tuning, larger budgets and new domains were deferred rather than expanding scope to chase success.

Historical GPT-4/Gemini GSM8K references use different tasks, prompting and training data. Direct comparison is unsuitable. No paid service was used. Original failures and configs are retained.
