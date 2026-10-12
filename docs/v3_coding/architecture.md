# Coding-learning architecture and boundaries

The new namespace is `mindscape.v3.coding_learning`; original `mindscape.coding`, its LoRA adapters and its WASI implementation remain unchanged. Shared V3 `Concept`, `Relationship`, canonical serialization and digest types represent observations and provenance. A domain-specific multi-round loop extends the kernel's observe/represent/propose/predict/act/verify/update semantics; it does not force repository replacements into the numeric kernel's integer action vocabulary.

| Component | Model-visible evidence | Authority and limits |
|---|---|---|
| Public task / contract | Broken source, entry point, intended behavior, input/output types, invariants, visible cases | No private cases or reference patch |
| Static concepts | AST nodes, lexical definitions/uses, assignment dependencies, branch/loop control edges, imports, boundary/mutation/order hypotheses | Syntactic, approximate; not semantic correctness, complete CFG, alias analysis or a type checker |
| Neural proposal model | Qwen prompt, prior visible failures, approved training examples | Produces candidate source and a brief hypothesis; no host execution |
| Verified memory | Training-only before/after source, contract, static concepts, actual outcomes and hashes | Admission re-executes visible and approved training-private suites; load distrusts stored success flags and re-executes |
| Structured retrieval | AST/operator/control counts and contract-invariant indicators | Cosine similarity rather than specification-word overlap; not an abstract rule engine or full graph matching |
| Learned evaluator | Before/after/delta features, syntax/path preconditions, edit magnitude and similarity to training repairs | Predicts visible usefulness and measured execution cost; private success always unknown |
| Bounded planner | At most two model-proposed alternatives, learned usefulness minus cost penalty | No private execution, hidden patches or oracle branches; no-planning selects candidate zero |
| Executor | Selected complete existing-file replacements | Uses unchanged pinned WASI guest; does not correct the repair |
| Visible verifier | Actual outputs, errors/stack traces and post-call input observations | Host compares value, type and specified mutation invariants; success can be false on private tests |
| Private scorer | Final repository and private suite after learner termination | Outside agent/model inputs, separately compares outcomes; first-edit audit is also terminal-only |
| Working update | Real visible outcomes and parse failures within episode | Persistent memory never changes during evaluation |

Structured retrieval is intentionally modest: count-based relational summaries and invariant indicators. Detailed graphs remain inspectable in traces, but no claim of deep semantic graph matching is made. The evaluator's candidate-after similarity uses historical repairs, while retrieval queries the current bug structure. Positive expert repairs and simple negative candidates differ from actual Qwen candidate distributions; strong training fit would not prove calibrated repair planning.

The action schema replaces up to four existing Python files atomically. Absolute/traversal/new paths and source-size violations are rejected. A conservative AST capability policy rejects external imports and reflective execution primitives. This policy is a bounded benchmark restriction, not a complete security proof. WASI remains the host isolation boundary: no model source enters host `exec`, `eval`, a Python import, or a host subprocess command. Actual VM wall time, fuel, memory, filesystem and output limits are inherited unchanged. Generated code may access task files and read-only guest stdlib, not host secrets or network/process capabilities. The source and compiled historical runtime are pinned and verified immutable.

The trusted guest harness never contains private expected outputs. Independent host comparison uses typed outputs; mutation invariants use a guest-produced post-call argument snapshot. This is research evaluation for bounded pure Python and is not a proof against every malicious guest attempting to falsify observations. Capability probes and transport rejection tests are distinct from correctness evidence.

Resumption binds dataset/protocol, model file hashes, training/checkpoint files, source and raw traces. Scalar and batched CPU runs retain separate identities/caches. Exact repeated requests can share actual generation; all logical calls and fresh work are recorded separately. Greedy Qwen seeds are fixed; only the learned evaluator has independent training initializations. Independent replay re-executes selected source inside WASI and re-scores visible/private results without loading Qwen.

## Additive safety policy

Final review reproduced indirect `__builtins__` access inside the WASI guest and nested bool/int equality in the original study comparator. `mindscape.v3.coding_safety.GuardedExecutor` adds rejection of reflective dunder identifiers/definitions and recursive JSON-domain type checks for outputs and mutation snapshots. New experiments should use this versioned guard. It does not alter frozen study source, prompts, weights or stored scores. `verify_guarded.py` independently replays the frozen episodes with this stricter verifier and must report any changed outcome. WASI isolation remains essential; static gating is not a security proof.
