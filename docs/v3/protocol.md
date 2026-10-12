# V3 foundation smoke protocol

This is an engineering validation protocol, not a test of reasoning superiority or
sample efficiency. It freezes 135 tasks: each of three domains has 20 training,
5 validation, 10 IID and 10 OOD identities. Training is not performed. All 75
nontraining tasks run under random and zero weights for seeds 0, 1 and 2 (450
executions). Zero copies are a single effective control, not three independent
learners. Development unit tests already exercise these generated families;
there is no claim these constitute an untouched final scientific lockbox.

Each task has the same 65 syntactically acceptable choices (-32..32). Wrong choices
execute unchanged. No semantic correctness mask, target score, phase oracle,
private evaluation result or task split reaches the model. A singleton candidate
set is a hard error. Fixed one-decision episodes preclude verifier-guided retries.

Mathematics submits an integer sum. Python repairs one offset in a bounded affine
expression from an observed example, then a nonexecuting AST interpreter runs
three private input tests. Dynamics predicts position under the declared one-step
semi-implicit Euler rule. These are small domain adapters, not comprehensive math,
Python or physics benchmarks. Python supports only bounded integer expressions;
no task source runs through host Python. General Python belongs in the preserved
WASI capability sandbox and needs a separate adapter/protocol.

Addition identity canonicalizes operand order. All exact/canonical identities are
disjoint across partitions. This does not exclude abstract concept or solution-
structure overlap; such overlap is intrinsic to this smoke design. OOD changes
first-input magnitude, not domain semantics. Outputs fit the fixed vocabulary.

The shared MLP has 1,287 parameters: 7 inputs, 16 tanh hidden units, 65 action
logits, and a separate 17-input, three-output transition prediction head. Both
conditions share capacity and information. No optimizer, label or verifier-query
training is used. Predictions, logits, candidates, confidence (uncalibrated),
actual outcomes, transition error, provenance and timings are saved.

Wilson 95% intervals describe task proportions separately per seed/domain/split.
No repeated seeds are pooled as independent tasks. No hypothesis test or competitive
model claim is predeclared. CPU seconds and wall episode latency exclude file
serialization and setup. Raw results include all incorrect proposals; missing or
exceptional runs must not become fabricated failures. Resume binds source files,
manifest, checkpoints, seeds, conditions and Python/platform; existing traces are
hashed and deterministically replayed without neural forward calls.

No large model is downloaded and no paid service is invoked. Optional historical
models require explicitly approved local file manifests, matching architecture
families and safetensors. Their pretraining exposure is unknown and their outcomes
are unrun. The protocol/manifest commit precedes the saved benchmark execution.

Telemetry amendment: initial code counted policy and transition prediction as one call.
Commit `99c09ee` corrects this to two neural forwards per decision before the final
saved run. The initial run is retained separately; task manifest, weights and outcomes
are unchanged. These repeated development runs do not increase independent task count.
