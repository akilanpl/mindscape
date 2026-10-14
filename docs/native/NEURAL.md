# Original R5.6 neural transition architecture

`R56Model(ModelConfig())` is an original PyTorch model. It has learned feature and
type encoders, typed sparse relation message passing shared across domains,
trainable concept and domain embeddings, action conditioning, bounded GRU working
memory, learned action scores, and separate digit/carry, coding candidate, and
physics delta heads. No pretrained LLM, rule engine, symbolic solver, or procedural
answer function appears in the forward pass. Construction does not establish a
learned capability. No optimizer updates or experiment campaign are performed.

The common API is `model(GraphInput, memory=None, stateless=False)`. GraphInput
contains node_features[N,8], node_types[N], edges[2,E], relation_types[E], action[4],
concept_ids[K], and integer domain (math=0, coding=1, physics=2). Nodes and edges
vary in count, subject to resource caps (default 256 nodes, 2048 edges). All tensors
must share the model's device. The model never silently chooses a fallback device.
Output keys are math_digit_logits[10], math_carry_logits[2], coding_logits[16],
physics_delta[2], action_scores[4], memory[width], pooled[width], node_states[N,width].

`pack_problem(problem, device)` encodes observed inputs without targets.
`infer_problem(model, problem, stateless=False)` returns prediction and separate
post-hoc verification. `model.predict(problem)` is a convenience wrapper.

- Math: domain math, operation add_digit or borrow_digit, left/right digits,
  incoming carry/borrow 0/1 (carry_in alias supported). The learned heads predict
  digit and outgoing binary carry/borrow. Operation add accepts unsigned decimal
  strings left/right through 128 digits and uses supplied LSD traversal, padding,
  stopping and output serialization. Every subsequent transition receives the
  model's own previous carry. No verifier corrections or answer masks intervene.
- Coding: domain coding, source Python (16KB / 256 AST-node budget), old_name and
  1–16 identifier candidates, and optional requested_name task instruction. The
  requested identifier is encoded as a separate observed node; target_name remains
  verifier-only. The model selects a candidate class; deterministic
  AST token rename applies that selected action. Optional target_name belongs only
  to the verifier and never to observation features. Renaming Name and argument
  tokens is a bounded action primitive, not scope-safe refactoring, execution,
  behavioral-equivalence validation or autonomous software engineering.
- Physics: domain physics, velocity [vx,vy], mass, force [fx,fy], dt and optional
  impulse [jx,jy]. The learned head predicts velocity delta; the controller adds
  that prediction to the observed velocity. The verifier separately compares
  against Newtonian impulse/constant-force dynamics. No exact dynamics overwrite
  model predictions. This is a local transition task, not a multibody simulator.

Caller-supplied concept_ids (0–31, at most 16) select trainable embeddings. The
labels themselves are explicitly supplied symbolic structure, not learned concept
discovery. Missing IDs produce a zero concept vector. Domain/type/relation labels,
encodings, action schedule, and heads are engineering priors. Learned action_scores
are distinct from deterministic argmax selection and domain traversal. The native
inference wrapper executes the requested transition; scoring a different action is
not yet an autonomous planning policy.

Working memory is a fixed-width GRU vector bounded by tanh. It carries no verifier
judgments, target labels or growing episodic history. The stateless control ignores
previous memory and removes state fusion from output prediction. Permuting concept
IDs, deleting relations, zeroing weights, resetting state and domain holdouts are
meaningful future controls, not experiments claimed here.

V5 supplied relation/action processing but fixed padded observation sizes and
backend-specific tensor construction. R5.4 showed local arithmetic competence and
full composed answer correctness must be measured separately, and historical
R5.2 recurrence performed worse than state reset. This implementation therefore
supports variable graphs and an explicit stateless control and preserves incorrect
model digits/carries throughout whole-addition inference. It does not claim that
memory, graph structure or shared parameters improve generalization without a
frozen, authorized and independently scored learning study.

Engineering tests check forward/backward connectivity, variable graph sizes,
relation/action/concept sensitivity, bounded memory, stateless isolation, label
nonleakage and deliberate wrong-output preservation. They perform no optimizer
construction, fitting, checkpoint selection or benchmark performance claims.
