# Agent A: paired mechanism diagnosis

Status **PARTIAL / MECHANISM_NOT_ESTABLISHED**. Read-only CPU replay of all nine
immutable exploratory checkpoints. Zero optimizer constructions/updates. Original
44,962-parameter architecture and checkpoint bytes remain untouched. Raw paired
initial/final predictions, reference derivatives, graph interventions, component
norms and checkpoint SHA256 appear in `evidence/agent-A/paired-predictions.json`.

## Actual dataflow and comparability

Graph: three physics nodes × eight scalar slots -> Linear(8,48), typed embedding,
action Linear(4,48), domain and mean concept embedding -> tanh -> two relation
message rounds. Four physics edges produce [E,192] concatenated source, receiver,
relation and context features -> [E,48] messages -> incoming incidence reduction
and degree normalization -> residual update/LayerNorm [N,48] -> mean pooling[48].
Reset GRUCell(48,48) -> tanh bounded memory[48]; pooled+memory fusion[96->48] ->
physics head[2], digit[10], carry[2], candidate[16], action[4]. The audit executes
stateless=False/memory=None exactly as the exploratory runner. This reset policy
uses full learned GRU/fusion paths but tests no across-query episodic memory.

Concept-off keeps exactly the same parameters and common initialization but zeros
the mean concept route; unused concept parameters do not receive that loss gradient.
Concept labels duplicate existing operation/domain. Same-weight duplicate concept
IDs produce identical means/output: duplicating a single ID is not added information.
Concept identity and embedding norm both change the context that enters *each*
node and *each* message. Context norm and per-component loss gradients are recorded;
there is no preexisting learned concept gate.

Matched GRU serializes each node and edge and a terminal context token into 77-wide
observed tokens. Node types/relations are one-hot, endpoint indices normalized by
256, action/domain/concept information explicitly retained. It has approximately
matched parameter capacity but a different algorithm and sequence length: the last
hidden vector rather than permutation-invariant pooling drives its heads. The
same information is not equally usable; graph endpoint interpretation and terminal
context retention have to be learned sequentially. Parameter counts and storage
are measured in the machine report. No equivalence of FLOPs or effective paths is
claimed. Both models use the same task references and independently decoded outputs.

## Actual paired results on six new controlled physics states

The probe varies mass 1/2/4 and dt .25/1, fixed force [2,-3], zero velocity.
This is a small diagnostic grid, not the original validation or a held-out score.

| Seed | On MSE | Off MSE | GRU MSE | GRU valid-relabel max change |
|---|---:|---:|---:|---:|
|17|1.1726|0.4309|0.5874|0.377659|
|29|0.6693|0.8985|0.9717|0.582350|
|43|1.3426|0.6730|0.8088|1.071706|

**SUPPORTED:** valid node-order permutation with corresponding edge endpoint
remapping preserves both graph conditions to numerical tolerance. The same valid
physical relabeling substantially changes GRU outputs. This identifies a real
serialization robustness difference, not proof that topology causes the aggregate
physics MSE gap. Removing edges, reversing edges and shuffling relations also have
paired outputs; these change represented relations and are interventions, not
valid invariance tests. They cannot alone establish a beneficial graph mechanism.

**SUPPORTED:** finite differences of force-x/force-y, mass and dt deviate from the
analytic reference derivatives. Sign reversals and all axes are preserved raw.
This demonstrates incomplete physical sensitivity even when MSE improves; neural
outputs are not replaced by the exact F*dt/m reference.

**PARTIAL:** at the first diagnostic state, zeroing the trained on-model concept
route changes its MSE from .2683 to .0663 (seed17), .0925 to .0900 (29), and 2.6138
to 1.6686 (43). Other supplied IDs also change performance. This supports context
route interference in these specific fixed checkpoints, not a causal explanation
of the independently trained off/on mean gap. Seed29 reverses the on/off ranking
on the six-state grid. Unit-norm-context interventions are included without fitting.

**UNRESOLVED:** optimization, random initialization effects and redundant context
scaling cannot be separated into a unique cause using read-only checkpoints.
No controlled counterfactual retraining was run. Equivalent pooled/linear learned
baseline fitting is deferred; analytic linear-in-force reference derivatives are
only a transparent task ceiling and not an equivalent learned comparator.

## Evidence-supported next experiments

A near-zero initialized, normalized residual concept gate is a reasonable *test*
that preserves the existing off path while letting supervision adjust context.
It is not proven to repair generalization. Match numeric and gate norms and retain
shared initial common weights. For GRU, explicitly test relabeling/permutation
robustness and access to numerical tokens before attributing gaps to lack of graph
structure. Keep force-axis derivatives and sign/mass/dt checks beside aggregate MSE.
No training-gap mechanism, transfer, significant result or production acceptance
has been established. Four executable tests pass; CPU/GPU optimizer updates: zero.
