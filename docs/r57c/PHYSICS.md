# Independent physics architecture diagnostic

**PARTIAL; graph-advantage mechanism UNRESOLVED.** Read-only CPU replay of all
nine R57 initial/final checkpoints; zero optimizer constructions or updates. No
source, checkpoints, old final evaluation or official protected data changed/read.
The new twelve-state diagnostic grid varies mass .7/2.3/6.1, dt .2/.8 and signed
forces ±[1.7,-2.1], with observed velocity [.3,-.7]. It is explicitly a diagnostic,
not a new final benchmark or model selection result.

| Seed | Graph on MSE | Graph off MSE | GRU MSE |
|---|---:|---:|---:|
|17|.4788|.4989|.4634|
|29|.7868|.6724|.4098|
|43|.4722|.4454|.4564|

The graph-vs-GRU ranking is not stable on this grid. A universal graph advantage
is rejected. Valid node relabeling preserves graph outputs; GRU relabeling changes
outputs by maxima .4155/1.1308/.9495 across seeds, confirming serialization
sensitivity remains. This robustness property does not establish a beneficial
causal topology effect. Edge removal/reversal/relation shuffling alter represented
relations and are invalid as invariance tests; their outputs are retained only as
interventions.

At the first diagnostic state, trained on-model gate zeroing changes MSE
.0236→.0199 (17), .8657→.8972 (29), .3187→.2443 (43). Context interference is
therefore state/seed dependent, not a confirmed cause of aggregate gaps. Gate raw
and tanh values, gate/embedding/feature/physics-head gradient norms and ID
interventions are preserved. Off-model outputs stay exactly unchanged across
context-ID/gate interventions, as intended. Embedding normalization is part of the
trained architecture; no alternative weights are fitted here.

Every initial/final prediction, axis-isolated finite difference for force-x/y,
mass and dt, corresponding analytic difference, graph intervention and checkpoint
hash is in `evidence/physics/raw-paired.json`. Zero, training-only constant and
training-only ordinary least-squares baselines use only supplied training records;
the linear model sees separate raw force/mass/dt/velocity/intercept, never Fdt/m.
Analytic Fdt/m is an explicitly separate deterministic ceiling, not learned neural
intelligence. Neither analytic values nor diagnostic targets enter model features.

Three executable tests validate analytic transformations, unchanged checkpoint
parameters under gradient/gate probing, valid graph permutation and exact
concept-off suppression. No inference ablation is adopted as a model change;
controlled retraining would be necessary to establish a training mechanism.
