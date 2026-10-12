# V3.1 stability candidate and pilot protocol

This is an additive research branch. V1/V2/V3 sources, frozen datasets, results, archives and tags are preserved. The immediate scope is one small development pilot, not another full study.

## Diagnosis before training

The V3 training-only receipt shows one active object in every motion training trajectory. Therefore the six relational inputs were zero throughout motion training and their scales were clamped to0.1. Relational input weights receive no informative gradient under that condition. Introducing a second object activates untrained channels. This is an identified support gap, not a conclusion obtained by tuning on the old final OOD holdout.

V3 also uses unbounded finite differences divided by dt=1/30, one-step teacher-observed training inputs, and recursive predictions during evaluation. Training acceleration features reached412.20 in motion and152.28 in interaction. Recursive velocity errors can be amplified into acceleration features, and inputs can leave the normalization range. The additional slot-flattened memory is not permutation invariant in general. These are concrete failure mechanisms/hypotheses; the native pilot measures V3 validation rollouts with concepts, memory, history and relationships removed. It never evaluates the old IID/OOD arrays.

## Candidate

Each object has a shared54-dimensional public encoding: current/previous observed state, visibility or predicted-status masks, known action and context, and six optional derived concept channels. Hidden mass/material values are ignored when their mask is zero. Inputs are bounded with tanh; shared pairwise messages exclude inactive objects and self-edges. Candidate edges use differences between normalized object features. The matched model uses the same shared pair network and all the same raw object information, without the explicit relative transform. Both have identical parameter dimensions, labels, batches, updates and kinematic prior; the transform is invertible given the source feature.

The neural head predicts bounded velocity increments, uncertainty and contact probability. Position advances using dt times predicted velocity. This explicit integration prior and increment bound6 are engineering assistance, not a learned physical law or simulator answer. They are identical in the candidate and baseline. The bound can bias very fast impacts; independent accuracy and physical-consistency gates prevent finite but incorrect outputs from being credited as success. Pairwise permutation consistency does not itself enforce momentum conservation.

Training backpropagates through eight free-running predicted steps without injecting future observations. Targets and contact labels belong to approved training trajectories and are used only in the loss. Masks distinguish observed(1), absent(0) and predicted(2) dynamic quantities. Evaluation uses the same free-run function as training. No simulator is imported by the predictive model/planner or queried during hypothetical selection.

Raw history removal, bounded concepts and object-local training-only residual memory are separate conditions. Memory is derived from the exact candidate weights with zero extra gradient updates; retrieval operates per object rather than on flattened scene slots. No evaluation experience is admitted to memory. All variants receive identical underlying supervision; a no-history ablation intentionally removes information and is not the equal-information baseline.

## Frozen small pilot

The protocol in configs/v3_1_physics/pilot.json predates neural execution. Two seeds(11,23),25 trajectories per family,160 updates, batch32, eight-step training rollouts. Six IID-like and seven structural stress development scenes per family, evaluated at8/16/32 steps. Total780 new forecasts and32 actual selected-action plans;16 trained fits plus four derived memory models. Additional V3 validation diagnosis is separate. No pretrained model, paid service or CPU neural execution is permitted.

New motion training exposes both one and two objects; stress configuration exposes three. This is a redesigned environment distribution, so improvements must be judged against the new matched baseline, not claimed as percentage improvements over the previous benchmark. Panel-specific seed derivation prevents identical physical tasks across training/development/sealed panels. Twelve IID and fourteen OOD sealed scenes per family are frozen under an independent base seed before training; the pilot never opens their trajectory files.

Strict development gates, declared before training: all outputs finite; mean32-step stress MSE≤10;32/8 error growth≤4; at least10% candidate error reduction versus matched in each family; nominal95% coverage≥80%; planning success no worse than matched; pilot runtime≤600 seconds. Units use fixed normalization1, so the pilot reports state-coordinate MSE, not V3's training-std-normalized benchmark metric. Paired scene bootstrap averages seeds within each scene, never treating frames or repeated seeds as independent samples. Seven stress scenes are too few for broad scientific claims.

Component retention requires reproducible benefit across both seeds and families and subsequent independent validation. A favorable single seed cannot select a component. Sealed testing and any larger study require a new preregistration after the development gates pass. A failed pilot triggers diagnosis rather than scaling.

The same public typed state/action/provenance boundaries remain suitable for eventual math/coding integration. No transfer experiment has run, and no cross-domain generalization is claimed.
