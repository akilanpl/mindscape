# R5.7 exploratory candidate changes

Evidence supports a targeted candidate, not a claim of improved held-out learning. Original R5.6 files and scientific authorization remain unchanged.

- Math: retain graph/recurrent state and add a learned head with one-hot observed left/right digits, incoming bit and operation. This is an explicit digit-identity/typed-readout prior. No addition, subtraction, modulo, outgoing-carry rule or target enters the forward path.
- Coding: shared learned byte recurrent encoder and query/candidate score head. The same scorer competes across candidates; it receives no correct index or equality flag. Canonical candidate ordering stabilizes the core context, and scores are returned in caller order. This is a query/candidate structural prior. Candidate permutation equivariance is tested; arbitrary alpha-renaming invariance is NOT guaranteed and must be reported separately. No symbolic equality solver is counted as neural output.
- Physics: add direct signed-log normalized force, mass, dt, velocity and impulse observations to a learned head. No force-times-time-over-mass feature, analytical prediction or physical constraint is inserted. Core graph observations retain their original representation.
- Concepts: unit-RMS embedding normalization and scalar tanh residual gate initialized exactly zero. Concept-on/off common weights and initial predictions match. Graph-off pins this contribution to zero. This tests interference; it does not establish a causal explanation of the R5.6 gap.
- GRU: retains recurrent observed-token processing; mean plus final sequence-state projection improves access to early observations. Direct task readouts are shared in form across all conditions. Graph core still uses message passing and GRU memory fusion. Memory resets per independent example.

Parameter matching does not establish compute matching. Original graph task heads remain registered but their predictions do not contribute to the new domain losses; list active-gradient counts separately. Byte encoding and pointer scoring increase computation. Actual native time and allocation will be recorded for every fit; end-of-fit allocation is not a peak-memory measurement.

Single fixed candidate and bounded tiny CPU proofs precede final-test scoring. The main campaign retains the exact legacy 128 records/order, Adam .001, batch one and 2,000 updates per fit. No loss reweighting or balanced sampling is adopted: the measured small comparison did not improve all domains. Final update 2,000 is the sole checkpoint rule. Final test never feeds training or checkpoint selection.

The new names/templates and numerical test distribution differ from the old exploratory validation. Compare R5.6 and R5.7 on the same new test, not old aggregate scores against new ones. Finite math identities are disjoint within each seed; splits across seeds are not claimed independent. No scientific significance or acceptance is established by this exploratory run.
