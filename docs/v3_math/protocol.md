# Mathematical concept-learning protocol v1

The protocol and datasets are committed before held-out evaluation. Unit tests use
separate synthetic `unit` problems and temporary synthetic training problems; they
do not use frozen holdout outcomes for model design. Hyperparameters are fixed at
600 Adam updates, batch 64, lr .003, seeds 0/1/2, nested distinct-example budgets
10/25/50/100/250/500/1000. No validation/IID/OOD result selects a checkpoint.

There are 1,000 train, 60 validation, 90 IID, 150 structural OOD (30 each for unseen
length, carry-chain, digit-composition, sign-change and two-operation composition),
and 90 separately encrypted lockbox tasks. Canonical identities include ordered
subtraction/programs and sorted single add/multiply inputs. All partitions are
identity-disjoint. Familiar training inputs use one/two digits from 0..7 with
positive operands or negative left operand. Length OOD uses three/six-digit inputs
from that alphabet; digit-composition OOD introduces 8/9 at familiar lengths;
sign-change OOD uses negative right operands; composition chains two familiar
operations at familiar operand lengths. Carry-chain OOD includes long chains and
9s; it intersects length/digit shifts and must not be interpreted as isolated.

Primary metric is exact task accuracy on the entire frozen OOD set. The aspirational
50% family is predeclared as `length`. Success requires exceeding the appropriately
matched procedural baseline, not just crossing the target or beating answer-only.
All incorrect proposals and unsuccessful episodes are retained. Wilson intervals
are per seed/task cell; paired bootstrap resamples tasks with their three seed
outcomes together. Exact paired tests receive Holm correction over the declared
budget/baseline/seed family. Seed copies are not independent tasks. Statistical
significance does not establish autonomous algorithm discovery.

The neural learner predicts operation, signed local raw result, digit, carry/borrow,
next digit/carry, sign and transition consistency. It never calls the arithmetic
teacher. Every proposed next-state value is written unchanged; the verifier runs
only after a complete rollout. No correctness feedback guides inference, rejects
an incorrect claim, or supplies corrected state. Syntactic vocabularies are full
factorized choices, independent of targets. Planning considers at most four neural
next-state alternatives and uses neural consistency/confidence only.

Place-value scheduling, signed magnitude sorting/normalization and decimal output
materialization are explicit deterministic scaffolds. They do not compute local
numerical answers but provide procedural knowledge. This experiment studies neural
local rules and their scaffolded composition, not autonomous planning discovery.
The final sign remains a learned readout; normalization does not return it.

The equal-information procedural model shares the exact trained checkpoint, input
relationships, auxiliary labels and final-answer labels with the concept wrapper.
Thus memory and planning ablations are fixed-weight inference interventions. Concept
memory stores finite demonstrated local transitions under typed relation/context
keys. It cannot query a calculator or add held-out entries. A memory hit is empirical
retrieval, not proof of neural generalization. Planning scores candidate claims with
a learned consistency classifier trained on positive and corrupted train claims.
The answer-only baseline has the same architecture and receives the same dataset,
but optimizes final labels only; it is not the primary causal comparator because
active supervision differs. All label and feedback counts are reported.

The compact MLP is trained for all budgets/seeds in both training regimes. A secondary
local MiniLM encoder has frozen pretrained weights and a trainable rank-eight
residual representation adapter plus the same categorical head system. It is not
full transformer fine-tuning or internal LoRA. The secondary study trains procedural
supervision for all budgets/seeds and evaluates the same inference mechanisms.
Pretrained encoder features are observations-only and cached. All tokenization and
forward work is accounted separately; cache-amortized timings are not cold latency.
No model download or paid call is authorized or needed.

The encrypted lockbox is never opened during development. The key is kept outside
the committed dataset directory. No lockbox score will be reported from these runs.
After freezing model/method selection, a separate final-evaluation protocol is
required to open it. Alternate-base evaluation is deferred unless the core target
and matched comparison succeed without adjusting test difficulty.
