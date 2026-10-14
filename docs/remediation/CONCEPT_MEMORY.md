# Supplemental concept and memory identifiability protocol

Status: preparation and random engineering probes only. No training, fitted
checkpoint, learning result, qualification signature or discovered-concept claim.
The historical 72-fit / 144,000-step primary grid remains unchanged. This supplement
has its own protocol identity and requires separate future authorization and a
frozen evaluation plan before any learning experiment.

The concept task supplies a categorical relational context independently of the
observed numeric state, action and domain. Four classes choose scale/offset pairs
(1,0), (1,3), (2,0), (2,3); a separately protected label is
`(scale*(left+right)+offset) mod 10`. Every class occurs at every included numeric
state with the same action/domain. For state (0,1), the four contexts require four
different outputs. Thus operation identity cannot determine context, and omitting
context makes these counterfactual labels ambiguous. Class meanings and labels are
hand-specified supervision; no concept discovery is implied. The label generator
is outside the model and never acts as an inference fallback.

`concept_curriculum` splits numeric state pairs deterministically into disjoint
training and held-out partitions, with all concepts represented in each.
`pack_concept` accepts only observed state/action/domain/context fields and rejects
full records, labels and extra keys. Model inputs contain no scale, offset, expected
class, outcome, identity or split indicator. Feature invariance under protected
label tampering is tested. The supplementary `ConceptProbe` is a small original
neural model, not a replacement for or modification of the frozen R56Model.

The context-vector and embedding controls receive the same categorical information
and use exactly the same trainable context matrix and prediction parameters.
Embedding indexing and one-hot matrix multiplication yield identical logits.
This explicitly prevents a false claim that an arbitrary categorical embedding
outperforms an equivalent context vector because it has access to extra labels.
An omitted-context control uses a zero vector with the same parameterization.
Corruptions are independent per-example categorical draws with replacement, fixed
by a local seed, rather than a global permutation the model could simply learn.
Independent corruption can coincide with the true context by chance; its chance
agreement rate must be retained in any future report. Corruption leaves protected
labels unchanged. No corruption ID is accidentally packed into observations.

The delayed-cue task presents a cue class, a neutral delay and a query class.
The protected query label is `(cue+query) mod 4`. Paired episodes have identical
current query and delay observations but different required labels. Only the
previous cue disambiguates them. The `DelayedCueProbe` shares exactly one encoder,
GRU, fusion layer and head across retain/reset/shuffled-past conditions. Reset
zeroes incoming state but still executes the same GRU and fusion path, avoiding a
parameter count or architecture difference. Shuffled-past requires explicit donor
hidden states and detaches them to prevent cross-episode gradient leakage. Donors
must be independently paired under a future frozen protocol; the helper does not
pretend that an arbitrary supplied donor assignment establishes a random control.

Each episode starts from empty state. Within retained episodes, original predicted
hidden state feeds the next transition without a verifier or target correction.
Hard class predictions are preserved in the returned records; they are not fed back
as a fabricated ground-truth observation. Bounded episode length and tanh state
prevent growing history. Return tensors preserve within-episode gradient flow for
engineering connectivity tests; no caller should carry them across episode
boundaries. Current random forward/backward probes show bounded state, identical
reset query outputs under paired cues, dependence on retained/shuffled histories,
and connected encoder/memory/fusion/head gradients. They do not establish learned
memory utility or generalization.

Six engineering tests pass with the installed PyTorch runtime. They snapshot model
weights before forward/backward and verify unchanged parameters afterward, and
install an optimizer-construction tripwire for the concept probe. No optimizer
construction, optimizer step, training campaign, historical rerun or self-issued
signature occurs. Full learning/evaluation of this supplement remains unrun.
