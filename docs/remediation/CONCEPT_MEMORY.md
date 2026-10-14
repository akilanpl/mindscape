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

The independent `conditional_information_audit` groups protected target counts by
observed numeric state/action/domain, then by those inputs plus supplied concept.
It reports empirical Bayes-optimal correct counts, conditional target entropy,
conflicting groups and their conditional information difference. For (0,1), the
four context classes require four distinct labels: without context, any fixed
prediction can satisfy at most one of four examples (25% empirical accuracy and
2 bits conditional entropy). With context, the four observations identify their
labels uniquely (100% empirical ceiling and zero conditional entropy). These are
finite-table identifiability facts, not model accuracies, training evidence or
population guarantees. Other state collisions are counted rather than hidden.

`delayed_cue_variants` leaves the original 16 default episodes unchanged and adds a
separate combination split. Training and held-out cue/query pairs are disjoint;
training delays are zero/one, while held-out evaluation includes delays zero, one,
four and eight. The latter separate combination generalization from increased
retention length. Labels stay outside `pack_memory` under every delay. The split
alone does not prevent a learner exploiting the supplied cue/query encoding, and
no learning result is asserted.

Trace records explicitly identify episode start, incoming-state reset, detached
donor use, within-episode graph retention and hidden-feedback source. Retain feeds
the original previous hidden tensor with its same-episode autograd graph intact;
reset replaces that incoming state with zeros while retaining the identical GRU
and fusion computation; shuffled-past detaches an explicitly supplied donor state
across episode boundaries. Tests demonstrate a gradient path to retained early
hidden state and no gradient path into donor episode tensors. Every new helper
call starts a new episode with no retained state. No label, verifier result or
corrected prediction enters these paths.

All these checks concern small standalone original probes. They neither execute
nor qualify the original R56 graph model, and they do not establish that its
concept or memory mechanisms work. Such a claim requires separately frozen,
authorized model-specific experiments. Nine engineering tests pass; optimizer
steps and all learning studies remain unrun.
