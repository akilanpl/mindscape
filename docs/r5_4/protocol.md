# R5.4 frozen protocol v1

Scientific question: can a trained unsigned-addition digit/carry transition compose across unseen lengths and generalize to unseen digit/carry contexts, using its own predicted carry? No multiplication, subtraction, signed arithmetic, specialist integration, deployment or Alpha launch is part of this experiment.

## Start with R5.2 failure

Read sources: `experiments/r5_2_math/recovery_v1/REPORT.md`, `summary.json`, `gate-values.json`, and the recovery implementation. The recovered recurrent exact OOD mean was 8.33%; reset-hidden 25%; memory-only 21.53%. Recurrent length accuracy was 11.11%, carry/borrow 16.67%, and unseen-context 2.78%. Intermediate accuracy was 81.7453%, despite exact OOD 8.33%. Seed exact counts were 12,18,6 /144. Optimization above zero weights is evidence of learning some predictions, not evidence of whole-answer compositional generalization. Error accumulation, sign/stage prediction and weak unseen combinations explain the distinction. Resetting hidden state did better, so recurrent retention is a hypothesis, not an advantage assumed here.

The attribution variable was overwritten inside a carry loop, and replay compared base verifier fields against added diagnostics. R5.3 also exposed float32 interval-score sensitivity. R5.4 uses immutable condition identifiers, distinct base/diagnostic records, integer truth and explicit floating dtype metadata. It preserves these historical artifacts without reruns or rewriting their conclusions.

## Models and supplied scaffold

Markov: one-hot inputs 10+10+2, Linear(22,32), tanh, Linear(32,32), tanh, Linear(32,12), split into 10 digit and 2 carry logits: 2,188 parameters. Recurrent: identical input, GRUCell(22,16), Linear(16,12): 2,124 parameters, 2.93% fewer than Markov. No embeddings that contain answers, sum features, positional features, output masks, verifier guidance or exact lookup in either model. Hidden state resets at the start of each task only; Markov retains none. Free inference returns unmasked argmax classes, including incorrect carries.

Supplied operations: canonical decimal inputs, LSD-first cursor, shorter-operand zero padding, initial carry zero, one model call per operand column, final predicted carry serialization, removal of leading output zeros. This is substantial algorithmic scaffolding: digit order, termination and representation are supplied. Learned operations: digit and carry classification. The model discovers neither the addition algorithm's traversal nor its stopping criterion.

`inference.Operands` exposes only ID/left/right; `model.adapter` consumes only digit observations and learned state. Neither imports the training/reference/scoring modules or reads files. Tests poison reference access and attach unreadable labels. This proves the implementation's input boundary and source separation; it is not an OS security isolation claim. Public operands and committed outcomes are accessible to humans, but outcomes are never passed to models. No OOD neural predictions are produced during qualification.

## Data and qualification

Generator RNG 54013, canonical identities exclude operand commutations across every split. Fixed allocation: 1,024 training additions of 2–4 digits; 64 independent same-distribution development tasks; four OOD families of 48 each. Length-known: 8–16 digits. Long-carry-known: 12–24 digits, carry initiated at the first column and propagated through every remaining column. Mixed-known: 8–16 digits, alternating carry/no-carry targets. Unseen-short: 2–4 digits with exactly one reserved digit pair, isolating unseen local combinations from length extrapolation. No allocation adjustment after prediction.

Reserved pairs: (1,9),(9,1),(2,7),(7,2),(3,6),(6,3),(4,8),(8,4), both incoming carries. Excluding the pair itself prevents indirect held-context exposure even when rollout training feeds incorrect carries. Training covers all other 184 contexts. Development covers 112 contexts. OOD covers 192 contexts. Family unique contexts: 175 length, 35 carry, 77 mixed, 77 unseen; the last includes 12 of the 16 held contexts. Separate reset-state probes cover all 200 contexts including all 16 held contexts, diagnosing gaps not naturally covered by the 48 tasks. These probes are not additional independent arithmetic tasks and do not inflate task intervals.

Qualification compares a column reference against Python integer addition for every task. Memory coverage independently confirms the three long known-context families contain no unseen transitions. Qualification may evaluate nonlearned controls and reference correctness, but never trained or untrained neural OOD predictions. Held exposure during the campaign is measured and must remain zero.

## Six fits and exposure objective

Seeds 11,23,47 for each model. Native MPS float32 only. AdamW lr .003, weight decay .0001; gradient clipping norm 1. Batch32; 800 updates (25 passes through 1,024 training problems), RNG permutation each pass, same data order policy per seed for both architectures. Final checkpoint update800 only; no development-based selection. No retries, new seeds, sweep, resumption, early-stop tuning or outcome-driven changes. Development performance is reported after fitting and is never used for model selection.

Updates1–400: teacher-forced local digit CE + carry CE. Updates401–800: equal average of the same teacher-forced objective and an own-carry trajectory objective. At every own-carry column, the labels describe the correct local transition for the *actually visited* input, calculated only within training supervision. A bad incoming carry is not corrected. The loss teaches a local transition under imperfect states; it is not end-to-end answer reinforcement learning and does not optimize a differentiable full-answer metric. Gradients flow through recurrent hidden state across up to four operand columns; hard carry argmax is detached. Both models receive the same training tasks, schedule and loss policy, but their own-state trajectories can differ. Padding is masked from loss. Exact arithmetic in training labels is explicitly allowed, excluded from learned inference.

Both teacher-forced diagnostics and free-running exact-answer accuracy are required. This single six-fit design evaluates the bounded exposure objective but cannot establish its causal advantage over TF-only training: no extra ablation fits are authorized. Report that limitation.

Resource qualification: nontraining MPS forward/backward of a 32-task training batch with the second-half objective, finite nonzero gradients in every parameter tensor, placement verified, parameters unchanged. Conservative projection = measured time ×800×1.5 must be <=720 seconds per fit; RSS <=4GiB. Runtime limit 900 seconds per fit, checked between updates. Thus bounded compute assumes a backend call returns; a hung driver is not terminated by this in-process budget. Six fits max; no optimizer if qualification fails. No dependency or model downloads; CPU regression checks are separate from MPS execution and never a fallback campaign.

## Controls, gates and claims

Local memory stores only exact training contexts, miss -> digit0/carry0. Whole-problem memory stores training operand identities, miss -> answer0. Zero-weight controls for both architectures, argmax first class. Independent deterministic reference is reported as 100% oracle ceiling, excluded from learned success and all gain requirements. Local memory already solves 144/192 OOD tasks and 0/48 unseen tasks. Overall headroom is exactly25 points. Requiring gains on saturated known-context families would be impossible, so compare there for parity/composition only.

Frozen gates, separately for each model and each of all three seeds:

- Overall exact OOD >=95%, Wilson95 lower >=90% (192 tasks).
- Every known-context family, including long-carry, exact >=95% (48 each).
- Unseen-short exact >=90%, Wilson95 lower >=78% (48 tasks).
- Reset-state held-context probe joint digit/carry accuracy >=90% (16 contexts; a diagnostic gate, no task-independence claim).
- Overall gain over local memory >=15 points; unseen-short gain >=75 points.
- One-sided exact McNemar vs local memory, overall and unseen-short, p <=.05/12. Bonferroni covers 2 models×3 seeds×2 planned tests.

A learned claim requires every gate in all three seeds for that model. These stringent gates require almost error-free composition and near-complete unseen-transition transfer, with intervals guarding small denominators. Report negative results rather than relaxing thresholds. Finite-table learning, known-context length composition and unseen-context transfer remain separate claims; one overall percentage cannot substitute for them. A stateless winner is a valid result. Recurrent-vs-Markov paired comparisons are exploratory, per seed, with no predeclared recurrence advantage gate. Seeds are repeated fits on the same task manifest, never 576 independent tasks. No pooled-seed binomial intervals. Wilson95 intervals per seed/family, exact task-level paired counts and full error traces are saved.

## Evidence and audit

Commit protocol, data, sources and source/data SHA256 lock before any training. Training command verifies frozen files are committed and unchanged. Every fit has a unique model/condition/seed/update identifier; every task a unique split/family/index ID; every trace has an explicit horizon, column and input/output carry. Save all 800 loss entries with float32 metadata, finite float32 parameter tensors, optimizer tensors and update counts, visited contexts, checkpoint hashes and predicted answers.

All six neural and nonlearned free predictions are written with exclusive creation and fsync, followed by a prediction seal, before teacher-forced diagnostics or outcome scoring. Local diagnostics and base verifier results remain separate. Final artifact receipt hashes all files. Interrupted campaigns retain failure records and partial evidence; read-only audit rejects them, missing completion receipts, altered hashes, duplicate/missing/out-of-order IDs, wrong carries, malformed dtype metadata, nonfinite tensors and incorrect optimizer counts. Audit uses independent divmod/reference reconstruction and float64 metric recomputation, independently replays nonlearned controls and compares saved metrics. Optional native MPS checkpoint replay performs no optimization and requires an already completed campaign. Outcome-open ordering is enforced by source control flow and evidence seal, not independently attested by an OS access log.

Before authorization, only dataset qualification, CPU regression tests and engineering qualification may execute. Neural optimizer campaign, neural OOD inference, TF evaluation and neural checkpoint replay are explicitly unrun. Preserve R5.1–R5.3, unified kernel, specialists, physics, website and tags; no deployment.

Preservation commands after an approved completed campaign: `python -m mindscape.r5_4 preserve --output results/r5_4/campaign-v1 --archive results/r5_4/campaign-v1.tar.gz`, and `python -m mindscape.r5_4 verify-archive --archive results/r5_4/campaign-v1.tar.gz`. Creation first audits completion and frozen sources, then archives source/data/lock and all campaign artifacts with a member hash receipt. Archive verification is read-only and never extracts files; reject unsafe paths, duplicate entries and missing/tampered members. No campaign archive exists before training.
