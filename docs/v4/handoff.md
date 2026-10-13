# V4 shared evaluator: native pilot handoff

Engineering status: implemented, nonneural checks pass; native neural execution is pending. No V4 learning result, transfer claim or scientific GO is reported yet.

## Verified foundation

GitHub PRs #2, #3, #5 and #6 were confirmed merged on 2026-10-13. They were stacked onto different research branches, not all directly into main. V4 branches from `afe1ea2b06addb3185b159d06e71aadb76532f0b`, the actual PR #6 merge on `development/v3.1-physics-stability`. Ancestor checks pass for integration `9ce27a3`, math `aafa49e`, coding `904887a` and physics `fd82446`. Existing checkout and pinned assets are reused. No historical tracked files or websites are changed.

## Frozen pilot

`configs/v4/protocol.json` is the pretraining decision specification. Each domain has eight fresh training identities, four IID identities and eight OOD identities. Each target fit receives the first four training tasks; auxiliary conditions use eight training tasks from each other domain. No historical evaluation datasets or sealed holdouts are read. Existing frozen specialists and approved historical training memory remain common inputs.

Primary direction: mathematics + physics → coding, with four target-domain training tasks. Secondary directions: the other two pairs → mathematics and physics. These pooled-source results cannot attribute an effect to either individual auxiliary domain.

A: original specialist rankings on the frozen alternatives. B: domain-only learned evaluator. C: shared evaluator with actual auxiliary labels. D: shared evaluator with within-domain randomized auxiliary labels. B/C/D have identical 4,865-parameter networks and initialization per seed: separate domain projections into 16 units and a shared 16→8→1 head. Three seeds: 11, 23, 47. Each of 27 fits uses 120 updates, 32 sampled candidates per update, and Adam learning rate .003. Every fit has 3,840 candidate presentations. B repeats target labels during its first 60 updates; C/D use auxiliary examples then all conditions receive 60 target updates. Source domain input dimensions differ; actual wall time is reported rather than claiming identical FLOPs. There is no zero-shot claim.

Every condition receives the same public features, proposals and frozen specialist scores. Math features include the actual local operands/context and proposed claim. Coding uses distinct before/after token hashes and numeric literal features, visible contract/tests and initial visible execution feedback. Physics includes masked current observation, observation mask, public context, target, horizon, proposed action and model-predicted terminal state/variance. Features never receive post-candidate execution, private tests, reference answers or hidden simulator state. Private training outcomes become supervision only after features/proposals are persisted; evaluation outcomes never enter optimization. Encoders remain imperfect: coding loses token ordering and hash collisions are possible; physics omits earlier public history even though its frozen predictor uses it.

Math exposes the four actual top-2 digit × top-2 carry claims of the frozen transition network, scored by its preserved consistency ranking. No correct answer is injected. Its endpoint is local digit/carry correctness, **not whole-problem mathematical accuracy or full claim consistency**. Coding exposes actual Qwen repairs plus unchanged input; identical repairs are deduplicated. Physics exposes all nine existing impulse alternatives with actual frozen dynamics forecasts. Original specialist rankings are preserved. Candidate availability, task success and success conditional on availability are all reported; unsolvable tasks stay in denominators. Malformed generated responses stop the run and preserve a failure receipt, rather than receiving invented replacements.

OOD is incoming carry/borrow for math, count instead of sum for coding, and 16 instead of eight planning steps for physics. These are bounded development shifts relative to evaluator training, not proof of unseen structures for pretrained specialists. Math OOD also differs in operand range. The small deterministic coding family is not a broad repair benchmark.

## Decision gate

GO means proceed to a separately frozen confirmatory study only. It requires at least +10 percentage points mean coding OOD accuracy versus both B and D, positive gains for every seed against both controls, no other domain worse than B by more than 10 points, at least two solvable OOD tasks per domain, and meaningful shared-model training Brier improvement (at least .01 over both initialization and constant-prevalence prediction). All target training sets must have both candidate labels. Otherwise NO-GO for scaling this mechanism. No threshold changes, extra seeds, larger budgets or repeated attempts are authorized by the launcher.

Task correctness, candidate calibration (Brier, five-bin ECE), candidate availability, raw probabilities/selections, losses, updates, candidate presentations, fit/inference times and checkpoint hashes are saved. Specialist scores are not treated as calibrated probabilities. Eight OOD tasks per domain and shared task sets across seeds are inadequate for a final transfer claim. A failed gate warrants diagnosing whether proposal availability, encodings or shared discrimination failed before proposing a new hypothesis.

## Native command

Run once in normal macOS Terminal:

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v4/native.command
```

The launcher requires MPS and forbids CPU fallback/downloads. It uses the existing Python environment, frozen checkpoints, Qwen and WASI runtime. Twenty actual Qwen generations dominate collection time; there is no large-backbone or physics retraining. Output is a new `results/v4/native-<timestamp>` directory. The terminal prints progress and its absolute result path.

The run saves the source/task/protocol manifest before collection, proposal snapshots before verification, raw outcomes, compact checkpoints with Adam state, losses, comparison metrics and independent replay/audit receipts. Independent outcomes are computed by the preserved write-only math executor plus separate arithmetic, actual WASI execution and fresh Pymunk environments per alternative. No generated Python runs in host Python. Audit recomputes selections and metrics from recorded probabilities; it does not independently rerun neural predictions.

`--resume <existing-run>` supports training continuation only after full candidate collection. Model/optimizer state is saved at updates 60 and 120; the sampling schedule is reconstructed from seed and step. Source, protocol and dataset identities must match. Updates since the last checkpoint repeat. Native bitwise continuation is not yet verified; this is not a claim of proven exact MPS reproducibility. Partial collection is preserved as a failure and is not silently retried.

## Current verification and outstanding delivery

Ten nonneural tests pass, including real WASI correct/near-miss execution, physics action reset/replay, split disjointness, private-field rejection, numeric literal sensitivity, arithmetic write-only behavior and metric denominators/calibration. The existing integration suite also passed (15 tests and four subtests). Scoped Ruff and Python compilation pass; launcher syntax passes. All eight existing asset preflight entries are present and hashed. No MPS availability probe or neural training was attempted in the sandbox.

Questions 1–5: implementation exists but native execution awaits verification; improvement, positive transfer and negative transfer are unmeasured; the scientific continue/revise/abandon decision awaits the frozen pilot. After native execution, independently inspect receipts/checkpoints, preserve evidence, write exact measured conclusions and open one V4 PR against the verified integration base. No final PR or trained-model result is represented as complete at this boundary.
