# V4 pilot: preserved NO_GO and independent diagnosis

Run `results/v4/native-1791854687819867000`, implementation `7d890e3`. Original gates, requests, checkpoints, outcomes and receipts are unchanged. Engineering is executable; beneficial learned cross-domain transfer is **not established**. Do not scale this mechanism on this result.

## Verification

Independent execution replay on a copy passed all **60 tasks and 320 candidate outcomes**: original write-only math executor plus separate arithmetic, actual visible/private WASI suites, and independent Pymunk action replay. All **27 checkpoints** match their recorded SHA-256; weights are finite, total 4,865 parameters each, Adam moments have correct shapes and finite values, and optimizer steps match the design (shared head 120; B target encoder 120; C/D target 60 and each auxiliary encoder 30). All **54 evaluation metric sets** were independently recalculated from saved probabilities. The original task/protocol/source/asset hashes match; original six gate booleans recompute exactly. All **120 original run files** are preserved byte-for-byte.

Limits: checkpoint tensors were inspected on CPU, without neural forward or training. Saved probabilities were not regenerated from checkpoint weights. Training Brier differences were checked arithmetically against saved scalar receipts; those training probabilities were not archived. Bitwise continuation remains untested. These limits are not concealed by the replay PASS.

Machine-readable diagnosis: `experiments/v4/review_v1/diagnostics.json`, `metrics.csv`, `training-distinction-gates.csv`, and label CSVs by split, source, task and individual candidate. The latter two enumerate all 60 tasks and 320 candidate identities. `experiments/v4/native_v1/evidence.tar.gz` contains the complete original run, with a hash manifest and verified fresh restoration. Original frozen source files are separately archived for reproduction.

## Exact outcomes

OOD correct selections, seeds 11 / 23 / 47. Every denominator is eight tasks; oracle counts do not change by seed.

| Domain | Specialist | Domain-only B | Shared C | Randomized auxiliary D | Oracle |
|---|---:|---:|---:|---:|---:|
| Mathematics | 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 2 |
| Coding | 8 | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 | 8 |
| Physics | 2 | 1 / 1 / 1 | 2 / 2 / 1 | 2 / 2 / 1 | 6 |

IID correct counts are identical across all conditions and seeds: math 3/4, coding 4/4, physics 0/4. Primary gain, seed consistency, learns_distinctions and mixed_target_training_labels fail. Candidate availability and no-large-negative-transfer pass. The original availability threshold was only two solvable OOD tasks, so math passes it despite a 25% ceiling.

No task-accuracy negative transfer relative to B was observed. Shared physics underperforms the original specialist in seed 47 (1/8 versus 2/8). Shared coding calibration is worse than B on average; this is a small descriptive negative effect, not a statistically established population effect. Twenty-seven fits took 4.368 seconds in saved receipts; collection and verification costs are separate and preserved in raw rows.

## Label diagnosis

Entries are positive / negative candidate labels, not task counts. The first four training tasks are the actual target-supervision set; the full eight are used when that domain is auxiliary.

| Domain | All 8 train | First 4 target train | IID | OOD |
|---|---:|---:|---:|---:|
| Mathematics | 7 / 25 | 4 / 12 | 3 / 13 | 2 / 30 |
| Coding | 8 / 16 | 4 / 8 | 4 / 8 | 10 / 14 |
| Physics | 1 / 71 | **0 / 36** | 0 / 36 | 9 / 63 |

Positive candidate counts in task order (negative count is 4 minus count for math, 3 minus count for coding, 9 minus count for physics):

| Domain | Train tasks 0–7 | IID tasks 0–3 | OOD tasks 0–7 |
|---|---|---|---|
| Math | 1,1,1,1,1,1,1,0 | 1,0,1,1 | 0,1,0,0,0,0,0,1 |
| Coding | 1,1,1,1,1,1,1,1 | 1,1,1,1 | 2,1,1,1,1,1,1,2 |
| Physics | 0,0,0,0,0,0,1,0 | 0,0,0,0 | 0,1,1,2,1,1,0,3 |

Physics's only training positive belongs to task 6, excluded from its four target tasks. Thus mixed_target_training_labels fails specifically for physics. Constant-zero prediction has Brier **0**, making the required .01 gain over that baseline mathematically impossible. Low final Brier here means predicting failure, not learning useful distinctions. Shared final physics training Brier is .000709 / .001235 / .001132.

The shared distinction gate also fails math seed 23: gain over constant .007869 (<.01). Math seeds 11/47 pass at .031853/.026671. All shared coding seeds fail at .009356/.009699/.008023. All shared fits improve over their random initialization; improvement over initialization alone is inadequate evidence.

Source counts: coding Qwen candidate 1 is correct on **all 20 tasks**; candidate 2 is positive on 0/8 train, 0/4 IID, 2/8 OOD; unchanged is negative on all 20. Eighteen tasks contain an AST-identical Qwen/no-change pair differing only in whitespace, so byte-level deduplication overweights negatives. Math positive train candidates occur at digit-rank/carry-rank (1,1): 5 and (2,1): 2; IID positives all at (1,1): 3; OOD positives all at (2,1): 2. Other math source ranks have zero positives. Physics positive train impulse is (-2,0): one; IID has none; OOD positives by impulse (-2,0),(-2,1),(-2,2),(0,0),(0,1),(0,2),(2,0),(2,1),(2,2) are 2,2,1,3,0,0,1,0,0. Complete positive and negative source counts are in `labels-by-source.csv`.

## Oracle ceilings

Math OOD uses incoming=1 throughout. All six unsolvable tasks omit the correct **digit**, not merely the carry, from the frozen two-digit proposal vocabulary. The correct carry is present in all eight candidate sets.

| Local state (a,b,incoming) | Operation | Correct digit/carry | Offered digits | Solvable |
|---|---|---|---|---|
| 6,8,1 | add | 5/1 | 3,1 | no |
| 6,1,1 | sub | 4/0 | 5,4 | yes |
| 7,4,1 | add | 2/1 | 1,3 | no |
| 7,7,1 | sub | 9/1 | 1,0 | no |
| 8,0,1 | add | 9/0 | 8,7 | no |
| 8,3,1 | sub | 4/0 | 1,0 | no |
| 9,6,1 | add | 6/1 | 3,1 | no |
| 9,9,1 | sub | 9/1 | 1,9 | yes |

A selector cannot recover an absent digit. This is a frozen-specialist proposal bottleneck combined with an incoming-context shift and changed operand range. No counterfactual larger-top-k experiment was run; no answer was inserted.

Physics OOD minimum distances over the nine actual actions, in task order: **1.311854, 0.381445, 0.583896, 0.325843, 0.717385, 0.476320, 0.814154, 0.225488** (see raw diagnostics for full precision). Tasks 0 and 6 exceed the unchanged .75 gate for every offered action. This establishes unreachability within this candidate set, not impossibility under all physically legal controls. Training uses eight steps and low absolute target heights unrelated to current scene position; all first-four training targets and all IID targets fail for every action. Sixteen-step OOD scenes yield more positives. This supports a task/action/horizon mismatch; it is not evidence that the learned evaluator solved planning. The simulator's independent outcomes determine these ceilings, not model forecast scores.

## Why physics shared and randomized match

For seeds 11 and 23, both C and D select action index **0 = impulse (-2,0)** on all eight OOD tasks, succeeding on tasks 3 and 7. Seed 47 selects index **6 = (+2,0)** on all eight, succeeding only on task 5. B always selects index 6. Thus C and D have identical actions and successes, not just coincident aggregate counts.

They are not duplicate models: C/D maximum weight differences are .335085/.177925/.183770, and maximum OOD probability differences are .002166/.004861/.003899. Actual auxiliary-label semantics did not improve these choices over label permutation. With only failure labels in target training, different source-conditioned initializations leave fixed-action ranking biases. That is a supported explanation for the measured behavior, not proof of an internal causal mechanism. The +8.33-point mean physics gain over B also occurs under D and stays below the original specialist's 25% success.

## Coding ceiling and secondary comparison

Every condition has 100% task selection, conditional selection and within-task positive-versus-negative pairwise ranking on OOD. A trivial first-candidate selector also scores 100%. None provides headroom for detecting transfer here. The candidate family is largely a repeated one-character boundary correction, with two semantically duplicate negatives on most tasks.

Candidate calibration retains numerical variation. Mean OOD Brier (lower better): **B .239065, C .243317, D .241523**. Mean five-bin ECE: **B .084414, C .111173, D .104165**. C is worse than B by .004252 Brier and .026759 ECE; seed 23 improves Brier, seeds 11/47 worsen it, and C is worse than D on Brier in all three seeds. Candidate probabilities understate the correct candidates despite ranking them first. These are descriptive secondary comparisons on 24 candidates/eight shared tasks, not new gates, rescued success or reliable significance evidence.

## Smallest next experiment — data qualification, not training

Freeze `configs/v4/fresh-development-proposal.json` separately. First run at most **36 fresh inference-only development requests** (8 train + 4 qualification per domain), with unchanged specialist checkpoints, candidate budgets and network. Balance math incoming contexts; construct physics goals from public current position rather than unrelated absolute low targets and balance horizons; diversify four small coding defect families, deduplicate AST-equivalent candidates and deterministically shuffle candidate order. No original OOD tasks are reused, relabeled, filtered or selected for the new study.

Before any new inference, implement and commit the complete fresh generator/task/scorer manifest. This has not been executed. Hard pretraining gates: each domain's fixed first four training tasks must contain at least four positives, four negatives and two mixed-label tasks; each four-task qualification split must have at least three solvable, mixed-label tasks; coding must retain at least one-task headroom over both the specialist and first-candidate selector. Any failed gate, generation or integrity check means **DATA_NO_GO and stop**, with no replenishment, broader candidate search or training. Passing means **DATA_READY and stop**.

Only after that screen and separate authorization should a single-direction, nine-fit coding-target pilot be considered, with a separately frozen 16-task evaluation set. Proposed gates are fixed in the JSON: +2/16 mean correct over both controls, positive per-seed gains, training distinction gains >=.01, no mean Brier worsening, and all replay/integrity checks. Failure stops the mechanism; success permits independent confirmation only. No architecture rewrite, large-model retraining, GPU training or new candidate inference was performed for this review.

## Decision

1. Implemented and native-executable: **yes**, 27 trained compact checkpoints preserved.
2. Better than equally informed domain-only learning: **no primary coding improvement**; physics gain lacks label-semantic specificity.
3. Positive cross-domain transfer: **not established**.
4. Negative transfer: no task-accuracy degradation versus B; descriptive coding calibration degradation and one physics seed below the original specialist.
5. **Preserve NO_GO; revise data qualification before considering this mechanism again.** Do not scale or declare the shared-learning hypothesis disproven by a degenerate target-label set.
