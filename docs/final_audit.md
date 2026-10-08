# Audit before final model changes

The phase-4 artifacts and all historical code remain intact. The audit used source inspection, the existing numerical-gradient regression, dataset validation and executed diagnostics in `experiments/final_audit/`.

| Component | Finding |
|---|---|
| Input/numeric representation | Answer baseline uses ordinal decimal digits; policy uses four one-hot local digits. This information/representation difference matters. Neither receives the final answer. |
| Labels/output/loss | Nine independent categorical heads; policy uses only two numerical heads while seven constant labels consume 7/9 of the loss. Answer labels contain many padding zeros. High per-digit accuracy is misleading for exact arithmetic. |
| State/targets | Strict typed serialization; reference targets replay authoritative claims. Own wrong claims are executed unchanged during inference. No backfilled trace. |
| Optimizer/normalization/batching | Adam implementation and gradients have regression coverage. Fixed 600 steps, replacement batches, bounded features. Six thousand steps improve tiny-set fit; this does not isolate one causal explanation. |
| Splits/OOD | Canonical operand identity prevents swapped-pair leakage. Structures are held out. All 45 nonzero canonical 1x1 pairs are exhausted in training, so the IID validation/test contain no 1x1 examples. |
| Seeds/checkpoints/decoding | Seeded initialization and minibatches; saved arrays checked for shape/finite values. Independent units/carry decoding makes exact claims harder. A skip score uses the sign head, a semantically artificial coupling. |
| Metrics/execution | Exact answer, entire verified trajectory and goal are separate. Wrong claims can cancel to a correct answer. Baselines without trajectories cannot satisfy evidence-based groundedness. |
| Candidates | 101 open candidates or 100 syntactically legal numerical values. No singleton correctness mask. |
| Supervision/information | B gets all transition labels; C gets bounded binary feedback on own proposals. Policy state exposes a hand-designed long-multiplication cursor. This is domain structure, not evidence of discovering a procedure. |
| Costs/class imbalance | C's 64 attempts often cannot finish an episode. Accepted replay is sparse and biased toward early states. Constant heads/padding dominate labels. B's multiple labels per problem are not comparable to one answer label without qualification. Dream increases model calls. |

Executed same-set diagnostics at seed 0 and 6000 updates: 10-example answer and policy accuracy both 100%; 45-example 1x1 answer accuracy 46.67%, policy accuracy 100%. These are memorization tests, not benchmark results. Further carry/within-structure diagnostics are recorded separately. The pipeline can learn and execute; the phase-4 failure is not explained by a universally broken label/gradient/decoder pipeline. Capacity, representation, optimization budget, feedback sparsity and distribution coverage remain plausible interacting causes.

Final interventions must preserve this negative study. Remove dummy policy losses and artificial skip/sign coupling in the new adapter, use an actual pretrained local backbone and a richer learned head, retain explicit resource and information differences, and freeze the new protocol before inspecting its held-out results. No intervention supplies computed arithmetic as a feature.

The additional seed-0 n100 trajectory diagnostic memorized its training set at 100%; on validation it reached 17% exact answers, 66.46% local claims and 81.43% carry labels. Thus local errors compound during execution. These measurements use validation only and do not select final-study hyperparameters.

The final representation also passed executed sanity checks: answer-only/structured/B memorized their n10 training subsets at100%; C achieved0% with incomplete accepted experience. All45 canonical1x1 problems and a mixed100-example set were memorized at100% by final answer/B heads with1200 updates. On the new validation set the separately trained mixed100 diagnostic reached3% answer-only and18% B exact accuracy; B local-claim accuracy62.47%, carry accuracy82.60%. C's missing successful sequences are a feedback-coverage failure, not evidence that the supervised pipeline cannot memorize. These diagnostics were recorded after protocol freeze and did not change hyperparameters.
