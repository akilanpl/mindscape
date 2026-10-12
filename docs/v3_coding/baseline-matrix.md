# Baselines and information access

Every model condition uses the exact same frozen Qwen2.5-Coder-1.5B-Instruct snapshot, greedy CPUfloat32 generation, at most two256-token calls, at most two proposals/call, the same initial visible failure and post-edit visible feedback, independent terminal private scoring, and no persistent evaluation update. A call can replace up to four existing files. No reference patch for an evaluation task enters any prompt.

| Condition | Runtime examples | AST concepts in prompt | Learned proposal selection | Contract framing |
|---|---|---|---|---|
| plain | Two fixed approved examples | No | Candidate0 | Contract still present in public task |
| structured | Same two fixed examples | No | Candidate0 | Additional explicit contract |
| typed | Same two fixed examples | No | Candidate0 | Explicit contract and typed-tool framing |
| mindscape | Two structured retrieved examples | Yes | Yes | Explicit contract |
| no_memory | None | Yes | Yes | Explicit contract |
| no_planning | Two structured retrieved examples | Yes | Candidate0 | Explicit contract |
| no_concepts | Two word retrieved examples | No | Yes, structural inputs zeroed | Explicit contract |
| word_memory | Two word retrieved examples | Yes | Yes | Explicit contract |

All conditions calculate evaluator predictions for audit/calibration; only the four learned-selection conditions use them to choose an action. Predictions for a candidate0 baseline therefore do not indicate that its policy uses a learned planner. A two-candidate cap does not guarantee that the model actually supplies two candidates; observed candidate counts and nonzero selections must be reported.

The24 training tasks are common, but this is not perfectly equal-information training: only learned-selection conditions use the96 proposal labels through trained weights; other baselines consume examples in context. Retrieval chooses different examples, prompt lengths differ, and no-memory intentionally removes examples. A future claim of neural-learning efficiency needs a same-backbone learned comparator given the same labels, gradient/feedback budget and cost. Ablations here are inference interventions on one trained evaluator, not separately trained architectural variants.

| Non-model control | Evidence | Interpretation |
|---|---|---|
| Unchanged source | Real visible/private WASI execution | Controller does not repair the task automatically |
| Constant-zero proposal | Real visible/private WASI execution | Uninformative valid source cannot obtain task success by controller assistance |
| Teacher reference | Real visible/private WASI execution | Explicit oracle upper bound; unavailable to evaluation learner |
| Literal-collision diagnostic | Training-only executed correct/wrong repairs | Falsifies semantic sufficiency of current evaluator features |
| Original B/C audit | Saved historical traces only | Behavioral diagnosis; no new-benchmark percentage comparison |

Software controls are not random/zero-weight Qwen experiments. The original V3 foundation separately checks random/zero neural numerical controls; those results must not be rebranded as coding-model evidence. Historical-adapter bridge code is optional and any execution would have unequal training history/action budgets; it is excluded from the matched study.
