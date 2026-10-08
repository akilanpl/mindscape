# Research conclusions

## Hypothesis

Explicit observations, entities, relations, states, actions, events, results, goals, memory and verification might improve learning and structural generalization, especially problem-budget efficiency. Architecture existence is not evidence for this hypothesis.

## Implementation

The final system adapts one frozen local SmolLM2 backbone under four conditions: conventional answer-only, structured answer-only, trajectory-supervised procedural B and binary-feedback experiential C. Procedural policies choose numerical claims among100 candidates. Real results execute those proposals unchanged, and the verifier judges after prediction. The multiplication cursor/decomposition is hand specified. Working/episodic/concept memory and bounded hypothetical simulation are implemented; conceptual interfaces are reusable, while current concrete schemas remain decimal specific.

## Observed results

Exact per-budget/seed measurements, mean±sample SD, intervals, paired tests and failure groups are in `docs/final_results.md` and `experiments/final_smollm2_v1/`. The old negative phase-4 study remains a separate frozen historical result. The fresh final dataset and larger representation/head produce higher absolute procedural accuracy, but no final target or threshold-crossing efficiency ratio is assumed from that fact.

## Interpretation

The experiments compare different information regimes, not only different wrappers. B gets several state/action labels per problem; C gets bounded scalar feedback on its own proposals; answer learners get one final-answer target. B's domain scaffold decomposes the numerical task. Any gain versus answer-only therefore cannot be attributed uniquely to Mindscape, memory, verification or dreams.

The matched-information conventional local supervised policy uses B's exact state/action training information and predictor while bypassing its memory wrapper. If its predictions match B, this does not establish an independent wrapper advantage. Groundedness is an evidence-output metric: absent trajectories cannot score as grounded, regardless of exact answer accuracy. State ablations establish only their measured condition/budget effects. Replay-storage and dream claims depend on their controls, not their existence.

Under the tested model, representation, supervision and compute budgets, the experiments do not establish a data-efficiency advantage or a broad structural-generalization advantage for the Mindscape architecture. Partial success at uncorrected local procedure execution is narrower than the proposed general hypothesis. The study is a rigorous measured outcome, including limitations and failed thresholds, rather than a claim that "Mindscape works."

## Limitations

One arithmetic domain, one generated final dataset, three seeds, hand-defined sequencing, unequal label/query amounts, a frozen indexed-feature language backbone and cached CPU inference limit general conclusions. Statistical significance against an answer-only condition does not remove these information differences. A second learned vertical and full language-model fine-tuning were deferred. Historical GPT/Gemini GSM8K references are not direct comparators.

## Future work

Use fully matched teacher information and interaction budgets across stronger sequence learners; test learned cursor/action planning; separate pretraining from categorical features and head capacity; design explicit carry-context holdouts before data generation; improve experiential exploration without providing full target trajectories; evaluate reuse rather than merely storage; benchmark cold compute; and add one small navigation backend with generic payload schemas. Any new work should pre-register its protocol and preserve these final artifacts.
