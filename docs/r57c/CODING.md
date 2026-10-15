# Focused coding audit: actual preserved-checkpoint measurements

Command (CPU only, zero updates):

```sh
PYTHONPATH=src ../py313/bin/python -m mindscape.r57c.coding_audit --output ../../outputs/r57-focused-correction-v1/evidence/coding --old-output ../../outputs/r57-neural-improvement-v1
```

The completed run evaluated all nine preserved R5.7 initial and final safe JSON
checkpoints on 160 diagnostic-development rows. No new final test was generated,
opened or used. The earlier final split is already scored historical evidence;
its distribution was audited read-only. Raw logits, winners, references, poisoned-
label replay checks, requested-name counterfactuals and scope breakdowns are saved
in `evidence/coding/`. Existing evidence directories cannot be overwritten.

CONFIRMED: each seed's historical 80-row coding final split requests lexical minimum
80/80 times, always suffix `_0`. Candidate positions are balanced 20 each, but requested
candidate identity is not. Thus position balance/permutation alone does not prevent
lexical rank shortcuts. Seed-43 graph initialization's earlier 100% cannot be treated
as independent evidence of learned name matching. On the new balanced diagnostic
its initial accuracy is 20%, final 75%.

| Seed | Graph on initial→final | Graph off initial→final | GRU initial→final |
|---|---:|---:|---:|
| 17 | 32.5%→77.5% | 32.5%→77.5% | 15%→87.5% |
| 29 | 27.5%→80% | 27.5%→80% | 20%→72.5% |
| 43 | 20%→75% | 20%→75% | 22.5%→77.5% |

Exact requested-name rule:100%. Lexical min/max and shortest/longest baselines:25%
each. These are explicit deterministic baselines, never neural accuracy.

Diagnostic identities include balanced requested candidate identities and positions,
cyclic permutations, novel ASCII/random/Unicode names and misleading shared prefixes/
suffixes. For fixed candidate order, requested-name changes span all four correct
classes. Raw counterfactual predicted-class counts and success are recorded by family.
All 18 checkpoints produce identical logits when only protected labels are poisoned;
model.prepare rejects labels, target_name and coding_index observation fields.

SUPPORTED: trained models use a useful request-conditioned signal beyond pure lexical
min, because balanced diagnostic accuracy substantially exceeds its25% baseline.
UNRESOLVED: these finite synthetic cases do not establish broad Unicode or identifier
matching generalization, and the training cause of remaining errors is not isolated.
The character-GRU/query-candidate score architecture contains no direct symbolic
correct-index field; its candidate-order equivariance alone does not prove matching.

Recommendation: fix development/final generation to balance requested identities as
well as positions and include fixed-set request counterfactuals and lexical distractors.
Do not change architecture on this audit alone. Freeze a genuinely independent final
split before correction selection; retain this set as diagnostic development only.
No training, scientific qualification, significance or product acceptance is claimed.
