# Read-only compact checkpoint evaluation

`python -m mindscape.r56.compact_evaluation --root ROOT --manifest MANIFEST
--protected EVAL_JSON --checkpoint CHECKPOINT_JSON --checkpoint-sha256 SHA256
--seed SEED --condition CONDITION --output NEW_REPORT_JSON` is an independently
run evaluator entrypoint. It never constructs optimizers, updates weights, repairs
answers, selects checkpoints or issues scientific/product qualification.

It verifies the exact compact manifest, protected evaluation bytes, scientific
source locks, final checkpoint hash, seed/condition, 2,000 completed updates, frozen
forward policy and model configuration. Checkpoints use the existing bounded safe
JSON loader. Every original prediction and protected reference remains in reports.
Raw dev/test, unseen-context, candidate permutation, OOD physics, repeated physics
and 2/4/8/16/32-digit carry composition records are preserved with exact-accuracy or
physical MSE metrics separately. Reports do not combine unlike domain metrics.

The compact policy runs the full GRU/fusion path (`stateless=False`) with empty
memory on every independent local query for all three conditions. Composition
feeds only the original predicted carry to the next digit; repeated physics feeds
only the original predicted velocity. Hidden state resets per query. Production
`infer_problem` composition carries recurrent hidden state across digits, so this
compact state policy deliberately differs. Product integration needs independent
review of state-policy matching; the evaluator makes no episodic-memory claim.

`build_protected_evaluation` constructs evaluator-only artifacts and canonicalizes
incoming/carry_in context identity before enumerating unseen math cases. This
prevents a training context with nonzero `incoming` from being falsely counted as
unseen because an older generator read only `carry_in`. Held-out references never
enter model input. Source freezing must include this evaluator before any package
is approved. Filesystem separation alone does not qualify an OS access boundary.

Four bookkeeping tests pass. Exact-rule fixture predictors are explicitly labelled
engineering fixtures, not learned or product backends. No fitted checkpoint was
evaluated and no optimizer was constructed or updated during this implementation.

Supplemental protected coding permutations cover all four candidate target
positions for each novel AST structure, with independently hashed problem identities
and matching protected class references. Actual packed-graph observation hashes
report training duplicates and train/dev/test/held-out overlap without deleting
conflicting records. Full-addition and repeated-physics episodes are explicitly
excluded from a misleading one-step encoding comparison. Summary keys include
domain, axis and metric, so mathematics and coding accuracy are never pooled just
because their split names match. Checkpoint provenance also requires equal 2,000
`optimizer_steps`/`optimizer_updates` and the exact frozen training-data hash.
Six nontraining evaluator tests pass; fitted-checkpoint evaluation remains unrun.
