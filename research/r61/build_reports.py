"""Render R6.1 diagnostic evidence without asserting unmeasured recovery."""
import argparse,json,statistics
from pathlib import Path
from research.r6 import common

def run(root):
    root=Path(root);d=common.read(root/'diagnostics-final/diagnostics.json');c=common.read(root/'correction-zero-update-evidence.json')
    tests=(root/'evidence/tests-final.txt').read_text().strip().splitlines()[-1]
    checks=d['checks']+[{'name':'input_target_and_structured_field_blind_prediction','status':'PASS','evidence':'correction-zero-update-evidence.json: all four fit/domain combinations, both pooling modes; removing targets and structured fields leaves predictions equal.'},{'name':'early_state_summary_regression_before','status':'FAIL','evidence':'evidence/regression-before.txt: original final-state behavior intentionally fails the proposed early-state information-retention assertion; not an unexpected suite failure.'},{'name':'early_state_summary_regression_after','status':'PASS','evidence':'evidence/tests-final.txt: corrected masked mean retains early-state signal and ignores padding.'},{'name':'corrected_frozen_weight_learning_accuracy','status':'FAIL','evidence':'0/16 train examples correct in each sampled domain/fit with either pooling mode. A mechanism change has not recovered learned answers.'},{'name':'recovery_training_and_generalization','status':'INCONCLUSIVE','evidence':'No optimizer updates or new cloud submission authorized; tiny-set memorization and fresh-case generalization not tested by fitting.'}]
    common.write(root/'R61_DIAGNOSTIC_TESTS.json',{'schema':'mindscape-r61-diagnostic-results-v1','optimizer_updates':0,'device':'cpu','checks':checks,'automated_regression_suite':tests,'full_evidence':'diagnostics-final/diagnostics.json','frozen_weight_correction_evidence':'correction-zero-update-evidence.json','raw_predictions':'diagnostics-final/predictions.json','interpretation':'PASS on implementation checks does not establish learning. Before-regression FAIL is deliberate; semantic failures are genuine.'})
    table=['| Checkpoint / domain | Original early/late gradient | Pooled early/late gradient |','|---|---:|---:|']
    for fit,entry in c['fits'].items():
        for domain in ('mathematics','coding'):table.append(f"| {fit} / {domain} | {entry['baseline'][domain]['median_early_to_late_input_gradient']:.6g} | {entry['pooled'][domain]['median_early_to_late_input_gradient']:.6g} |")
    gradient_table='\n'.join(table)
    rootcause=f'''# R6.1 collapse root-cause investigation

## Completed result

Zero-update diagnostics locate **weak problem conditioning through a final-state encoder bottleneck, followed by prior-dominated autoregressive decoding**. There is no evidence of target/codec corruption, lost input bytes, disconnected text parameters, incorrect checkpoint loading, or an evaluator that fabricates constant outputs. This is a measured proximate mechanism and a justified intervention target, not proof of a unique ultimate cause or a successful learning recovery.

All diagnostics ran on CPU on the development Mac with **zero optimizer updates**. Forward/backward reachability probes changed no model parameters or buffers. No R6.1 Kaggle job was submitted. The proposed physics follow-up is on hold.

## Evidence separating the domains

Raw math and coding selected checkpoints produce only `-12` and `12` respectively across all 512 training examples/domain, scoring 2/512 each. The normalized checkpoint still scores math 2/512 and coding 1/512, despite lower teacher-forced CE. Training-set failure rules out explaining collapse solely as unseen-case difficulty.

The math training set has 374 distinct answers and entropy {d['target_distributions']['mathematics']['train']['entropy_bits']:.4f} bits; coding has 324 distinct answers and entropy {d['target_distributions']['coding']['train']['entropy_bits']:.4f} bits. The best train-derived constant gets only 5/512 math and 7/512 coding. Collapsed strings are not majority full answers; digit `1`, digit `2`, EOS and sign frequencies can support low-loss answer-prefix priors. This observation is consistent with a shortcut, but does not prove every decoded token was chosen because of its marginal frequency.

R6 uses byte embeddings and a single 48-wide forward GRU for text. Only its last state reaches fusion; structured operands, expression trees, ASTs and binding fields do not enter the model. Numeric graph properties are masked out for math/coding. Input bytes are distinguishable, but essential early information is very weakly coupled to answer loss. Short math strings and longer coding programs therefore require separate interpretation.

For the raw coding checkpoint, changing a loop body binding changes the exact result from 6 to 16 but changes context norm by only about 0.000094; the output stays `12`. Changing the final loop-bound argument changes context by about 0.728 but still produces `12`. Prefix syntax and bindings are especially attenuated relative to suffix arguments. The normalized checkpoint is more input-sensitive but still wrong, and a semantics-preserving rename changes its answer on the example. Math operand/operator changes produce distinct contexts but the same wrong answer. Nonzero context distance alone is not learned semantics.

## Causal probes and correction

The table uses the first16 original training cases/domain, per-byte answer-loss embedding gradients, and the median ratio of mean gradient norm in the first versus last input third. The same frozen checkpoint tensors are used for both pooling modes.

{gradient_table}

This intervention demonstrates that final-only aggregation caused the observed gradient attenuation route: supplying all valid recurrent states restores direct early-token gradient access without updating any weights. It does **not** prove the gradients encode the correct arithmetic/program algorithm. In fact both modes score 0/16 on each sampled domain/fit before any retraining.

The corrected model uses a masked mean of the encoder's recurrent output states instead of its final state. It adds no parameters and changes no input bytes, targets, decoder, loss or physical equations. All other R6 components remain the original imported baseline. Masked pooling can dilute useful suffix information and retains a fixed-width bottleneck; it is a candidate intervention, not a universal solution. No extra speculative representation changes were added.

## Optimization and evaluator exclusions

Both historical runs recorded 600 actual updates at LR0.001, nonzero finite text-gradient norms, and reduced teacher-forced CE. Per-parameter selected-vs-initial weight changes and current per-layer gradients are saved. PAD is ignored correctly; manual tokenwise NLL equals worker CE without broadcasting. Text-specific backward excludes expected unused physical graph/head parameters rather than falsely calling those dead text components.

A selected-checkpoint, first16/domain shared-encoder probe finds raw physics gradient norm about21.86 versus math0.415 and coding0.200; normalized physics0.682 versus math0.348 and coding0.328. Raw loss imbalance is a plausible contributor. These are local checkpoint probes, not a reconstruction of the training trajectory; normalized text still collapses, so physics scaling alone was insufficient. The recovery pilot isolates tiny-set text learnability in both matched arms; it is not a physics follow-up.

A second greedy implementation, safe checkpoint reloads, target-blind inference and 768 exact prediction comparisons with the original separate CPU cloud evaluator all agree. Greedy decoding exposes the failed conditional model; there is no evidence that sampling or a solver would establish native reasoning. High teacher-forced token accuracy does not imply correct free-running answers.

## Remaining uncertainty

No fitting is allowed until explicit budget approval. We cannot yet show tiny-set memorization, causally attribute improved accuracy to pooling, determine whether 600 historical updates were enough, or rule out width/optimization/decoder exposure effects. One seed and a narrow gradient sample do not establish generality. The root-cause account is supported at the information-flow level; learning-level sufficiency remains INCONCLUSIVE.

Automated checks: **{tests}**. These checks are not a model-learning score. See R61_DIAGNOSTIC_TESTS.json for explicit PASS/FAIL/INCONCLUSIVE outcomes and diagnostics-final for full measurements.
'''
    (root/'R61_COLLAPSE_ROOT_CAUSE.md').write_text(rootcause)
    for domain,name in [('mathematics','MATH'),('coding','CODING')]:
        dist=d['target_distributions'][domain];lines=[f'# R6.1 {domain} dataflow audit','', '## Input and representation','', 'ASCII bytes map bijectively to IDs byte+1; PAD0 is distinct. Inputs over256 bytes are rejected, not truncated. Packed GRU lengths exclude padding and preserve row order. GRU recurrence supplies order information, but no explicit place-value encoding or syntax/operand graph is consumed. Structured fields are dataset verification metadata; removing them leaves inference unchanged.', '',f"Observed input maxima by split: { {s:x['max_input_bytes'] for s,x in dist.items()} }. All are below the input bound. Distinct sampled contexts are not evidence of retained semantics.", '', 'For math, signed multi-digit addition/subtraction/multiplication span distinct operand combinations; division, fractions, decimals and algebra were not trained. For coding, functions, loops and conditionals are serialized as bytes; variable names, operators and indentation survive the codec, but there is no AST/binding encoder. The raw model strongly attenuates early program-body features.', '', '## Targets, masks and loss','', 'Integer string vocabulary: PAD0, BOS1, EOS2, minus3, digits4..13. Targets append EOS; decoder inputs shift by one with BOS. PAD is ignored in token CE, while EOS is scored. All original2304 records were independently oracle-verified; every text target roundtrips exactly. Maximum target lengths are below12. There is no scalar-label broadcast or vocabulary-class indexing mismatch. Current per-domain backward reaches every relevant text parameter tensor with finite nonzero norm; unused physics-only components are expected.', '',f"Training labels: {dist['train']['distinct_answers']} distinct; entropy {dist['train']['entropy_bits']:.6f} bits; train-majority answer `{dist['train']['training_majority_answer']}`, accuracy {dist['train']['training_majority_accuracy']:.6f}. Hindsight best constants are diagnostic upper bounds only and never deployable baselines. Token counts below use token IDs, not digit labels:", '', '```json',json.dumps(dist['train']['target_token_counts'],indent=2),'```','', '## Measured conditioning and decoding','']
        for fit,m in d['models'].items():
            x=m['domains'][domain];ce=x['teacher_forced_cross_entropy'];g=x['gradients']
            lines += [f"### {fit}",'',f"32/32 sampled contexts distinct. Mean feature std {x['context_feature_std_mean']:.6g}; median pairwise distance {x['context_pairwise_distance_median']:.6g}. Context entries with |tanh|>0.95: {x['context_tanh_abs_gt_095_fraction']:.4%}. This partial saturation does not by itself diagnose total dead activations.",'',f"Teacher-forced CE: actual {ce['actual']:.6f}; reversed across problems {ce['reversed']:.6f}; domain-mean context {ce['domain_mean']:.6f}; zero context {ce['zero']:.6f}. Greedy outputs changed on reversal for {x['greedy_changed_on_reversal']:.2%}, and on mean replacement for {x['greedy_changed_on_mean_replacement']:.2%}. Domain information and some magnitude/sign cues are therefore used; absence of all input influence would be an overclaim.",'',f"Native train exact: {x['splits']['train']['selected_correct']}/512; output counts: {x['splits']['train']['selected_output_counts']}. Original initialization correct: {x['splits']['train']['initial_correct']}/512. Initial outputs: {x['splits']['train']['initial_output_counts']}.",'','| Intervention | Reference | Native output | Context distance from base |','|---|---:|---|---:|']
            lines += [f"| {p['kind']} | {p['reference']} | {p['prediction']['raw_text']} | {p['context_distance_from_base']:.6g} |" for p in x['probes']]
            lines += ['',f"Manual NLL difference: {g['manual_loss_abs_delta']:.6g}; original-worker CE difference: {g['original_worker_loss_abs_delta']:.6g}. Model state unchanged after backward. All per-layer gradient and initial-to-selected weight norms are in diagnostics-final/diagnostics.json.",'']
        lines += ['## Evaluation and leakage exclusions','', 'No exact or canonical overlap across any original train/dev/final split pair. Saved row IDs, inputs and targets remain paired through70 balanced-sampler batches (1680 cases). Original and independently implemented decoder predictions agree; separately reloaded checkpoints agree. Fresh local recomputation matches768 original cloud CPU text predictions across both selected fits. Answer-level perturbations fail; invariant wrong answers do not earn a semantics pass.', '', 'The R6.1 tiny dataset is freshly generated, independently checked and deduplicated against all R6 splits and every other R6.1 split. Its intentionally varied labels are diagnostic data; no fitting or new generalization score has been produced. The proposed private CPU evaluation must keep protected files absent from trainers and report original R6 predictions, paired controls, constants and external deterministic solvers separately.']
        (root/f'R61_{name}_DATAFLOW_AUDIT.md').write_text('\n'.join(lines)+'\n')
    (root/'R61_CORRECTION_DIFF.md').write_text(f'''# R6.1 targeted correction

Only intervention: replace final-only encoder summarization with a padding-masked mean of all forward-GRU output states. The original R6 source remains unchanged in research/r6. New research/r61/model.py subclasses the baseline and supplies the alternative summary through a scoped GRU hook. Both models have54,014 parameters and identical state-dict schemas; initialization tensors match exactly at seed61117. Baseline mode delegates directly to original R6 encode. The hook is removed in finally, including failures; this implementation assumes single Python-thread model invocation and is not advertised as concurrent-serving infrastructure.

```python
# Original R6
_, state = self.sequence(packed)
sequence = state[-1]

# R6.1 encoder summary (same GRU parameters)
outputs, state = self.sequence(packed)
padded, _ = pad_packed_sequence(outputs, batch_first=True)
mask = arange(padded.shape[1])[None, :] < lengths[:, None]
sequence = (padded * mask[..., None]).sum(1) / lengths[:, None]
```

Actual implementation moves masks/lengths to the correct device. No analytic solver enters prediction. Decoder, answer vocabulary, CE, Adam configuration, width and hidden graph are unchanged within the proposed pair. Physics is excluded from both tiny text arms to test math/coding learning capability; comparison to the original multitask R6 checkpoint is descriptive, not an isolated pooling-effect comparison.

The regression constructs equal last states with a distinguishable early state. Original last-mode summarization loses that difference and deliberately fails the new assertion (evidence/regression-before.txt). Masked mean passes and has a nonzero early-state gradient. Padding is independently tested. Full current suite: **{tests}**.

Frozen-checkpoint gradient evidence:

{gradient_table}

No retraining occurred. Both modes remain 0/16 correct for each frozen-weight sampled fit/domain. This is a repair of the measured gradient-access mechanism; successful collapse recovery remains unproven. Matching initial tensors and equal run budgets are required for the proposed next test.

Prepared pilot worker refuses missing or mismatched config-bound approval records and refuses the development Mac before constructing an optimizer. Its CUDA loop has not been executed or validated on CUDA. Each proposed arm runs the same32-case full batch for up to1200 updates, with no best-checkpoint selection or automatic retry. Timed stops are checked at operation boundaries, so an in-flight CUDA operation can exceed a wall-clock boundary; update counting is exact. Failure stops the campaign for review.
''')
    print(json.dumps({'reports_written':5,'optimizer_updates':0,'learning_recovery':'UNPROVEN'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
