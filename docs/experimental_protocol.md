# Experimental protocol

Freeze datasets and holdout rules before fitting. Record random seed, complete
configuration, model identifier/parameter count, dataset version/hash/size, regime,
split, hyperparameters, checkpoint, software versions, hardware and runtime.
Create unique result directories; never overwrite. Save config, metrics,
predictions and summary. Retain failed runs. Use 3–5 seeds where practical,
mean and standard deviation and appropriate confidence intervals. Raw graph data
must remain available. Ablations must remove operational components rather than
only toggling labels. Historical references require exact protocol compatibility;
no comparison across unrelated benchmark tasks is valid.

## Phase 2 operational rules

Freeze dataset_version, generator_version, manifest content hash and split structures
before fitting. The current development dataset tests infrastructure only and may
inform infrastructure debugging; do not treat it as an untouched final research test.
Allocate new versioned final datasets before neural experiments. Any subsequent
validation-informed design or split revision must be recorded.

Train only from a recorded nested subset and regime-specific training view. Regime C
receives concepts/limited examples/environment feedback rather than full trace labels;
this phase provides its data view, not a learning algorithm. Record procedural
knowledge access as an experimental condition. A reference algorithm is not a fair
learned baseline and its perfect arithmetic score is not evidence of data efficiency.

Evaluate baseline, structured baseline and Mindscape on identical IID/OOD splits,
backbone/parameter count, decoding, epochs and compute where practical. Repeat 3–5
seeds for final studies. Keep accuracy, groundedness, trajectory validity, goal success,
OOD accuracy, runtime and data efficiency separate. N* uses observed sizes only and
unreached thresholds remain censored. Retain failures and every prediction file.
Run directories record null for unmeasured training/checkpoint fields; future training
must populate them. Final test sets must not drive tuning.

## First learned development protocol

Before evaluation, record fairness.json: same dataset hash/subset IDs, train/validation
and IID/OOD sets, same hidden backbone, seeds, optimizer steps, batch size, learning
rate, evaluator and validation-loss selection. Head sizes and supervision count differ.
Policy-only access to exact arithmetic transitions and singleton action masks are
major confounds. Include unmasked policy runs and raw action validation diagnostics;
never infer architecture superiority from the forced masked execution.

The fixed schedule uses budgets 50/100/250; seeds 0/1/2 at 50, seed 0 at larger budgets.
These are development results on development_v1, not final untouched thesis results.
The protocol and implementation are committed before evaluation. No model/prompt
or test-set change is made in response to held-out scores. Match or explicitly vary
procedural knowledge/tool access in later experiments. Do not compute DER from the
masked algorithm's guaranteed accuracy. Historical reference scores remain unpopulated.

## Controlled numerical-claim revision

The revised study is documented in research_protocol.md and milestone_04.md.
Prior singleton/oracle-assisted results are exploratory, not current evidence.
Primary numerical-claim execution writes model proposals without correction;
Regime C uses training-only scalar feedback and positive replay. All four conditions
share capacity and the evaluator; remaining procedural/supervision/compute differences
are explicit. All reported thresholds are censored; no data-efficiency advantage
has been established. Use the research-specific CLI/config files for this study.
