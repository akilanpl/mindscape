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
