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
