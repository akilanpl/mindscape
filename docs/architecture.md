# Architecture

The state contains operand entities, their multiplication relationship, decimal
positions, carry, row accumulator, accumulated partial rows, result sign and goal.
Each step records state_before, action, event, actual result and state_after.
The original observation is retained on the trajectory. Inference, prediction,
actual result and hypothetical result are distinct EvidenceKind values; execution
accepts only actual results. Prediction/simulation are not implemented.

For each multiplier digit, process multiplicand digits from least significant to
most significant, flush carry, accumulate the shifted partial row, then advance.
A separate finish action materializes the signed answer. Goal evaluation checks
completion and the independent product oracle. Verification reconstructs and
replays from the observation, checking every complete transition and claimed answer.
Shared replay logic catches corruption but is not an independently implemented
arithmetic proof; exhaustive small cases and independent oracle tests mitigate
shared implementation bugs. No memory, learning or planning exists yet.
