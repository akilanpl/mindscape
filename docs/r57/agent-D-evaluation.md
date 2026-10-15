# Agent D independent exploratory evaluation

`evaluate(predictor, rows)` receives immutable rows with identity, family, problem,
labels and optional scope. Predictor receives a deep-copied problem only, excluding
verifier targets/labels; it returns normal tensor heads or decoded math_digit,
math_carry, coding_index, physics_delta fields. Raw predictions are never repaired.
Stable normalized problem identities reject duplicates, including incoming-carry
aliases. All conditions within a seed must use the identical frozen row inventory.

Results preserve every case, reference, original prediction, correctness/error and
family identifier. Math reports joint digit/carry and separate submetrics; coding
reports exact, target-position and scope breakdowns (data generator must label
permutation/alpha-renaming groups); physics reports MSE, MAE and axis-sign accuracy.
Force-plus-one perturbations independently report monotonicity, strict increase and
reference delta. Constant zero passing nondecrease is explicitly not physical learning.

`fit_physics_baselines(training_rows)` fits a training-only constant and raw-feature
linear SVD baseline, alongside zero and explicit analytic Fdt/m reference. It does
not supply analytic features to a neural model. `analytic` is always a separate
rule baseline, never merged into neural accuracy. Paired family bootstrap requires
identical unique case IDs and cluster membership; intervals are descriptive, with
raw differences and no statistical-significance claim. Use problem-family clusters,
not independently resampled candidate permutations of the same problem.

`evaluate_r56_checkpoint` safely loads original JSON checkpoints and evaluates CPU
inference without optimizer updates. Call it in a separate evaluator process whose
PYTHONPATH points first to the preserved original source; the helper refuses an
incorrect original neural module path. This prevents a cached R5.7 class replacing
R5.6 architecture. For each preserved fit record checkpoint/source/data hashes,
condition, seed and phase initial/final; aggregate only matching final row IDs.
Initial checkpoint availability must be reported rather than fabricated.

Final evaluator records must stay outside exploratory training inputs. The new final
split must be generated/frozen before candidate modifications are scored, with its
hash and source record, and only evaluated after the model candidate is frozen.
Do not return final scores for architecture/hyperparameter selection. Original
protected official R5.6 evaluator data is not accessed by this module.

Current status: four CPU tests pass. No final test scores or checkpoints have been
loaded or scored in this workstream pending candidate freeze. No MPS runs, updates,
significance, scientific qualification or product acceptance are claimed here.
