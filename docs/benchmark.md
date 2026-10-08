# Benchmark specification (not implemented)

Accuracy = exact correct answers / examples; OOD accuracy uses withheld operand
length structures. Grounded answer rate = answers supported by complete valid
trajectories / answers. Unsupported rate = 1 - grounded rate. Transition validity
= valid transitions / transitions. Goal success = successful episodes / episodes.
Empty denominators are undefined, not automatically perfect.

Initial targets: accuracy 95%, OOD 90%, groundedness 95%, transition validity 99%,
goal success 95%. These are goals, not findings. N*(alpha) is minimum training
examples reaching alpha; DER = N*_baseline / N*_Mindscape. Unreached thresholds
remain censored/undefined. No DER is available.

Proposed train structures: 1x1, 2x1, 2x2; OOD: 3x2 and 3x3. Keep canonical
operand pairs disjoint across splits, including swapped pairs. Record zero/sign
and carry distributions separately. Benchmark generation/evaluation is next phase.
