"""Versioned model-visible features: no oracle values or future state."""
import numpy as np
from mindscape.models.encoding import features
from mindscape.environments.multiplication.claims import local_inputs

VERSION = "claims-features-1.0.0"
CONCEPT_SEED = "Decimal long multiplication: propose a*b+existing+carry; write units and carry."


def answer_features(view, structured=False):
    x = features(view["observation"]["operands"])
    if structured:
        x[18:22] = [len(str(abs(v))) / 6 for v in view["observation"]["operands"]] + [1, 1]
    return x


def policy_features(state, no_state=False, no_relation=False, no_goal=False):
    x = np.zeros(52)
    a, b, existing, carry = local_inputs(state)
    if not no_state:
        for i, value in enumerate((a, b, existing, carry)):
            x[10 * i + value] = 1
        x[40:47] = [state.position / 6, state.row_position / 6,
            len(state.left_digits) / 6, len(state.right_digits) / 6,
            int(state.phase == "flush_claim"), int(state.position == 0), int(state.row_position == 0)]
    x[47] = 0 if no_relation else 1
    x[48] = 0 if no_goal else 1
    return x


def claim_labels(value):
    # Identical head capacity to answer-only models; only first two heads encode a claim.
    return [value % 10, value // 10] + [0] * 7


def candidate_scores(backend, x):
    logits = backend.logits(np.asarray(x)[None, :])[0]
    raw = np.array([logits[v % 10] + logits[10 + v // 10] for v in range(100)])
    skip = raw.max() + logits[-1] - logits[-2]
    return np.r_[raw, skip]
