"""Canonical policy serialization v1 and bounded numeric feature adapter."""
from dataclasses import asdict
import math
import numpy as np

FORMAT_VERSION = "policy-state-1.0"
INPUT_SIZE = 52
OPERAND_DIGITS = 6
ANSWER_DIGITS = 8
ACTION_NAMES = ("multiply", "flush", "accumulate", "finish")


def serialize_state(problem, state, goal, valid_actions, previous_action=None):
    return {"format_version": FORMAT_VERSION, "problem": problem,
            "state": asdict(state), "goal": asdict(goal),
            "valid_actions": [asdict(a) for a in valid_actions], "previous_action": previous_action}


def features(operands, state=None, previous_action=None):
    if len(operands) != 2 or any(type(v) is not int for v in operands):
        raise ValueError("Expected integer operand pair")
    x = np.zeros(INPUT_SIZE, dtype=np.float64)
    for k, value in enumerate(operands):
        ds = str(abs(value))[::-1]
        if len(ds) > OPERAND_DIGITS:
            raise ValueError("Model operand width exceeded")
        offset = k * 9
        for i, digit in enumerate(ds):
            x[offset + i] = int(digit) / 9
        x[offset + 6] = int(value < 0)
        x[offset + 7] = len(ds) / OPERAND_DIGITS
        x[offset + 8] = 1
    if state is not None:
        # Exclude phase and valid-action identities: either would reveal the target.
        left, right = state["left_digits"], state["right_digits"]
        pos, row = state["position"], state["row_position"]
        x[18:28] = [pos / OPERAND_DIGITS, row / OPERAND_DIGITS,
                    len(left) / OPERAND_DIGITS, len(right) / OPERAND_DIGITS,
                    state["carry"] / 10, int(pos == len(left)), int(row == len(right)),
                    int(pos == 0), int(row == 0), state["sign"]]
        for j, key in enumerate(("row_value", "accumulated")):
            value = state[key]
            for i in range(ANSWER_DIGITS):
                x[28 + j * ANSWER_DIGITS + i] = (abs(value) // 10 ** i % 10) / 9
        x[44] = left[pos] / 9 if pos < len(left) else 0
        x[45] = right[row] / 9 if row < len(right) else 0
        x[46] = math.log1p(abs(state["row_value"])) / 20
        x[47] = 1
    if previous_action is not None:
        x[48 + ACTION_NAMES.index(previous_action)] = 1
    return x


def answer_labels(answer):
    if type(answer) is not int or len(str(abs(answer))) > ANSWER_DIGITS:
        raise ValueError("Answer width exceeded")
    return [abs(answer) // 10 ** i % 10 for i in range(ANSWER_DIGITS)] + [int(answer < 0)]


def decode_answer(labels):
    value = sum(int(digit) * 10 ** i for i, digit in enumerate(labels[:-1]))
    return -value if labels[-1] else value
