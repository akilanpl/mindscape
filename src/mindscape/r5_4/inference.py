"""Answer-blind inference boundary: no labels, reference arithmetic or filesystem I/O."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Operands:
    task_id: str
    left: str
    right: str

    def __post_init__(self):
        for value in (self.left, self.right):
            if not value.isascii() or not value.isdigit() or str(int(value)) != value:
                raise ValueError('Noncanonical unsigned operand')


def columns(task):
    left, right = task.left[::-1], task.right[::-1]
    for position in range(max(len(left), len(right))):
        yield (int(left[position]) if position < len(left) else 0,
               int(right[position]) if position < len(right) else 0)


def serialize(trace):
    digits = [str(row['digit']) for row in trace]
    if trace[-1]['carry_out']:
        digits.append('1')
    return ''.join(reversed(digits)).lstrip('0') or '0'


def rollout(task, predictor):
    """Cursor/padding/serialization supplied; every outgoing carry is predicted."""
    carry, hidden, trace = 0, None, []
    for position, (left, right) in enumerate(columns(task)):
        digit, carry_out, hidden = predictor(left, right, carry, hidden)
        if digit not in range(10) or carry_out not in (0, 1):
            raise ValueError('Invalid predicted class')
        trace.append(dict(position=position, left=left, right=right, carry_in=carry,
                          digit=digit, carry_out=carry_out))
        carry = carry_out
    return dict(task_id=task.task_id, horizon=len(trace), answer=serialize(trace), trace=trace)
