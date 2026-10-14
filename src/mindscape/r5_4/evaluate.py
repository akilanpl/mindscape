"""Predictions first. Free inference consumes operands and predictor only."""
import json
from pathlib import Path
from .inference import rollout, Operands


def predict_rows(tasks, predictor, model_id, condition, seed):
    for task in tasks:
        result = rollout(Operands(task['task_id'], task['left'], task['right']), predictor)
        yield dict(result, model_id=model_id, condition=condition, seed=seed,
                   split=task['split'], family=task['family'], tensor_metadata={
                       'parameter_dtype': 'float32' if condition == 'neural' else 'not_applicable',
                       'recorded_classes_dtype': 'json_integer', 'trace_shape': [result['horizon'], 6]})


def local_diagnostics(tasks, predictor, model_id, condition, seed):
    # Called after all free predictions are sealed; never feeds outcomes into free rollout.
    from .data import reference, operand
    rows = []
    for task in tasks:
        hidden, trace = None, []
        for position, row in enumerate(reference(operand(task))):
            left, right, carry = row[:3]
            digit, carry_out, hidden = predictor(left, right, carry, hidden)
            trace.append(dict(position=position, left=left, right=right, carry_in=carry,
                              digit=digit, carry_out=carry_out))
        rows.append(dict(task_id=task['task_id'], model_id=model_id, condition=condition,
                         seed=seed, mode='teacher_forced', trace=trace))
    for left in range(10):
        for right in range(10):
            for carry in (0, 1):
                digit, carry_out, _ = predictor(left, right, carry, None)
                rows.append(dict(task_id=f'context-{left}-{right}-{carry}', model_id=model_id,
                    condition=condition, seed=seed, mode='context_probe', context=[left, right, carry],
                    digit=digit, carry_out=carry_out))
    return rows
