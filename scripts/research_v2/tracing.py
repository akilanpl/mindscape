"""Observed numeric-policy traces; reconstructed activations are explicitly labelled."""
import hashlib
import time
from dataclasses import asdict, is_dataclass

import numpy as np

from mindscape.data.backends import get_backend
from mindscape.models.learned import MindscapeModel


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def memory_snapshot(memory):
    result = {k: plain(getattr(memory, k, None)) for k in
              ('current_state', 'current_goal', 'current_observation')}
    for name in ('action_history', 'event_history', 'result_history', 'trajectory_so_far'):
        values = getattr(memory, name, [])
        result[name] = {'length': len(values), 'latest': plain(values[-1]) if values else None}
    return result


def trace_case(backend, example, *, mask=True, state_features=True, previous_action_features=True):
    actor = MindscapeModel(backend, mask=mask, max_steps=256, timeout_seconds=float('inf'))
    events = []
    original_logits = backend.logits
    started = time.perf_counter()
    def emit(component, computation, value):
        events.append({'index': len(events), 'elapsed_seconds': time.perf_counter()-started,
                       'component': component, 'computation': computation, 'observed': plain(value)})
    def observed_logits(x):
        used = np.asarray(x).copy()
        if not state_features:
            used[:, 18:48] = 0
        if not previous_action_features:
            used[:, 48:52] = 0
        began = time.perf_counter()
        scores = original_logits(used)
        forward_seconds = time.perf_counter()-began
        # The frozen function returns logits only. This intermediate is reconstructed,
        # not falsely described as captured from its stack or as a causal explanation.
        hidden = np.tanh(used @ backend.weights['w1'] + backend.weights['b1'])
        shifted = scores-scores.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)/np.exp(shifted).sum(axis=-1, keepdims=True)
        emit('NumpyMLP.logits', 'learned neural forward', {
            'raw_features': np.asarray(x), 'features_used': used, 'input_shape': list(used.shape),
            'weight_shapes': {k: list(v.shape) for k, v in backend.weights.items()},
            'hidden_shape': list(hidden.shape), 'logits_shape': list(scores.shape),
            'logits': scores, 'derived_uncalibrated_softmax': probs,
            'reconstructed_hidden_summary': {'mean': float(hidden.mean()), 'std': float(hidden.std()),
                                             'min': float(hidden.min()), 'max': float(hidden.max())},
            'hidden_provenance': 'Recomputed from observed input and checkpoint; not stack-captured',
            'device': 'cpu', 'dtype': str(used.dtype), 'forward_seconds': forward_seconds,
            'forward_and_summary_seconds': time.perf_counter()-began})
        return scores
    backend.logits = observed_logits
    for name in ('reset', 'update'):
        original = getattr(actor.memory, name)
        def wrapped(*args, _original=original, _name=name):
            before = memory_snapshot(actor.memory)
            value = _original(*args)
            emit('WorkingMemory.'+_name, 'deterministic control',
                 {'before': before, 'after': memory_snapshot(actor.memory)})
            return value
        setattr(actor.memory, name, wrapped)
    original_select = actor.select_action
    def observed_select(representation, operands):
        selected, raw = original_select(representation, operands)
        emit('MindscapeModel.select_action', 'deterministic selection after neural forward',
             {'valid_actions': representation['valid_actions'], 'selected': selected,
              'raw_neural_action': raw, 'legal_mask_enabled': mask})
        return selected, raw
    actor.select_action = observed_select
    model_input = example.view('experiential')
    emit('MindscapeModel.predict', 'input boundary', {'model_input': model_input,
         'normalization': 'Frozen numeric features: sign, reverse decimal digits and bounded state scalars',
         'tokenization': {'status': 'not applicable', 'reason': 'Numeric MLP; no text tokenizer'}})
    try:
        prediction = actor.predict(model_input)
        valid, goal, error = get_backend(example.environment).assess(
            example, prediction.answer, prediction.trajectory)
        result = {'prediction': plain(prediction), 'independent_verification':
                  {'trajectory_valid': valid, 'goal_reached': goal, 'error': error},
                  'score': bool(goal), 'execution_status': 'finished', 'episode': plain(actor.last_episode)}
    except (RuntimeError, ValueError, ArithmeticError) as exc:
        result = {'prediction': None, 'score': False, 'execution_status': 'executed_failure',
                  'failure': {'type': type(exc).__name__, 'message': str(exc)},
                  'episode': plain(actor.last_episode),
                  'independent_verification': {'trajectory_valid': None, 'goal_reached': False,
                                               'error': 'Executed policy exception; no invented trajectory'}}
    finally:
        backend.logits = original_logits
    return {'schema': 'mindscape-computation-trace-2.0', 'task_id': example.example_id,
            'environment': example.environment, 'exact_model_input': model_input,
            'configuration': {'legal_action_mask': mask, 'state_features': state_features,
                              'previous_action_features': previous_action_features,
                              'max_steps': 256, 'wall_clock_cutoff': None},
            'model': {'identifier': actor.identifier, 'backend': backend.identifier,
                      'parameter_count': backend.parameter_count, 'adapter': None,
                      'seed': backend.seed, 'weights_sha256':
                      {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in backend.weights.items()}},
            'events': events, 'result': result, 'wall_seconds': time.perf_counter()-started,
            'unimplemented_here': ['text tokenizer', 'learned entity/relationship extraction',
                                    'planning search', 'dreaming', 'feedback learning', 'recovery policy'],
            'interpretation': 'Numeric-core observation, not coding-foundation inference or causal activation attribution'}
