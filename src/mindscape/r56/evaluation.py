"""Bounded uncorrected composition/OOD evaluation and transfer preparation."""
from __future__ import annotations
from .curriculum import digest


def axis_datasets():
    rows = []
    for depth in (2, 4, 8):
        for variant in (0, 1):
            left = [9 if variant == 0 else (index * 3 + 2) % 10 for index in range(depth)]
            right = [1 if variant == 0 else (index * 7 + 4) % 10 for index in range(depth)]
            row = {'axis': 'unseen-composition', 'domain': 'math', 'depth': depth,
                   'left_digits_lsd_first': left, 'right_digits_lsd_first': right}
            row['identity'] = digest(row)
            rows.append(row)
        for axis, mass, force in [('composition', 2.0, [3.0, -2.0]),
                                  ('ood-physics', 20.0, [20.0, -20.0])]:
            row = {'axis': axis, 'domain': 'physics', 'depth': depth, 'mass': mass,
                   'initial_velocity': [1.0, -1.0], 'force': force, 'dt': .5}
            row['identity'] = digest(row)
            rows.append(row)
    return {'version': 'r56-axes-v1', 'episodes': rows,
            'interpretation': 'Preparation examples only; original untrained predictions establish no generalization.'}


def evaluate_episode(episode, predictor):
    """predictor(problem)->original decoded dict. No correction or oracle input."""
    original = dict(episode)
    identity = original.pop('identity', None)
    if identity != digest(original) or not 1 <= episode['depth'] <= 8:
        raise ValueError('Invalid frozen episode identity/depth')
    records, status = [], 'completed'
    if episode['domain'] == 'math':
        carry, reference_carry, digits = 0, 0, []
        left, right = episode['left_digits_lsd_first'], episode['right_digits_lsd_first']
        if len(left) != episode['depth'] or len(right) != episode['depth']:
            raise ValueError('Digit depth mismatch')
        for step, (a, b) in enumerate(zip(left, right)):
            problem = {'domain': 'math', 'operation': 'add_digit', 'left': a,
                       'right': b, 'incoming': carry, 'concept_ids': [0]}
            prediction = predictor(problem)
            total = a + b + reference_carry
            reference_carry = total // 10
            digit, predicted_carry = prediction.get('math_digit'), prediction.get('math_carry')
            valid = type(digit) is int and 0 <= digit <= 9 and type(predicted_carry) is int and predicted_carry in (0, 1)
            records.append({'step': step, 'input': problem, 'original_prediction': prediction,
                'reference': {'math_digit': total % 10, 'math_carry': reference_carry},
                'exact_step': valid and digit == total % 10 and predicted_carry == reference_carry,
                'repaired': False})
            if not valid:
                status = 'invalid-prediction'; break
            digits.append(digit); carry = predicted_carry
        expected = sum((a + b) * 10 ** i for i, (a, b) in enumerate(zip(left, right)))
        actual = sum(d * 10 ** i for i, d in enumerate(digits)) + carry * 10 ** len(digits) if status == 'completed' else None
        metric = {'exact_composition': actual == expected, 'expected': expected,
                  'original_result': actual, 'absolute_error': abs(actual - expected) if actual is not None else None}
    elif episode['domain'] == 'physics':
        import math
        velocity = list(episode['initial_velocity'])
        delta = [f * episode['dt'] / episode['mass'] for f in episode['force']]
        for step in range(episode['depth']):
            problem = {'domain': 'physics', 'operation': 'force', 'mass': episode['mass'],
                'velocity': list(velocity), 'force': list(episode['force']),
                'dt': episode['dt'], 'concept_ids': [2]}
            prediction = predictor(problem)
            proposed = prediction.get('physics_delta')
            valid = isinstance(proposed, list) and len(proposed) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in proposed)
            if valid:
                velocity = [v + d for v, d in zip(velocity, proposed)]
            reference = [v + (step + 1) * d for v, d in zip(episode['initial_velocity'], delta)]
            records.append({'step': step, 'input': problem, 'original_prediction': prediction,
                'reference_velocity': reference, 'original_velocity': list(velocity),
                'accumulated_squared_error': sum((a - b) ** 2 for a, b in zip(velocity, reference)) / 2 if valid else None,
                'repaired': False})
            if not valid:
                status = 'invalid-prediction'; break
        metric = {'final_mse': records[-1]['accumulated_squared_error']}
    else:
        raise ValueError('Unsupported episode domain')
    return {'identity': identity, 'axis': episode['axis'], 'depth': episode['depth'],
            'status': status, 'records': records, 'metric': metric,
            'scientifically_qualified': False}


CORE_PREFIXES = ('features.', 'types.', 'relations.', 'concepts.', 'domains.', 'action.',
                 'message.', 'update.', 'norm.', 'working_memory.', 'state_fusion.')


def transfer_preparation(source_state, target_initial_state, source_ids, target_train_ids,
                         target_holdout_ids, source_domain, target_domain):
    """Return core-only initialization and provenance, never run transfer training."""
    if source_domain == target_domain or source_domain not in ('math', 'coding', 'physics') or target_domain not in ('math', 'coding', 'physics'):
        raise ValueError('Transfer requires distinct supported domains')
    for identities in (source_ids, target_train_ids, target_holdout_ids):
        if not identities or any(not isinstance(x, str) or len(x) != 64 for x in identities):
            raise ValueError('Actual exposure identity hashes required')
    if set(source_ids) & (set(target_train_ids) | set(target_holdout_ids)) or set(target_train_ids) & set(target_holdout_ids):
        raise ValueError('Exposure/holdout identity overlap')
    if set(source_state) != set(target_initial_state):
        raise ValueError('Model key mismatch')
    result, core_keys, head_keys = {}, [], []
    for key, target in target_initial_state.items():
        source = source_state[key]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError('Model tensor schema mismatch')
        is_core = key.startswith(CORE_PREFIXES)
        result[key] = (source if is_core else target).detach().clone()
        (core_keys if is_core else head_keys).append(key)
    if not core_keys or not head_keys:
        raise ValueError('Need both compatible shared core and independently initialized heads')
    head_hash = digest({key: target_initial_state[key].detach().cpu().tolist() for key in head_keys})
    return result, {'source_domain': source_domain, 'target_domain': target_domain,
        'source_exposure_ids': list(source_ids), 'target_train_ids': list(target_train_ids),
        'target_holdout_ids': list(target_holdout_ids), 'copied_core_keys': core_keys,
        'unchanged_head_keys': head_keys, 'same_initial_head_sha256': head_hash,
        'training_executed': False, 'scientifically_qualified': False,
        'interpretation': 'Initialization preparation; compare with scratch using identical target head/optimizer/exposure budgets.'}
