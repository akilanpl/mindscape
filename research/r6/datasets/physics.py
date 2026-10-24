"""Original SI motion problems. Targets are external labels, never model features."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import random

GENERATOR_VERSION = 'r6-v1'
FEATURE_NAMES = ('initial_vx', 'initial_vy', 'initial_x', 'initial_y', 'mass') + tuple(
    f'stage_{i}_{name}' for i in range(3) for name in ('fx', 'fy', 'dt', 'active'))
# Fixed human unit references, not statistics estimated from any split.
FEATURE_SCALES = (10., 10., 20., 20., 5.) + (10., 10., 2., 1.) * 3
SPLITS = ('train', 'dev', 'final_iid', 'final_ood')


def _canonical(system):
    return json.dumps(system, sort_keys=True, separators=(',', ':'))


def solve(system):
    """Direct sequential constant-acceleration kinematics (label generator)."""
    validate(system)
    vx, vy, x, y = map(float, system['initial_state'])
    mass = system['mass']
    for stage in system['stages']:
        fx, fy = stage['force']; t = stage['dt']
        x += vx * t + .5 * fx / mass * t * t
        y += vy * t + .5 * fy / mass * t * t
        vx += fx / mass * t; vy += fy / mass * t
    return [vx, vy, x, y]


def verify(system):
    """Independent exact impulse moments integrated over the full time interval.

    Position contribution of impulse density F/m on [a,b] is
    F/m * integral_a^b (T-t)dt. This is not the sequential generator recurrence.
    """
    validate(system)
    q = lambda value: Fraction(str(value))
    initial = list(map(q, system['initial_state']))
    mass = q(system['mass'])
    total = sum((q(stage['dt']) for stage in system['stages']), Fraction())
    velocity = initial[:2]; position = [initial[2+i] + initial[i]*total for i in range(2)]
    elapsed = Fraction()
    for stage in system['stages']:
        dt = q(stage['dt']); end = elapsed + dt
        moment = total * dt - (end*end - elapsed*elapsed)/2
        for axis in range(2):
            force = q(stage['force'][axis])
            velocity[axis] += force * dt / mass
            position[axis] += force * moment / mass
        elapsed = end
    return list(map(float, velocity + position))


def validate(system):
    if system['units'] != {'mass': 'kg', 'force': 'N', 'time': 's', 'position': 'm', 'velocity': 'm/s'}:
        raise ValueError('Only explicitly declared SI units are accepted')
    if len(system['initial_state']) != 4 or not 1 <= len(system['stages']) <= 3:
        raise ValueError('Invalid state or stage count')
    values = list(system['initial_state']) + [system['mass']]
    if system['mass'] <= 0:
        raise ValueError('Mass must be positive')
    for stage in system['stages']:
        if len(stage['force']) != 2 or stage['dt'] <= 0:
            raise ValueError('Each force is 2D and time must be positive')
        values += list(stage['force']) + [stage['dt']]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite physical input')


def features(system):
    validate(system)
    values = list(system['initial_state']) + [system['mass']]
    for i in range(3):
        values.extend(list(system['stages'][i]['force']) + [system['stages'][i]['dt'], 1.]
                      if i < len(system['stages']) else [0., 0., 0., 0.])
    return values


def _record(system, split, suffix=''):
    key = _canonical(system)
    digest = hashlib.sha256(key.encode()).hexdigest()
    n = len(system['stages'])
    text = f"SI motion: initial(vx,vy,x,y)={system['initial_state']}; mass={system['mass']}kg; " + ';'.join(
        f"F={s['force']}N t={s['dt']}s" for s in system['stages'])
    if len(text.encode()) > 192:
        raise ValueError('Input exceeds 192 bytes')
    target = solve(system)
    if any(abs(a-b) > 1e-9 for a,b in zip(target, verify(system))):
        raise AssertionError('Independent physics verifier disagrees')
    return {'record_id': 'physics-'+digest[:24]+suffix, 'domain': 'physics',
            'topic': 'multi-stage-2d-motion', 'difficulty': 'foundation' if n == 2 else 'structural-ood',
            'input': text, 'structured_input': system, 'target': {'state': target},
            'features': features(system), 'verification_method': 'exact-rational-impulse-moments',
            'source': 'generated', 'split': split, 'generator_version': GENERATOR_VERSION,
            'template_id': f'motion-{n}-stages', 'canonical_key': key}


def generate_with_metadata(split, seed, count):
    if split not in SPLITS or not isinstance(count, int) or count < 0:
        raise ValueError('Invalid split or count')
    rng = random.Random(f'physics:{GENERATOR_VERSION}:{seed}:{split}')
    rows = []; seen = set(); attempts = 0; duplicate_rejections = 0; partition_rejections = 0
    while len(rows) < count:
        attempts += 1
        if attempts > max(1000, count*100):
            raise RuntimeError('Generator exhausted acceptance budget')
        ood = split == 'final_ood'
        system = {'initial_state': [rng.choice([i for i in range(-8,9) if i]) for _ in range(4)],
                  'mass': rng.choice([1,2,3,4,5] if not ood else [6,8,10]),
                  'stages': [{'force': [rng.randint(-10,10) if not ood else rng.choice([-20,-15,15,20]) for _ in range(2)],
                              'dt': rng.choice([.25,.5,1.,2.] if not ood else [2.5,3.,4.])} for _ in range(3 if ood else 2)],
                  'units': {'mass':'kg','force':'N','time':'s','position':'m','velocity':'m/s'}}
        key = _canonical(system)
        if key in seen:
            duplicate_rejections += 1
            continue
        # Globally stable context partition irrespective of generation seed.
        bucket = int(hashlib.sha256(key.encode()).hexdigest(),16) % 10
        if not ood and ((split == 'train' and bucket >= 8) or (split == 'dev' and bucket != 8) or (split == 'final_iid' and bucket != 9)):
            partition_rejections += 1
            continue
        seen.add(key); rows.append(_record(system, split))
    return rows, {'attempted': attempts, 'accepted': len(rows),
                  'canonical_duplicate_rejections': duplicate_rejections,
                  'partition_rejections': partition_rejections, 'invalid_rejections': 0}


def generate(split, seed, count):
    return generate_with_metadata(split, seed, count)[0]


def perturbations(row):
    """Named physical interventions, with independently checked reference targets."""
    base = row['structured_input']
    results = {}
    system = deepcopy(base)
    for stage in system['stages']:
        stage['force'] = [2*f for f in stage['force']]
    results['double_force'] = _record(system, row['split'])
    system = deepcopy(base); system['mass'] *= 2
    results['double_mass'] = _record(system, row['split'])
    # Splitting a constant-force interval preserves the exact final state.
    if len(base['stages']) < 3:
        system = deepcopy(base); first = system['stages'].pop(0)
        first['dt'] /= 2
        system['stages'] = [deepcopy(first), deepcopy(first)] + system['stages']
        results['split_time_same_force'] = _record(system, row['split'])
    system = deepcopy(base)
    for stage in system['stages']:
        stage['dt'] *= 2
    results['double_time'] = _record(system, row['split'])
    return results
