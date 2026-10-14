"""Reproducible bounded synthetic tasks. Targets never enter predictor problems."""
from __future__ import annotations
import hashlib
import json
import random

SEEDS = (17, 29, 43)
BUDGETS = (32, 128, 512)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def generate(split: str, count: int = 512, seed: int = 17) -> list[dict]:
    if split not in ('train', 'dev', 'test') or not 1 <= count <= 4096 or seed not in SEEDS:
        raise ValueError('Unsupported frozen curriculum configuration')
    rng = random.Random(seed + {'train': 0, 'dev': 10000, 'test': 20000}[split])
    rows = []
    # Disjoint problem identities and paraphrase families; numeric overlap is deliberate
    # for matched exposure. This is an IID split, not an OOD generalization claim.
    for index in range(count):
        domain = ('math', 'coding', 'physics')[index % 3]
        if domain == 'math':
            while True:
                left, right, carry = rng.randrange(10), rng.randrange(10), rng.randrange(2)
                operation = rng.choice(('add_digit', 'borrow_digit'))
                problem = {'domain': 'math', 'operation': operation, 'left': left,
                           'right': right, 'carry_in': carry}
                if int(digest(problem), 16) % 3 == ('train', 'dev', 'test').index(split):
                    break
            total = left + right + carry if operation == 'add_digit' else left - right - carry
            labels = {'math_digit': total % 10,
                      'math_carry': total // 10 if operation == 'add_digit' else int(total < 0)}
        elif domain == 'coding':
            old = f'value_{index}'
            candidates = [f'{split}_candidate_{index}_{n}' for n in range(4)]
            chosen = rng.randrange(4)
            problem = {'domain': 'coding', 'operation': 'rename',
                       'source': f'{old} = 1\nresult = {old} + {old}',
                       'old_name': old, 'candidates': candidates,
                       'requested_name': candidates[chosen],
                       'instruction': f'Rename {old} to {candidates[chosen]}.'}
            # Candidate requested in instruction is legitimate task data, not verifier context.
            labels = {'coding_index': chosen, 'target_name': candidates[chosen]}
        else:
            while True:
                mass = float(rng.randrange(1, 10))
                force = [float(rng.randrange(-10, 11)), float(rng.randrange(-10, 11))]
                dt = rng.choice([0.1, 0.25, 0.5, 1.0])
                problem = {'domain': 'physics', 'operation': 'force', 'mass': mass,
                           'velocity': [float(rng.randrange(-5, 6)), float(rng.randrange(-5, 6))],
                           'force': force, 'dt': dt}
                if int(digest(problem), 16) % 3 == ('train', 'dev', 'test').index(split):
                    break
            labels = {'physics_delta': [force[0] * dt / mass, force[1] * dt / mass]}
        problem['concept_ids'] = ([0] if problem['operation'] == 'add_digit' else [1]) if domain == 'math' else ([2] if domain == 'physics' else [3])
        row = {'split': split, 'seed': seed, 'index': index, 'problem': problem,
               'labels': labels, 'paraphrase_family': f'{split}-wording-{index % 3}'}
        row['problem_identity'] = digest(problem)
        row['identity'] = digest(row)
        rows.append(row)
    return rows


def prepare_data(seed=17):
    return {split: generate(split, seed=seed) for split in ('train', 'dev', 'test')}


def extension_fixtures():
    """Preparation examples, explicitly outside the primary frozen neural campaign."""
    return [
        {'problem': {'domain': 'math', 'operation': 'borrow_digit', 'left': 3, 'right': 7, 'incoming': 0},
         'labels': {'math_digit': 6, 'math_carry': 1}, 'primary_training_supported': True},
        {'problem': {'domain': 'math', 'operation': 'equation_update', 'coefficient': 2, 'offset': 3, 'rhs': 11},
         'labels': {'solution': 4}, 'primary_training_supported': False},
        {'problem': {'domain': 'physics', 'operation': 'impulse', 'mass': 2.0,
                     'velocity': [1.0, 0.0], 'impulse': [4.0, -2.0]},
         'labels': {'physics_delta': [2.0, -1.0]}, 'primary_training_supported': True},
        {'problem': {'domain': 'physics', 'operation': 'elastic_collision_1d',
                     'masses': [1.0, 1.0], 'velocities': [2.0, -1.0]},
         'labels': {'velocities': [-1.0, 2.0]}, 'primary_training_supported': False},
    ]
