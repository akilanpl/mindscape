"""New exploratory splits; never access official protected evaluator artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
from collections import Counter

SEEDS = (17, 29, 43)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalized(problem):
    domain = problem['domain']
    if domain == 'math':
        return {'domain': domain, 'operation': problem['operation'], 'left': problem['left'],
                'right': problem['right'], 'incoming': problem.get('incoming', problem.get('carry_in', problem.get('borrow_in', 0)))}
    if domain == 'coding':
        return {key: problem[key] for key in ('domain', 'source', 'old_name', 'candidates', 'requested_name')}
    return {key: problem[key] for key in ('domain', 'mass', 'velocity', 'force', 'dt')}


def identity(problem):
    return digest(normalized(problem))


def labels(problem):
    domain = problem['domain']
    if domain == 'math':
        p = normalized(problem)
        value = p['left'] + p['right'] + p['incoming'] if p['operation'] == 'add_digit' else p['left'] - p['right'] - p['incoming']
        return {'math_digit': value % 10, 'math_carry': value // 10 if p['operation'] == 'add_digit' else int(value < 0)}
    if domain == 'coding':
        return {'coding_index': problem['candidates'].index(problem['requested_name'])}
    return {'physics_delta': [value * problem['dt'] / problem['mass'] for value in problem['force']]}


def _row(problem, split, family):
    return {'identity': identity(problem), 'problem': problem, 'labels': labels(problem),
            'split': split, 'family': family}


def _math_candidates():
    for operation in ('add_digit', 'borrow_digit'):
        for left in range(10):
            for right in range(10):
                for incoming in (0, 1):
                    yield {'domain': 'math', 'operation': operation, 'left': left, 'right': right,
                           'incoming': incoming, 'concept_ids': [0 if operation == 'add_digit' else 1]}


def _select_math(candidates, count, rng):
    """Greedy deterministic balancing of joint operation/carry/digit strata where feasible."""
    remaining = list(candidates); rng.shuffle(remaining)
    selected, joint, digits, carry, operation = [], Counter(), Counter(), Counter(), Counter()
    while remaining and len(selected) < count:
        def score(problem):
            y = labels(problem); key = (problem['operation'], y['math_carry'], y['math_digit'])
            return (joint[key], digits[y['math_digit']], carry[y['math_carry']], operation[problem['operation']])
        chosen = min(range(len(remaining)), key=lambda index: score(remaining[index]))
        problem = remaining.pop(chosen); y = labels(problem)
        joint[(problem['operation'], y['math_carry'], y['math_digit'])] += 1
        digits[y['math_digit']] += 1; carry[y['math_carry']] += 1; operation[problem['operation']] += 1
        selected.append(problem)
    return selected


def split_rows(seed, split, blocked, math_count=80, coding_count=80, physics_count=80):
    rng = random.Random(seed + (1000 if split == 'dev' else 2000))
    available = [p for p in _math_candidates() if identity(p) not in blocked]
    math = _select_math(available, math_count, rng)
    rows = [_row(p, split, f'math-{p["operation"]}-{labels(p)["math_carry"]}-{labels(p)["math_digit"]}') for p in math]
    for index in range(coding_count // 4):
        old = f'x_{seed}_{split}_{index}'
        names = [f'candidate_{seed}_{split}_{index}_{k}' for k in range(4)]
        template = index % 4
        sources = [f'{old} = 1\nresult = {old} + 2', f'def f({old}):\n return {old} * {old}',
                   f'{old} = [1, 2]\nresult = {old}[0]', f'def f({old}):\n y = {old} - 1\n return y + {old}']
        for position in range(4):
            candidates = names[position:] + names[:position]
            problem = {'domain': 'coding', 'operation': 'rename', 'source': sources[template],
                       'old_name': old, 'candidates': candidates, 'requested_name': names[0], 'concept_ids': [3]}
            if identity(problem) not in blocked:
                rows.append(_row(problem, split, f'coding-{seed}-{split}-{index}'))
    index = 0
    while sum(r['problem']['domain'] == 'physics' for r in rows) < physics_count:
        problem = {'domain': 'physics', 'operation': 'force', 'mass': rng.choice([1., 2., 3., 5., 8.]),
                   'velocity': [rng.uniform(-5, 5), rng.uniform(-5, 5)],
                   'force': [rng.uniform(-10, 10), rng.uniform(-10, 10)], 'dt': rng.choice([.1, .25, .5, 1.]), 'concept_ids': [2]}
        if identity(problem) not in blocked:
            rows.append(_row(problem, split, f'physics-{seed}-{split}-{index}')); index += 1
    ids = [row['identity'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) & blocked:
        raise ValueError('Split identity violation')
    return rows


def counts(rows):
    result = {}
    for domain in ('math', 'coding', 'physics'):
        subset = [r for r in rows if r['problem']['domain'] == domain]
        record = {'records': len(subset), 'distinct': len({identity(r['problem']) for r in subset})}
        if domain == 'math':
            record.update(digit=dict(Counter(str(labels(r['problem'])['math_digit']) for r in subset)),
                          carry=dict(Counter(str(labels(r['problem'])['math_carry']) for r in subset)),
                          operation=dict(Counter(r['problem']['operation'] for r in subset)))
        if domain == 'coding':
            record['positions'] = dict(Counter(str(labels(r['problem'])['coding_index']) for r in subset))
        result[domain] = record
    return result


def prepare(output, legacy_root):
    """legacy_root must be r56-exploratory. No recursive discovery/protected reads."""
    output, legacy_root = Path(output), Path(legacy_root)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('train', 'dev', 'evaluator-only'):
        (output / folder).mkdir()
    manifest = {'status': 'EXPLORATORY_NOT_OFFICIAL', 'seeds': list(SEEDS), 'files': {}, 'counts': {}}
    for seed in SEEDS:
        base = legacy_root / ('seed17-2000-updates-v1' if seed == 17 else f'remaining-eight-v1/{seed}-graph-concept-on')
        train_path = base / 'training-data.json'
        train_raw = train_path.read_bytes(); train = json.loads(train_raw)
        if len(train) != 128:
            raise ValueError('Primary comparison requires exact legacy128')
        old_validation = json.loads((legacy_root / f'remaining-eight-v1/shared-validation-{seed}.json').read_bytes())
        blocked = {identity(row['problem']) for row in train + old_validation}
        dev = split_rows(seed, 'dev', blocked)
        final = split_rows(seed, 'final', blocked | {row['identity'] for row in dev})
        train_dest = output / f'train/{seed}.json'; train_dest.write_bytes(train_raw)
        for split, rows in [('dev', dev), ('evaluator-only/final', final)]:
            destination = output / f'{split}-{seed}.json' if '/' in split else output / f'{split}/{seed}.json'
            destination.write_text(json.dumps(rows, sort_keys=True, indent=2))
        paths = [train_dest, output / f'dev/{seed}.json', output / f'evaluator-only/final-{seed}.json']
        for path in paths:
            manifest['files'][str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['counts'][str(seed)] = {'train': counts(train), 'dev': counts(dev), 'final': counts(final)}
    manifest['manifest_sha256'] = digest(manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
