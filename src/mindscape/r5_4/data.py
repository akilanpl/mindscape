"""Deterministic generation and training-only supervision; never imported by neural inference."""
import hashlib
import json
import random
from pathlib import Path
from .inference import Operands, columns

HELD_PAIRS = {(2, 7), (7, 2), (3, 6), (6, 3), (4, 8), (8, 4), (1, 9), (9, 1)}
FAMILIES = ('length_known', 'long_carry_known', 'mixed_known', 'unseen_short')
ROOT = Path('experiments/r5_4/frozen')


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2) + '\n'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(task):
    carry, trace = 0, []
    for left, right in columns(task):
        total = left + right + carry
        trace.append([left, right, carry, total % 10, total // 10])
        carry = total // 10
    serialized = ((str(carry) if carry else '') + ''.join(str(x[3]) for x in trace[::-1])).lstrip('0') or '0'
    if serialized != str(int(task.left) + int(task.right)):
        raise ValueError('Reference implementations disagree')
    return trace


def generate():
    rng = random.Random(54013)
    known = [(a, b) for a in range(10) for b in range(10) if (a, b) not in HELD_PAIRS]
    identities, tasks = set(), []
    for split, family, count in [('train', 'short_known', 1024), ('development', 'short_known', 64)] + [
            ('ood', family, 48) for family in FAMILIES]:
        accepted = attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError('Bounded generator exhausted')
            n = rng.randint(2, 4) if split != 'ood' or family == 'unseen_short' else rng.randint(8, 16)
            if family == 'long_carry_known':
                n = rng.randint(12, 24)
                pairs = [rng.choice([p for p in known if sum(p) >= 10])]
                pairs += [rng.choice([p for p in known if sum(p) == 9]) for _ in range(n - 1)]
            elif family == 'mixed_known':
                pairs = [rng.choice([p for p in known if (sum(p) >= 10 if j % 2 == 0 else sum(p) <= 7)])
                         for j in range(n)]
            else:
                pairs = [rng.choice(known) for _ in range(n)]
                if family == 'unseen_short':
                    pairs[rng.randrange(n)] = rng.choice(sorted(HELD_PAIRS))
            if not pairs[-1][0] or not pairs[-1][1]:
                continue
            left = ''.join(str(p[0]) for p in pairs[::-1])
            right = ''.join(str(p[1]) for p in pairs[::-1])
            identity = tuple(sorted((left, right)))  # exclude commuted duplicates too
            if identity in identities:
                continue
            identities.add(identity)
            task_id = f'r54-{split}-{family}-{accepted:04d}'
            task = dict(task_id=task_id, split=split, family=family, left=left, right=right)
            tasks.append(task)
            accepted += 1
    return tasks


def operand(task):
    return Operands(task['task_id'], task['left'], task['right'])


def freeze_data(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    tasks = generate()
    for split in ('train', 'development', 'ood'):
        rows = [t for t in tasks if t['split'] == split]
        (root / f'{split}.json').write_text(canonical(rows))
    # Public operands separate from outcomes. Only scorer opens these after predictions commit.
    outcomes = {t['task_id']: dict(answer=str(int(t['left']) + int(t['right'])),
                                 trace=reference(operand(t))) for t in tasks if t['split'] != 'train'}
    (root / 'outcomes.json').write_text(canonical(outcomes))


def qualify(root=ROOT):
    tasks = sum([json.loads((root / f'{split}.json').read_text())
                 for split in ('train', 'development', 'ood')], [])
    if tasks != generate():
        raise ValueError('Manifest not reproducible')
    contexts, seen, counts = {}, set(), {}
    for split in ('train', 'development', 'ood'):
        rows = [t for t in tasks if t['split'] == split]
        local = [r for t in rows for r in reference(operand(t))]
        contexts[split] = len({tuple(r[:3]) for r in local})
        counts[split] = len(rows)
        if split == 'train':
            seen = {tuple(r[:3]): tuple(r[3:]) for r in local}
            assert not any(tuple(r[:2]) in HELD_PAIRS for r in local)
    from .inference import rollout
    memory = lambda a, b, c, h: (*seen.get((a, b, c), (0, 0)), None)
    baseline, family_contexts = {}, {}
    for family in FAMILIES:
        rows = [t for t in tasks if t['family'] == family]
        baseline[family] = sum(rollout(operand(t), memory)['answer'] == str(int(t['left']) + int(t['right']))
                               for t in rows)
        local = [r for t in rows for r in reference(operand(t))]
        family_contexts[family] = dict(unique=len({tuple(r[:3]) for r in local}),
            unseen=len({tuple(r[:3]) for r in local if tuple(r[:3]) not in seen}),
            columns=len(local), unseen_columns=sum(tuple(r[:3]) not in seen for r in local))
        if family != 'unseen_short' and any(tuple(r[:3]) not in seen for r in local):
            raise ValueError('Known-context length diagnostic contaminated')
    hidden = json.loads((root / 'outcomes.json').read_text())
    for task in tasks:
        if task['split'] != 'train':
            expected = dict(answer=str(int(task['left']) + int(task['right'])), trace=reference(operand(task)))
            if hidden[task['task_id']] != expected:
                raise ValueError('Independent reference mismatch')
    ready = counts == {'train': 1024, 'development': 64, 'ood': 192} and sum(baseline.values()) <= 144
    return dict(status='DATA_READY' if ready else 'DATA_NO_GO', counts=counts,
        unique_contexts=contexts, family_contexts=family_contexts,
        held_contexts=[(a, b, c) for a, b in sorted(HELD_PAIRS) for c in (0, 1)],
        indirect_training_exposures=0, explanation='Both carries excluded for held digit pairs, including own-carry training.',
        memory_correct=baseline, memory_miss_policy='digit=0, carry=0',
        independent_ood_denominator=192, oracle_correct=192)
