"""Untrained benchmark controls and paired statistics; no research success defaults."""
from __future__ import annotations
import math
import random
from .curriculum import digest

BASELINES = ('exact-rule', 'memorization', 'mlp', 'gru', 'graph-concept-on',
             'graph-concept-off', 'graph-concept-shuffled', 'graph-concept-ablated')


def exact_rule(problem):
    domain = problem['domain']
    if domain == 'math':
        total = problem['left'] + problem['right'] + problem.get('carry_in', 0)
        return {'math_digit': total % 10, 'math_carry': total // 10}
    if domain == 'coding':
        return {'coding_index': problem['candidates'].index(problem['requested_name'])}
    return {'physics_delta': [f * problem['dt'] / problem['mass'] for f in problem['force']]}


class Memorization:
    """Exact task-key table, not a neural architecture or generalization claim."""
    def __init__(self, rows):
        self.table = {digest(r['problem']): r['labels'] for r in rows}

    def predict(self, problem):
        return self.table.get(digest(problem))


def paired_bootstrap(left, right, seed=17, replicates=2000):
    if len(left) != len(right) or not left or not 100 <= replicates <= 100000:
        raise ValueError('Require paired nonempty observations and bounded replicates')
    deltas = [float(a) - float(b) for a, b in zip(left, right)]
    if not all(math.isfinite(x) for x in deltas):
        raise ValueError('Nonfinite observations')
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(replicates))
    return {'mean_difference': sum(deltas) / len(deltas),
            'ci95': [samples[int(.025 * replicates)], samples[int(.975 * replicates)]],
            'resampling_unit': 'caller-supplied independent problem-family observation'}


def holm(p_values):
    if not p_values or any(not 0 <= p <= 1 for p in p_values):
        raise ValueError('Require finite valid p values')
    result, previous = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        previous = max(previous, min(1.0, p_values[index] * (len(p_values) - rank)))
        result[index] = previous
    return result


def pooled_baseline(kind, width=48):
    """Explicit information-mismatched pooled controls; not graph-equivalent baselines.

    Both retain graph node features but discard edges/types/concepts. Report this
    loss of information and match parameters/compute before scientific comparison.
    """
    if kind not in ('mlp', 'gru') or not 8 <= width <= 256:
        raise ValueError('Unsupported pooled baseline')
    import torch
    from torch import nn
    class PooledControl(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(8, width), nn.Tanh()) if kind == 'mlp' else nn.GRU(8, width, batch_first=True)
            self.digit, self.carry = nn.Linear(width, 10), nn.Linear(width, 2)
            self.coding, self.physics = nn.Linear(width, 16), nn.Linear(width, 2)
        def forward(self, graph, memory=None, stateless=False):
            if kind == 'mlp':
                pooled = self.encoder(graph.node_features).mean(dim=0)
            else:
                _, hidden = self.encoder(graph.node_features.unsqueeze(0))
                pooled = hidden[0, 0]
            return {'math_digit_logits': self.digit(pooled), 'math_carry_logits': self.carry(pooled),
                    'coding_logits': self.coding(pooled), 'physics_delta': self.physics(pooled),
                    'pooled': pooled, 'memory': pooled}
    return PooledControl()
