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
        incoming = problem.get('carry_in', problem.get('incoming', 0))
        total = problem['left'] + problem['right'] + incoming if problem['operation'] == 'add_digit' else problem['left'] - problem['right'] - incoming
        return {'math_digit': total % 10, 'math_carry': total // 10 if problem['operation'] == 'add_digit' else int(total < 0)}
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


def model_accounting(model):
    """Exact tensor storage accounting, not total process/device peak memory."""
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {'parameter_count': sum(p.numel() for p in parameters),
            'trainable_parameter_count': sum(p.numel() for p in parameters if p.requires_grad),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in parameters),
            'buffer_bytes': sum(p.numel() * p.element_size() for p in buffers),
            'gradient_bytes': sum(p.grad.numel() * p.grad.element_size() for p in parameters if p.grad is not None),
            'estimated_adam_moment_bytes': 2 * sum(p.numel() * p.element_size() for p in parameters if p.requires_grad),
            'note': 'Tensor storage only; excludes allocator overhead, activations, and process peak RSS.'}


def controlled_graph(graph, control):
    """Apply frozen control consistently to training and evaluation."""
    from dataclasses import replace
    import torch
    if control in ('graph-concept-on', 'graph-memory-recurrent', 'mlp', 'gru', 'matched-mlp', 'matched-rnn', 'matched-gru'):
        return graph
    if control == 'graph-concept-off':
        return replace(graph, concept_ids=graph.concept_ids[:0])
    if control == 'graph-concept-shuffled':
        # Explicit v2 amendment: stable per-observation random IDs, not a
        # learnable one-to-one global relabeling. Uses no labels or references.
        import hashlib
        observed = {name: getattr(graph, name).detach().cpu().tolist() for name in
                    ('node_features', 'node_types', 'edges', 'relation_types', 'action')}
        observed['domain'] = graph.domain
        import json
        key = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).digest()
        ids = [key[i % len(key)] % 4 for i in range(len(graph.concept_ids))]
        return replace(graph, concept_ids=torch.tensor(ids,dtype=torch.long,device=graph.concept_ids.device))
    if control == 'graph-concept-ablated':
        return replace(graph, concept_ids=torch.zeros_like(graph.concept_ids))
    raise ValueError('Unsupported neural control')


def matched_baseline_plan(target_parameters=44962, tolerance=.05):
    """Deterministic capacity selection, no labels/losses/task tuning involved."""
    if target_parameters < 1 or not 0 < tolerance <= .05:
        raise ValueError('Invalid fixed capacity target')
    plan = {}
    for kind in ('mlp', 'rnn', 'gru'):
        counts = lambda width: ((width * width + 113 * width + 34) if kind != 'gru'
                                else (3 * width * width + 271 * width + 34))
        width = min(range(8, 513), key=lambda w: (abs(counts(w) - target_parameters), w))
        count = counts(width)
        if abs(count - target_parameters) / target_parameters > tolerance:
            raise ValueError('No allowed width meets frozen capacity tolerance')
        plan[kind] = {'width': width, 'parameter_count': count,
                      'relative_gap': abs(count - target_parameters) / target_parameters}
    return plan


def observed_tokens(graph):
    """Same observable nodes/types/edges/relations/action/domain/concepts, no labels.

    Token schema 77: node features8, node-type16, edge endpoints2,
    relation8, action4, domain3, concept32, token-kind4. Integer endpoints
    are normalized by fixed256. Padding is zero. Sequence order is observed
    node/edge order, not an answer-derived ordering.
    """
    import torch
    device = graph.node_features.device
    tokens = []
    for index in range(len(graph.node_features)):
        token = torch.zeros(77, device=device)
        token[:8] = graph.node_features[index]
        token[8 + int(graph.node_types[index])] = 1
        token[73] = 1
        tokens.append(token)
    for index in range(graph.edges.shape[1]):
        token = torch.zeros(77, device=device)
        token[24:26] = graph.edges[:, index].float() / 256
        token[26 + int(graph.relation_types[index])] = 1
        token[74] = 1
        tokens.append(token)
    context = torch.zeros(77, device=device)
    context[34:38] = graph.action
    context[38 + graph.domain] = 1
    for concept in graph.concept_ids:
        context[41 + int(concept)] += 1
    context[75] = 1
    tokens.append(context)
    return torch.stack(tokens)


def matched_baseline(kind, target_parameters=44962, seed=17):
    """Untrained original MLP/basic RNN/GRU with observed-input parity and <=5% capacity gap."""
    if kind not in ('mlp', 'rnn', 'gru'):
        raise ValueError('Unsupported matched baseline')
    import torch
    from torch import nn
    width = matched_baseline_plan(target_parameters)[kind]['width']
    class ObservedControl(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == 'mlp':
                self.core = nn.Sequential(nn.Linear(77, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
            else:
                self.core = (nn.RNN if kind == 'rnn' else nn.GRU)(77, width, batch_first=True)
            self.heads = nn.ModuleDict({name: nn.Linear(width, count) for name, count in
                [('math_digit_logits', 10), ('math_carry_logits', 2), ('coding_logits', 16),
                 ('physics_delta', 2), ('action_scores', 4)]})
        def forward(self, graph, memory=None, stateless=False):
            tokens = observed_tokens(graph)
            if kind == 'mlp':
                pooled = self.core(tokens).mean(0)
            else:
                initial = None if stateless or memory is None else memory.reshape(1, 1, -1)
                _, hidden = self.core(tokens.unsqueeze(0), initial)
                pooled = hidden[0, 0]
            return {**{name: head(pooled) for name, head in self.heads.items()},
                    'memory': pooled, 'pooled': pooled}
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ObservedControl()
    actual = model_accounting(model)['parameter_count']
    if actual != matched_baseline_plan(target_parameters)[kind]['parameter_count']:
        raise ValueError('Actual capacity differs from frozen constructor plan')
    return model
