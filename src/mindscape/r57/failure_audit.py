"""Exploratory representation/loss audit with explicit bounded engineering updates."""
from __future__ import annotations
from collections import Counter
import math
from .data import counts, labels


def entropy(values):
    n = len(values)
    return -sum((count/n) * math.log2(count/n) for count in Counter(values).values()) if n else None


def gradient_audit(model, rows, condition='graph-concept-on', limit=24):
    """Actual margins and task gradient magnitudes; no optimizer updates."""
    import torch
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import controlled_graph
    from mindscape.r56.compact_gpu import domain_loss
    records = []
    model.eval()
    for row in rows[:limit]:
        model.zero_grad(set_to_none=True)
        output = model(controlled_graph(pack_problem(row['problem']), condition), memory=None, stateless=False)
        domain = row['problem']['domain']; y = row.get('labels', labels(row['problem']))
        if domain in ('math', 'coding'):
            key, target = ('math_digit_logits', y['math_digit']) if domain == 'math' else ('coding_logits', y['coding_index'])
            logits = output[key][:len(row['problem']['candidates'])] if domain == 'coding' else output[key]
            rivals = logits.detach().clone(); rivals[target] = -float('inf')
            margin = float((logits[target] - rivals.max()).detach())
            correct = int(logits.argmax()) == target
        else:
            margin, correct = None, None
        loss = domain_loss(output, {'problem': row['problem'], 'labels': y}, torch, torch.device('cpu'))
        loss.backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters() if parameter.grad is not None}
        records.append({'domain': domain, 'identity': row.get('identity', row.get('problem_identity')),
            'loss': float(loss.detach()), 'target_margin': margin, 'correct': correct,
            'gradient_l2': math.sqrt(sum(value ** 2 for value in gradients.values())),
            'shared_feature_gradient': gradients.get('features.weight'), 'gradients': gradients})
    return {'records': records, 'optimizer_updates': 0,
        'class_entropy_bits': {'math_digit': entropy([labels(r['problem'])['math_digit'] for r in rows if r['problem']['domain']=='math']),
        'coding_index': entropy([labels(r['problem'])['coding_index'] for r in rows if r['problem']['domain']=='coding'])},
        'coverage': counts(rows)}


def signal_probe(model, problem, changed_problem, condition='graph-concept-on'):
    """Actual encoded/node/pooled/logit distances; not causal proof from differences."""
    import torch
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import controlled_graph
    model.eval()
    graphs = [controlled_graph(pack_problem(p), condition) for p in (problem, changed_problem)]
    with torch.no_grad():
        outputs = [model(g, memory=None, stateless=False) for g in graphs]
    result = {'encoded_shape_equal': graphs[0].node_features.shape == graphs[1].node_features.shape}
    if result['encoded_shape_equal']:
        result['encoded_l2'] = float((graphs[0].node_features - graphs[1].node_features).norm())
    for key in ('node_states', 'pooled', 'coding_logits', 'math_digit_logits'):
        if key in outputs[0] and outputs[0][key].shape == outputs[1][key].shape:
            result[key + '_l2'] = float((outputs[0][key] - outputs[1][key]).norm())
    return result


def bounded_sampling_probe(rows, updates=120, seed=17):
    """Paired CPU engineering test, fixed total budget <=600 across two strategies."""
    if not 1 <= updates <= 300:
        raise ValueError('Two strategies combined cannot exceed600 updates')
    import torch
    from mindscape.r56.neural import R56Model
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.compact_gpu import domain_loss
    groups = {d: [r for r in rows if r['problem']['domain']==d] for d in ('math','coding','physics')}
    if not all(groups.values()):
        raise ValueError('All three domains required')
    result = {'optimizer_updates_total': updates * 2, 'device':'cpu', 'engineering_only': True, 'strategies': {}}
    for strategy in ('legacy-order', 'balanced-normalized'):
        torch.manual_seed(seed); model = R56Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        scales = {}
        for domain, members in groups.items():
            with torch.no_grad():
                losses = [float(domain_loss(model(pack_problem(r['problem']),memory=None,stateless=False),r,torch,torch.device('cpu'))) for r in members[:8]]
            scales[domain] = max(sum(losses)/len(losses), 1e-6)
        before = gradient_audit(model, rows, limit=12)
        for step in range(updates):
            if strategy == 'legacy-order':
                row = rows[step % len(rows)]
            else:
                domain = ('math','coding','physics')[step%3]; members=groups[domain]
                row = members[(step//3)%len(members)]
            loss = domain_loss(model(pack_problem(row['problem']),memory=None,stateless=False),row,torch,torch.device('cpu'))
            if strategy == 'balanced-normalized': loss = loss / scales[row['problem']['domain']]
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        result['strategies'][strategy] = {'initial':before,'final':gradient_audit(model,rows,limit=12),
                                          'updates':updates,'frozen_initial_task_loss_scales':scales}
    return result
