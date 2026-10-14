"""Read-only independent verifier: no training, no writes, no shared scoring oracle."""
import json
import math
from pathlib import Path
from .data import ROOT, digest, HELD_PAIRS, FAMILIES
from .freeze import verify_lock, PROTOCOL


def independent_reference(task):
    a, b = task['left'][::-1], task['right'][::-1]
    carry, trace = 0, []
    for position in range(max(len(a), len(b))):
        left = int(a[position]) if position < len(a) else 0
        right = int(b[position]) if position < len(b) else 0
        carry_out, digit = divmod(left + right + carry, 10)
        trace.append([left, right, carry, digit, carry_out])
        carry = carry_out
    return {'answer': str(int(task['left']) + int(task['right'])), 'trace': trace}


def wilson(correct, total):
    if not total or not 0 <= correct <= total:
        raise ValueError('Invalid denominator')
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def paired(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('Paired denominator mismatch')
    wins = sum(a and not b for a, b in zip(left, right))
    losses = sum(b and not a for a, b in zip(left, right))
    discordant = wins + losses
    p = sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / 2**discordant
    return dict(tasks=len(left), wins=wins, losses=losses, mean_gain=(wins - losses) / len(left), p_one_sided=p)


def identities():
    return [(f'r54-{kind}-neural-seed-{seed}-u800', 'neural', seed)
            for kind in PROTOCOL['models'] for seed in PROTOCOL['seeds']] + [
            ('r54-local-memory', 'local_memory', 0), ('r54-markov-zero', 'zero_weight', 0),
            ('r54-recurrent-zero', 'zero_weight', 0), ('r54-whole-memory', 'whole_memory', 0)]


def validate_record(record, task):
    if record['split'] != task['split'] or record['family'] != task['family']:
        raise ValueError('Condition/task attribution mismatch')
    n = max(len(task['left']), len(task['right']))
    if record['horizon'] != n:
        raise ValueError('Horizon mismatch')
    trace = record['trace']
    metadata = record['tensor_metadata']
    if metadata['recorded_classes_dtype'] != 'json_integer' or metadata['trace_shape'] != [len(trace), 6]:
        raise ValueError('Missing tensor metadata')
    if record['condition'] == 'whole_memory':
        if trace or record['answer'] != '0':
            raise ValueError('Whole-memory miss policy mismatch')
        return
    if len(trace) != n:
        raise ValueError('Incomplete trajectory')
    carry, digits = 0, []
    for position, row in enumerate(trace):
        left = int(task['left'][-1-position]) if position < len(task['left']) else 0
        right = int(task['right'][-1-position]) if position < len(task['right']) else 0
        if any(type(row[k]) is not int for k in ('position', 'left', 'right', 'carry_in', 'digit', 'carry_out')):
            raise ValueError('Class integer precision/type mismatch')
        if (row['position'], row['left'], row['right'], row['carry_in']) != (position, left, right, carry):
            raise ValueError('Carry feedback or cursor replay mismatch')
        if row['digit'] not in range(10) or row['carry_out'] not in (0, 1):
            raise ValueError('Invalid class')
        digits.append(str(row['digit']))
        carry = row['carry_out']
    serialized = (('1' if carry else '') + ''.join(digits[::-1])).lstrip('0') or '0'
    if record['answer'] != serialized:
        raise ValueError('Serialization replay mismatch')


def metrics(rows, tasks, outcomes):
    answer_correct, local_correct, local_total, errors = [], 0, 0, []
    for record, task in zip(rows, tasks):
        expected = outcomes[task['task_id']]
        answer_correct.append(record['answer'] == expected['answer'])
        bad = []
        for position, row in enumerate(record['trace']):
            # True trajectory local accuracy; own-state transition diagnostic kept separate below.
            truth = expected['trace'][position]
            good = (row['digit'], row['carry_out']) == tuple(truth[3:])
            local_correct += good
            local_total += 1
            if not good:
                bad.append(dict(position=position, predicted=row, reference=truth))
        if bad or not answer_correct[-1]:
            errors.append(dict(task_id=task['task_id'], answer=record['answer'], expected=expected['answer'], errors=bad))
    correct = sum(answer_correct)
    return dict(tasks=len(tasks), correct=correct, exact_accuracy=correct / len(tasks),
                wilson95=wilson(correct, len(tasks)), local_correct=local_correct, local_total=local_total,
                local_accuracy=local_correct / local_total if local_total else None,
                task_success=answer_correct, error_traces=errors)


def checkpoint_audit(destination, seal):
    import torch
    inspected = []
    expected = [(kind, seed) for kind in PROTOCOL['models'] for seed in PROTOCOL['seeds']]
    if len(seal['checkpoints']) != 6:
        raise ValueError('Incomplete fit count')
    for entry, (kind, seed) in zip(seal['checkpoints'], expected):
        fit_id = f'r54-{kind}-neural-seed-{seed}-u800'
        if (entry['kind'], entry['seed'], entry['model_id'], entry['path']) != (kind, seed, fit_id, f'{fit_id}.pt'):
            raise ValueError('Checkpoint attribution mismatch')
        path = destination / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError('Checkpoint hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['model_id'], payload['kind'], payload['seed'], payload['updates'], payload['parameters']) != (
                fit_id, kind, seed, 800, PROTOCOL['models'][kind]):
            raise ValueError('Checkpoint identity/update mismatch')
        from .model import Transition
        model = Transition(kind)
        model.load_state_dict(payload['model'], strict=True)
        for name, tensor in payload['model'].items():
            meta = payload['tensor_metadata'][name]
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
                raise ValueError('Checkpoint dtype/nonfinite')
            if meta != dict(dtype='torch.float32', shape=list(tensor.shape), device='mps:0'):
                raise ValueError('Checkpoint metadata mismatch')
        optimizer = payload['optimizer']
        if len(optimizer['state']) != len(list(model.parameters())):
            raise ValueError('Incomplete optimizer state')
        groups = optimizer['param_groups']
        if len(groups) != 1 or groups[0]['lr'] != .003 or groups[0]['weight_decay'] != .0001 or groups[0]['betas'] != (.9, .999):
            raise ValueError('Optimizer policy mismatch')
        for state, parameter in zip(optimizer['state'].values(), model.parameters()):
            if state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape:
                raise ValueError('Optimizer tensor shape mismatch')
            if int(state['step'].item()) != 800:
                raise ValueError('Optimizer update mismatch')
            if any(not torch.isfinite(value).all().item() for value in state.values() if isinstance(value, torch.Tensor)):
                raise ValueError('Nonfinite optimizer tensor')
        progress = [json.loads(line) for line in (destination / f'{fit_id}.progress.jsonl').read_text().splitlines()]
        if [p['update'] for p in progress] != list(range(1, 801)):
            raise ValueError('Incomplete optimizer progress')
        if any(p['dtype'] != 'torch.float32' or not math.isfinite(p['loss']) for p in progress):
            raise ValueError('Loss precision/finite failure')
        if not payload['exposure'] or any(not (0 <= a <= 9 and 0 <= b <= 9 and c in (0, 1)) or (a, b) in HELD_PAIRS for a, b, c in payload['exposure']):
            raise ValueError('Held context exposure/incomplete exposure evidence')
        inspected.append(dict(model_id=fit_id, finite=True, updates=800, parameters=payload['parameters'],
                              indirectly_exposed_held_contexts=0, visited_contexts=len(payload['exposure'])))
    return inspected


def audit(destination, replay_mps=False, require_complete=True):
    verify_lock()
    destination = Path(destination)
    if require_complete and not (destination / 'COMPLETE.json').exists():
        raise ValueError('Incomplete evidence: completion receipt absent')
    if (destination / 'FAILED.json').exists():
        raise ValueError('Failed campaign is not complete evidence')
    seal = json.loads((destination / 'PREDICTIONS_SEALED.json').read_text())
    if seal['outcomes_opened'] is not False or digest(destination / 'predictions.json') != seal['prediction_sha256']:
        raise ValueError('Prediction seal failure')
    start = json.loads((destination / 'start.json').read_text())
    if start['lock_sha256'] != digest(ROOT / 'LOCK.json') or start['protocol_sha256'] != digest(ROOT / 'protocol.json'):
        raise ValueError('Campaign freeze provenance mismatch')
    if start['engineering']['status'] != 'ENGINEERING_READY':
        raise ValueError('Unqualified campaign')
    if (destination / 'COMPLETE.json').exists():
        receipt = json.loads((destination / 'COMPLETE.json').read_text())
        files = {p.name for p in destination.iterdir() if p.is_file()} - {'COMPLETE.json'}
        if set(receipt['files']) != files or receipt['status'] != 'COMPLETE':
            raise ValueError('Incomplete artifact receipt')
        for filename, expected in receipt['files'].items():
            if digest(destination / filename) != expected:
                raise ValueError('Artifact receipt hash mismatch')
    tasks = json.loads((ROOT / 'development.json').read_text()) + json.loads((ROOT / 'ood.json').read_text())
    if len(tasks) != 256 or sum(t['split'] == 'ood' for t in tasks) != 192:
        raise ValueError('Evaluation denominator mismatch')
    records = json.loads((destination / 'predictions.json').read_text())
    expected_keys = [(model_id, condition, seed, t['task_id']) for model_id, condition, seed in identities() for t in tasks]
    actual_keys = [(r['model_id'], r['condition'], r['seed'], r['task_id']) for r in records]
    if actual_keys != expected_keys or seal['records'] != len(expected_keys):
        raise ValueError('Incomplete/duplicate/order-mismatched attribution')
    train = json.loads((ROOT / 'train.json').read_text())
    memory_table = {tuple(row[:3]): tuple(row[3:]) for task in train for row in independent_reference(task)['trace']}
    for i, record in enumerate(records):
        validate_record(record, tasks[i % len(tasks)])
        if record['condition'] in ('local_memory', 'zero_weight'):
            for row in record['trace']:
                expected = memory_table.get((row['left'], row['right'], row['carry_in']), (0, 0)) if record['condition'] == 'local_memory' else (0, 0)
                if (row['digit'], row['carry_out']) != expected:
                    raise ValueError('Independent control replay mismatch')
    # Base verification remains separate from diagnostics, fixing augmented dictionary replay defects.
    outcomes = json.loads((ROOT / 'outcomes.json').read_text())
    if set(outcomes) != {t['task_id'] for t in tasks}:
        raise ValueError('Outcome identity mismatch')
    for task in tasks:
        if outcomes[task['task_id']] != independent_reference(task):
            raise ValueError('Independent reference mismatch')
    results = {}
    for group, (model_id, condition, seed) in enumerate(identities()):
        rows = records[group * len(tasks):(group + 1) * len(tasks)]
        result = {}
        for family in ('development', 'ood', *FAMILIES):
            idx = [i for i, t in enumerate(tasks) if t['split'] == family or t['family'] == family]
            result[family] = metrics([rows[i] for i in idx], [tasks[i] for i in idx], outcomes)
        results[model_id] = result
    diagnostics = json.loads((destination / 'diagnostics.json').read_text())
    diag_ids = identities()[:-1]
    expected_diag = [(mid, cond, seed, mode, tid) for mid, cond, seed in diag_ids for mode, tid in (
        [('teacher_forced', t['task_id']) for t in tasks] +
        [('context_probe', f'context-{a}-{b}-{c}') for a in range(10) for b in range(10) for c in (0, 1)])]
    if [(r['model_id'], r['condition'], r['seed'], r['mode'], r['task_id']) for r in diagnostics] != expected_diag:
        raise ValueError('Diagnostic attribution/denominator mismatch')
    teacher_metrics, probes = {}, {}
    for model_id, condition, seed in diag_ids:
        subset = [r for r in diagnostics if r['model_id'] == model_id]
        local_good = local_n = held_good = held_n = probe_good = 0
        for row in subset:
            if row['mode'] == 'context_probe':
                a, b, c = row['context']
                if any(type(row[k]) is not int for k in ('digit', 'carry_out')) or row['digit'] not in range(10) or row['carry_out'] not in (0, 1):
                    raise ValueError('Probe class dtype/range mismatch')
                if row['task_id'] != f'context-{a}-{b}-{c}' or not (0 <= a <= 9 and 0 <= b <= 9 and c in (0, 1)):
                    raise ValueError('Probe identity mismatch')
                outgoing, digit = divmod(a + b + c, 10)
                good = (row['digit'], row['carry_out']) == (digit, outgoing)
                probe_good += good
                if (a, b) in HELD_PAIRS:
                    held_n += 1
                    held_good += good
            else:
                expected = outcomes[row['task_id']]['trace']
                if len(row['trace']) != len(expected):
                    raise ValueError('Incomplete TF trace')
                for position, (predicted, truth) in enumerate(zip(row['trace'], expected)):
                    if (predicted['position'], predicted['left'], predicted['right'], predicted['carry_in']) != (position, *truth[:3]):
                        raise ValueError('TF cursor/context mismatch')
                    local_good += (predicted['digit'], predicted['carry_out']) == tuple(truth[3:])
                    local_n += 1
        if held_n != 16:
            raise ValueError('Held probe denominator')
        probes[model_id] = dict(all_correct=probe_good, all_total=200, held_correct=held_good,
                               held_total=held_n, held_accuracy=held_good / held_n, held_wilson95=wilson(held_good, held_n))
        teacher_metrics[model_id] = dict(local_correct=local_good, local_total=local_n, accuracy=local_good / local_n)
    thresholds = PROTOCOL['gates_per_seed_per_model']
    gates, comparisons = {}, {}
    memory = results['r54-local-memory']
    for model_id, condition, seed in identities()[:6]:
        result = results[model_id]
        compared = {s: paired(result[s]['task_success'], memory[s]['task_success']) for s in ('ood', 'unseen_short')}
        comparisons[model_id] = compared
        gates[model_id] = dict(
            ood_exact=result['ood']['exact_accuracy'] >= thresholds['ood_exact_min'],
            ood_ci=result['ood']['wilson95'][0] >= thresholds['ood_wilson95_lower_min'],
            known_families=all(result[f]['exact_accuracy'] >= thresholds['each_known_family_exact_min'] for f in FAMILIES[:3]),
            long_carry=result['long_carry_known']['exact_accuracy'] >= thresholds['long_carry_exact_min'],
            unseen=result['unseen_short']['exact_accuracy'] >= thresholds['unseen_short_exact_min'],
            unseen_ci=result['unseen_short']['wilson95'][0] >= thresholds['unseen_short_wilson95_lower_min'],
            held_probes=probes[model_id]['held_accuracy'] >= thresholds['held_context_probe_accuracy_min'],
            memory_gain=compared['ood']['mean_gain'] >= thresholds['ood_gain_over_memory_min'],
            unseen_memory_gain=compared['unseen_short']['mean_gain'] >= thresholds['unseen_gain_over_memory_min'],
            paired_significance=all(v['p_one_sided'] <= thresholds['paired_p_max'] for v in compared.values()))
    seed_consistency = {kind: all(all(gates[f'r54-{kind}-neural-seed-{seed}-u800'].values())
        for seed in PROTOCOL['seeds']) for kind in PROTOCOL['models']}
    recurrent_comparisons = {str(seed): paired(
        results[f'r54-recurrent-neural-seed-{seed}-u800']['ood']['task_success'],
        results[f'r54-markov-neural-seed-{seed}-u800']['ood']['task_success']) for seed in PROTOCOL['seeds']}
    inspected = checkpoint_audit(destination, seal)
    replay_status = 'UNRUN'
    if replay_mps:
        import torch
        from .model import Transition, adapter
        from .evaluate import predict_rows
        if not torch.backends.mps.is_available():
            raise ValueError('MPS replay unavailable; no CPU fallback')
        for entry in seal['checkpoints']:
            payload = torch.load(destination / entry['path'], map_location='mps', weights_only=True)
            model = Transition(entry['kind']).to('mps')
            model.load_state_dict(payload['model'])
            model.eval()
            replay = list(predict_rows(tasks, adapter(model), entry['model_id'], 'neural', entry['seed']))
            recorded = [r for r in records if r['model_id'] == entry['model_id']]
            if replay != recorded:
                raise ValueError('Independent neural checkpoint replay mismatch')
        replay_status = 'PASS'
    report = dict(evidence_status='VERIFIED', learned_claims=seed_consistency, gates=gates,
        metrics=results, teacher_forced=teacher_metrics, local_probes=probes,
        paired_memory_comparisons=comparisons, exploratory_recurrent_vs_markov=recurrent_comparisons,
        checkpoint_checks=inspected, replay_mps=replay_status,
        independent_ood_tasks=192, oracle_ceiling={'correct': 192, 'tasks': 192},
        numerical_metadata={'metric_dtype': 'python_float64', 'counts_dtype': 'python_int', 'interval': 'Wilson95'},
        limitations=['Finite context probes are dependent diagnostic contexts, not independent arithmetic tasks.',
                     'Input API and source isolation, not an operating-system security sandbox.'])
    if (destination / 'metrics.json').exists():
        saved = json.loads((destination / 'metrics.json').read_text())
        for key in report:
            if key != 'replay_mps' and saved[key] != report[key]:
                raise ValueError('Independent metric recomputation mismatch')
    return report
