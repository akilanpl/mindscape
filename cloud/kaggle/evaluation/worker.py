"""Hash-pinned CPU evaluator. Access isolation requires external control-plane evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def verified_path(root, entry):
    path = Path(entry['path'])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or digest(path) != entry['sha256']:
        raise ValueError('Immutable input hash mismatch: ' + str(path))
    return path


def require_isolation(evidence, *, trusted_control_plane_verifier=None):
    """Validate a trusted controller's attestation, never infer ACLs from folders.

    This validates evidence structure; only a trusted control-plane integration can
    authenticate the attester. User-authored JSON alone is not ACL verification.
    """
    if trusted_control_plane_verifier is None or trusted_control_plane_verifier(evidence) is not True:
        raise PermissionError('INDEPENDENT_ENVIRONMENT_UNVERIFIED: no authenticated provider ACL verifier configured')
    # Provider adapter checks distinct frozen executions and actual read-only input
    # access. The owner/controller may control both accounts; the threat boundary
    # is the trainer runtime, not an adversarial account owner changing future runs.
    from .isolation import credential_guard
    credential_guard()


def overlap_check(training, evaluation):
    from mindscape.r57.evaluation import canonical_id
    train_ids = [canonical_id(row['problem']) for row in training]
    final_ids = [canonical_id(row['problem']) for row in evaluation]
    if len(set(final_ids)) != len(final_ids):
        raise ValueError('Duplicate normalized evaluation cases')
    overlap = set(train_ids) & set(final_ids)
    if overlap:
        raise ValueError('Training/evaluation overlap: ' + str(len(overlap)))
    return {'training_count': len(train_ids), 'evaluation_count': len(final_ids),
            'training_normalized_duplicates': len(train_ids) - len(set(train_ids)),
            'evaluation_normalized_duplicates': 0,
            'normalized_overlap': 0, 'normalization': 'r57 canonical_id'}


def load_model(entry, checkpoint):
    """Safe JSON loading without importing learning, runner or optimizer modules."""
    import torch
    from mindscape.r57c.ordinal_model import make_model
    if entry['version'] != 'R57c':
        raise ValueError('This worker accepts frozen R57c only; cross-version adapters require reviewed source pins')
    model = make_model(entry['condition'], entry['seed']).cpu().eval()
    if Path(checkpoint).stat().st_size > 64_000_000:
        raise ValueError('Checkpoint exceeds safe JSON bound')
    payload = read_json(checkpoint)
    if payload.get('format') != 'r56-json-tensors-v1' or payload.get('scientifically_qualified') is not False:
        raise ValueError('Safe unqualified JSON checkpoint required')
    metadata = payload['metadata']
    if (metadata.get('seed'), metadata.get('condition'), metadata.get('optimizer_steps')) != (entry['seed'], entry['condition'], 2000):
        raise ValueError('Checkpoint run identity/update budget mismatch')
    state = model.state_dict()
    if set(payload['tensors']) != set(state):
        raise ValueError('Checkpoint keys differ')
    restored = {}
    for name, expected in state.items():
        value = payload['tensors'][name]
        if value['shape'] != list(expected.shape) or value['dtype'] != str(expected.dtype):
            raise ValueError('Checkpoint tensor schema differs')
        tensor = torch.tensor(value['values'], dtype=expected.dtype, device='cpu')
        if tensor.shape != expected.shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Invalid checkpoint tensor')
        restored[name] = tensor
    model.load_state_dict(restored, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    def predict(problem):
        with torch.inference_mode():
            output = model.predict(problem)
        if any(value.device.type != 'cpu' for value in output.values() if torch.is_tensor(value)):
            raise RuntimeError('CPU inference required')
        return output
    return predict


def raw_values(output):
    result = {}
    for key, value in output.items():
        if hasattr(value, 'detach'):
            value = value.detach().cpu().tolist()
        if isinstance(value, (int, float, list, tuple, bool)):
            result[key] = value
    # Fails on nonfinite predictions rather than silently repairing them.
    json.dumps(result, allow_nan=False)
    return result


def evaluate(predict, rows):
    from mindscape.r57.evaluation import evaluate as score
    raw = []
    def capture(problem):
        output = predict(deepcopy(problem))
        raw.append(raw_values(output))
        return output
    result = score(capture, rows)
    for row, record, values in zip(rows, result['records'], raw):
        record['problem'] = deepcopy(row['problem'])
        record['raw_output'] = values
    return result


def perturbations(predict, rows):
    """Derived probes are preserved separately; they never repair primary outputs."""
    from mindscape.r57.evaluation import analytic, decode, canonical_id
    records = []
    for row in rows:
        original = row['problem']
        variants = []
        domain = original['domain']
        if domain == 'math':
            for key in ('left', 'right'):
                changed = deepcopy(original)
                changed[key] = (changed[key] + 1) % 10
                variants.append((key + '_plus_one_mod10', changed))
            changed = deepcopy(original)
            key = next((k for k in ('incoming', 'carry_in', 'borrow_in') if k in changed), 'incoming')
            changed[key] = 1 - changed.get(key, 0)
            variants.append(('incoming_flip', changed))
        elif domain == 'physics':
            for key in ('mass', 'dt'):
                changed = deepcopy(original)
                changed[key] = changed.get(key, 1.) * 2
                variants.append((key + '_times_two', changed))
            for axis in (0, 1):
                changed = deepcopy(original)
                changed.setdefault('force', [0., 0.])[axis] += 1.
                variants.append(('force_axis_' + str(axis) + '_plus_one', changed))
        else:
            for shift in range(1, len(original['candidates'])):
                changed = deepcopy(original)
                changed['candidates'] = original['candidates'][shift:] + original['candidates'][:shift]
                variants.append(('candidate_rotation_' + str(shift), changed))
            changed = deepcopy(original)
            position = changed['candidates'].index(changed['requested_name'])
            changed['requested_name'] = changed['candidates'][(position + 1) % len(changed['candidates'])]
            variants.append(('requested_name_change', changed))
        base = decode(predict(deepcopy(original)), original)
        for name, changed in variants:
            output = predict(deepcopy(changed))
            prediction = decode(output, changed)
            record = {'identity': canonical_id(original), 'probe': name,
                      'original_prediction': base, 'problem': changed,
                      'raw_output': raw_values(output), 'prediction': prediction,
                      'analytic_reference': analytic(changed), 'repaired': False}
            if domain == 'coding':
                first = base['coding_index']; second = prediction['coding_index']
                record['same_selected_name'] = (original['candidates'][first] == changed['candidates'][second]) if 0 <= first < len(original['candidates']) and 0 <= second < len(changed['candidates']) else False
            records.append(record)
    return records


def baseline_results(training, rows):
    from mindscape.r57.evaluation import analytic
    math_labels = {}
    for operation in ('add_digit', 'borrow_digit'):
        labels = Counter((r['labels']['math_digit'], r['labels']['math_carry']) for r in training if r['problem']['domain'] == 'math' and r['problem']['operation'] == operation)
        math_labels[operation] = labels.most_common(1)[0][0] if labels else (0, 0)
    physics = [r['labels']['physics_delta'] for r in training if r['problem']['domain'] == 'physics']
    constant = [sum(v[a] for v in physics) / len(physics) for a in (0, 1)] if physics else [0., 0.]
    coding = Counter(r['labels']['coding_index'] for r in training if r['problem']['domain'] == 'coding')
    position = coding.most_common(1)[0][0] if coding else 0
    def majority(problem):
        if problem['domain'] == 'math':
            digit, carry = math_labels[problem['operation']]
            return {'math_digit': digit, 'math_carry': carry}
        if problem['domain'] == 'coding':
            return {'coding_index': position}
        return {'physics_delta': constant}
    return {'analytic_rule': evaluate(analytic, rows), 'training_majority_constant': evaluate(majority, rows),
            'classification': 'Deterministic/non-neural controls; never attributed to native model'}


def run(manifest_path, output, *, trusted_control_plane_verifier=None):
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent
    manifest_hash = digest(manifest_path); manifest = read_json(manifest_path)
    if manifest.get('schema') != 'mindscape-independent-evaluation-v1':
        raise ValueError('Unknown evaluation manifest')
    isolation_path = verified_path(root, manifest['isolation_evidence'])
    require_isolation(read_json(isolation_path), trusted_control_plane_verifier=trusted_control_plane_verifier)
    if not manifest.get('source_files') or not manifest.get('evaluations') or not manifest.get('checkpoints'):
        raise ValueError('Pinned sources and evaluation cases required')
    # Verify every immutable input before reading evaluation labels or importing a model.
    pins = manifest['source_files'] + manifest['data_files'] + manifest['checkpoints']
    paths = {entry['path']: verified_path(root, entry) for entry in pins}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('mindscape.r57c.runner', 'mindscape.r56.learning', 'mindscape.r56.compact_gpu')):
            raise RuntimeError('Training helper loaded in evaluator: ' + name)
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    destination = Path(output); destination.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        # Global union catches cross-seed train/final leakage, not merely within-seed overlap.
        all_training = []
        for path in sorted({job['training_path'] for job in manifest['evaluations']}):
            all_training.extend(read_json(paths[path]))
        from mindscape.r57.evaluation import canonical_id
        train_by_id = {canonical_id(row['problem']): row for row in all_training}
        for job in manifest['evaluations']:
            training = read_json(paths[job['training_path']]); rows = read_json(paths[job['evaluation_path']])
            audit = overlap_check(training, rows)
            audit['global_training_union'] = overlap_check(list(train_by_id.values()), rows)
            runs = []
            for entry in manifest['checkpoints']:
                if entry['seed'] != job['seed']:
                    continue
                checkpoint = paths[entry['path']]
                predict = load_model(entry, checkpoint)
                pinned_sources = {paths[item['path']].resolve() for item in manifest['source_files']}
                for name, module in tuple(sys.modules.items()):
                    if name.startswith('mindscape.') and getattr(module, '__file__', None):
                        if Path(module.__file__).resolve() not in pinned_sources:
                            raise ValueError('Imported source is not pinned: ' + name)
                result = evaluate(predict, rows)
                result.update(seed=entry['seed'], condition=entry['condition'], version=entry['version'],
                              checkpoint_sha256=entry['sha256'], perturbations=perturbations(predict, rows))
                filename = f"{entry['seed']}-{entry['condition']}-{entry['version']}.json"
                with (destination / filename).open('x') as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
                runs.append({'seed': entry['seed'], 'condition': entry['condition'], 'version': entry['version'],
                             'metrics': result['metrics'], 'artifact': filename, 'sha256': digest(destination / filename)})
            if not runs:
                raise ValueError('No checkpoint for evaluation seed')
            results.append({'seed': job['seed'], 'distribution_id': job['distribution_id'],
                            'historical_evaluation_reuse': job['historical_evaluation_reuse'],
                            'evaluation_sha256': digest(paths[job['evaluation_path']]),
                            'training_sha256': digest(paths[job['training_path']]),
                            'source_files': manifest['source_files'],
                            'overlap_audit': audit, 'runs': runs, 'baselines': baseline_results(training, rows)})
        for entry in pins:
            verified_path(root, entry)
        if digest(manifest_path) != manifest_hash or digest(isolation_path) != manifest['isolation_evidence']['sha256']:
            raise ValueError('Manifest/isolation evidence changed during evaluation')
        receipt = {'status': 'COMPLETED_CLOUD_EVALUATION', 'device': 'cpu',
                   'evaluation_optimizer_updates': 0, 'qualification': False, 'selection_performed': False,
                   'manifest_sha256': manifest_hash, 'isolation_evidence_sha256': digest(isolation_path),
                   'isolation_trust': 'Controller-authenticated provider metadata/source plus effective runtime checks; account owner remains trusted',
                   'runtime_isolation_audit': getattr(trusted_control_plane_verifier, 'runtime_audit', None),
                   'torch': str(torch.__version__), 'python': sys.version, 'platform': platform.platform(),
                   'results': results}
        with (destination / 'evaluation_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        return receipt
    except BaseException as exc:
        with (destination / 'failure.json').open('x') as stream:
            json.dump({'status': 'FAILED', 'error_type': type(exc).__name__, 'evaluation_optimizer_updates': 0}, stream)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == '__main__':
    main()
