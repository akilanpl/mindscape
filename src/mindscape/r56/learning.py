"""Externally gated actual graph-model training. Freeze preparation never trains."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from .curriculum import BUDGETS, SEEDS, digest, prepare_data, exposure_summary

STATUS = 'EXPERIMENTAL — GENERALIZATION NOT YET DEMONSTRATED'


def source_locks(root):
    root = Path(root).resolve()
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*(root / 'src/mindscape/r56').rglob('*.py'),
                             *(root / 'src/mindscape/mind1').rglob('*.py'),
                             *(root / 'tests_native').rglob('*.py'),
                             *(root / 'configs/r56_native').rglob('*.json'),
                             *(root / 'scripts/native').rglob('*.py'),
                             *(root / 'scripts').glob('*native*.py'),
                             *(root / '.github/workflows').glob('mind1-native.yml'),
                             *(root / 'docs/qualification').glob('*.md'),
                             *(root / 'docs/remediation').glob('*.md'),
                             *[p for p in (root / 'deploy/r56-isolation').rglob('*') if p.is_file() and '__pycache__' not in p.parts],
                             *[root / name for name in ('requirements-mind1.txt','pyproject.toml') if (root / name).is_file()]])}


def freeze(root, directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    from .benchmarks import matched_baseline_plan
    datasets = {str(seed): prepare_data(seed) for seed in SEEDS}
    from .evaluation import axis_datasets
    datasets['evaluation_axes'] = axis_datasets()
    manifest = {'version': 'r56-native-preparation-v1', 'source_locks': source_locks(root),
        'data_hash': digest(datasets),
        'exposure_summary': {seed: {str(budget): exposure_summary(data['train'][:budget])
            for budget in BUDGETS} for seed, data in datasets.items() if seed != 'evaluation_axes'},
        'success_threshold_status': 'CANDIDATE_REQUIRES_INDEPENDENT_REVIEW',
        'seeds': list(SEEDS), 'budgets': list(BUDGETS),
        'optimizer': 'Adam', 'learning_rate': .001, 'optimizer_steps': 2000,
        'batch_size': 1, 'model': 'R56Model', 'device': 'cpu',
        'controls': ['exact-rule', 'memorization', 'matched-mlp', 'matched-rnn', 'matched-gru',
                     'graph-concept-on', 'graph-concept-off', 'graph-concept-shuffled',
                     'graph-concept-ablated', 'graph-memory-recurrent'],
        'capacity_target': 44962, 'capacity_relative_tolerance': .05,
        'capacity_plan': matched_baseline_plan(),
        'neural_run_count': 72, 'total_optimizer_steps': 144000,
        'success_thresholds': {'classification_target_accuracy': .90,
            'guided_accuracy_advantage_pp': 5, 'paired_ci95_lower_bound_gt': 0,
            'lower_budgets': [32, 128], 'physics_absolute_error_tolerance': .1,
            'physics_mse_target': .01, 'holm_alpha': .05},
        'prepared_unexecuted_axes': ['composition-depth', 'OOD', 'cross-domain-transfer'],
        'memory_control': 'graph-concept-on stateless versus graph-memory-recurrent detached state',
        'metrics': ['original-domain-accuracy', 'physics-mse', 'verifier-acceptance',
                    'wall-seconds', 'parameter-count'],
        'status': STATUS, 'authorization': 'UNISSUED'}
    manifest['freeze_hash'] = digest(manifest)
    for name, value in [('datasets.json', datasets), ('manifest.json', manifest)]:
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2))
    for role in ('authorization', 'qualification'):
        template = {'role': role, 'freeze_hash': manifest['freeze_hash'], 'issuer': '',
                    'issued': False, 'scope': 'native-r56-training', 'signature_hex': ''}
        (output / f'{role}.UNISSUED.json').write_text(json.dumps(template, indent=2))
    return manifest


def validate_gate(root, manifest, datasets, authorization, qualification, trusted_keys):
    """Must execute before importing torch or creating models/optimizers.

    trusted_keys is externally maintained issuer->Ed25519 public key hex.
    Artifact signatures cover every field except signature_hex. Neither templates
    nor a field claiming qualification can pass without independent signatures.
    """
    frozen = dict(manifest)
    claimed = frozen.pop('freeze_hash', None)
    if claimed != digest(frozen) or source_locks(root) != manifest.get('source_locks'):
        raise PermissionError('Source/protocol freeze mismatch')
    if digest(datasets) != manifest.get('data_hash'):
        raise PermissionError('Data freeze mismatch')
    if not isinstance(trusted_keys, dict) or not trusted_keys:
        raise PermissionError('External trusted signing keys required')
    required = {'src/mindscape/r56/learning.py', 'src/mindscape/r56/curriculum.py',
                'src/mindscape/r56/benchmarks.py', 'src/mindscape/r56/neural.py',
                'src/mindscape/r56/native_domains.py'}
    if not required.issubset(manifest['source_locks']):
        raise PermissionError('Freeze does not bind the complete actual model/training sources')
    issuers = []
    for artifact, role in ((authorization, 'authorization'), (qualification, 'qualification')):
        issuer = artifact.get('issuer')
        if (artifact.get('role') != role or artifact.get('issued') is not True
                or artifact.get('freeze_hash') != claimed
                or artifact.get('scope') != 'native-r56-training' or issuer not in trusted_keys):
            raise PermissionError('External authorization and independent qualification required')
        signed = dict(artifact)
        signature = signed.pop('signature_hex', '')
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_keys[issuer]))
            key.verify(bytes.fromhex(signature), json.dumps(signed, sort_keys=True, separators=(',', ':')).encode())
        except (ImportError, ValueError, TypeError) as exc:
            raise PermissionError('Signature verification unavailable or invalid') from exc
        except Exception as exc:
            raise PermissionError('Invalid external signature') from exc
        issuers.append(issuer)
    if issuers[0] == issuers[1]:
        raise PermissionError('Qualification must use an independent issuer')
    return claimed


def save_checkpoint(model, path, metadata):
    """Safe JSON tensors only; never pickle executable model objects."""
    tensors = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                      'values': value.detach().cpu().tolist()}
               for name, value in model.state_dict().items()}
    payload = {'format': 'r56-json-tensors-v1', 'metadata': metadata,
               'tensors': tensors, 'scientifically_qualified': False}
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True).encode()
    with Path(path).open('xb') as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(model, path, expected_hash):
    path = Path(path)
    if path.stat().st_size > 64_000_000:
        raise ValueError('Checkpoint exceeds bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Checkpoint hash mismatch')
    payload = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
    if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict) or not isinstance(payload.get('tensors'), dict):
        raise ValueError('Checkpoint payload, metadata and tensors must be mappings')
    steps = payload['metadata'].get('optimizer_steps', 0)
    if type(steps) is not int or not 0 <= steps <= 10000000:
        raise ValueError('Checkpoint optimizer_steps must be a bounded nonnegative integer')
    if payload.get('format') != 'r56-json-tensors-v1' or payload.get('scientifically_qualified') is not False:
        raise ValueError('Invalid checkpoint format/status')
    import torch
    state = model.state_dict()
    if set(payload['tensors']) != set(state):
        raise ValueError('Checkpoint keys differ')
    restored = {}
    for name, expected in state.items():
        entry = payload['tensors'][name]
        if not isinstance(entry, dict):
            raise ValueError('Checkpoint tensor entry must be a mapping')
        if entry['shape'] != list(expected.shape) or entry['dtype'] != str(expected.dtype):
            raise ValueError('Checkpoint tensor schema differs')
        tensor = torch.tensor(entry['values'], dtype=expected.dtype)
        if tensor.shape != expected.shape or not torch.isfinite(tensor).all():
            raise ValueError('Invalid tensor shape/values')
        restored[name] = tensor
    model.load_state_dict(restored, strict=True)
    model.eval()
    return payload['metadata']


def train(root, package, authorization, qualification, trusted_keys, output, seed=17, budget=32, control='graph-concept-on'):
    if Path(output).exists() or Path(str(output) + '.failure.json').exists():
        raise FileExistsError('Refuse to overwrite checkpoint or failure evidence')
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    # Legacy PR28 bundles include held-out labels and are not scientifically qualified.
    # Do not decode them into the fitting process, even if old signatures exist.
    from .qualification import validate_scientific_gate, load_training_only, CONTROLS
    scientific = validate_scientific_gate(root, package, authorization, qualification, trusted_keys)
    if seed not in SEEDS or budget not in BUDGETS or control not in CONTROLS:
        raise PermissionError('Seed/budget/control outside frozen campaign')
    training_rows = load_training_only(package, scientific, seed)
    freeze_hash = scientific['package_sha256']
    # Fixed scientific workload remains 2000 updates per fit, not a reduced pilot.
    manifest = {**scientific, 'learning_rate': .001, 'optimizer_steps': 2000, 'capacity_target': 44962}
    # All neural imports/construction and optimizer steps are strictly below the gate.
    import torch
    from .neural import R56Model, ModelConfig
    from .native_domains import pack_problem
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if control in ('exact-rule', 'memorization'):
        raise ValueError('Non-neural controls require no optimizer and use benchmarks helpers')
    if control in ('matched-mlp', 'matched-rnn', 'matched-gru'):
        from .benchmarks import matched_baseline
        model = matched_baseline(control.removeprefix('matched-'), manifest['capacity_target'], seed)
    else:
        model = R56Model(ModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=manifest['learning_rate'])
    rows = training_rows[:budget]
    began, losses, memory, prior_episode = time.monotonic(), [], None, None
    try:
        model.train()
        for step in range(manifest['optimizer_steps']):
            row = rows[step % len(rows)]
            graph = pack_problem(row['problem'], device='cpu')
            from .benchmarks import controlled_graph
            graph = controlled_graph(graph, control)
            episode = row.get('episode_id')
            if episode is None or episode != prior_episode:
                memory = None
            prior_episode = episode
            prediction = model(graph, memory=memory, stateless=control != 'graph-memory-recurrent')
            memory = prediction['memory'].detach() if control == 'graph-memory-recurrent' else None
            labels, domain = row['labels'], row['problem']['domain']
            if domain == 'math':
                loss = sum(torch.nn.functional.cross_entropy(prediction[key].reshape(1, -1),
                    torch.tensor([labels[label]], dtype=torch.long)) for key, label in
                    [('math_digit_logits', 'math_digit'), ('math_carry_logits', 'math_carry')])
            elif domain == 'coding':
                loss = torch.nn.functional.cross_entropy(prediction['coding_logits'][:len(row['problem']['candidates'])].reshape(1, -1),
                    torch.tensor([labels['coding_index']], dtype=torch.long))
            else:
                loss = torch.nn.functional.mse_loss(prediction['physics_delta'],
                    torch.tensor(labels['physics_delta'], dtype=torch.float32))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite training gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        from .benchmarks import model_accounting
        import resource
        import sys
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata = {'model_accounting': model_accounting(model),
            'process_lifetime_peak_rss_bytes': int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024),
            'peak_memory_note': 'Process lifetime RSS, not isolated campaign activation peak.',
            'freeze_hash': freeze_hash, 'seed': seed, 'budget': budget, 'control': control,
            'optimizer_steps': len(losses), 'losses': losses,
            'wall_seconds': time.monotonic() - began,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'device': 'cpu', 'torch_version': torch.__version__, 'status': STATUS, 'qualification': 'TRAINED_NOT_INDEPENDENTLY_EVALUATED'}
        return {'checkpoint_sha256': save_checkpoint(model, output, metadata), 'metadata': metadata}
    except Exception as exc:
        with Path(str(output) + '.failure.json').open('x') as stream:
            stream.write(json.dumps({'freeze_hash': freeze_hash,
                'completed_steps': len(losses), 'error_type': type(exc).__name__, 'status': 'FAILED'}))
        raise


def main():
    parser = argparse.ArgumentParser(description='R5.6 nontraining freeze preparation')
    parser.add_argument('command', choices=['freeze'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()


def evaluate_model(model, rows, control='graph-concept-on'):
    """Original held-out outputs only. No repair, optimizer, or success declaration."""
    import torch
    from .native_domains import pack_problem
    if len(rows) > 4096:
        raise ValueError('Evaluation row budget exceeded')
    began, records, memory, prior_episode = time.monotonic(), [], None, None
    model.eval()
    with torch.no_grad():
        for row in rows:
            from .benchmarks import controlled_graph
            episode = row.get('episode_id')
            if episode is None or episode != prior_episode:
                memory = None
            prior_episode = episode
            output = model(controlled_graph(pack_problem(row['problem'], device='cpu'), control),
                           memory=memory, stateless=control != 'graph-memory-recurrent')
            memory = output['memory'].detach() if control == 'graph-memory-recurrent' else None
            labels, domain = row['labels'], row['problem']['domain']
            if domain == 'math':
                prediction = {'math_digit': int(output['math_digit_logits'].argmax()),
                              'math_carry': int(output['math_carry_logits'].argmax())}
                metric = float(all(prediction[k] == labels[k] for k in prediction))
                metric_name = 'exact-domain-accuracy'
            elif domain == 'coding':
                prediction = {'coding_index': int(output['coding_logits'][:len(row['problem']['candidates'])].argmax())}
                metric = float(prediction['coding_index'] == labels['coding_index'])
                metric_name = 'exact-domain-accuracy'
            else:
                prediction = {'physics_delta': output['physics_delta'].cpu().tolist()}
                metric = sum((a - b) ** 2 for a, b in zip(prediction['physics_delta'], labels['physics_delta'])) / 2
                metric_name = 'physics-mse'
            records.append({'identity': row['identity'], 'domain': domain,
                            'original_prediction': prediction, 'metric': metric,
                            'metric_name': metric_name, 'repaired': False})
    return {'records': records, 'wall_seconds': time.monotonic() - began,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'scientifically_qualified': False, 'status': STATUS}
