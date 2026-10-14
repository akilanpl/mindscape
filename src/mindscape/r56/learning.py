"""Externally gated actual graph-model training. Freeze preparation never trains."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from .curriculum import BUDGETS, SEEDS, digest, prepare_data

STATUS = 'EXPERIMENTAL — GENERALIZATION NOT YET DEMONSTRATED'


def source_locks(root):
    root = Path(root).resolve()
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'src/mindscape/r56').glob('*.py'))}


def freeze(root, directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    datasets = {str(seed): prepare_data(seed) for seed in SEEDS}
    manifest = {'version': 'r56-native-preparation-v1', 'source_locks': source_locks(root),
        'data_hash': digest(datasets), 'seeds': list(SEEDS), 'budgets': list(BUDGETS),
        'optimizer': 'Adam', 'learning_rate': .001, 'optimizer_steps': 2000,
        'batch_size': 1, 'model': 'R56Model', 'device': 'cpu',
        'controls': ['exact-rule', 'memorization', 'mlp', 'gru', 'graph-concept-on',
                     'graph-concept-off', 'graph-concept-shuffled', 'graph-concept-ablated'],
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
    Path(path).write_text(json.dumps(payload, allow_nan=False, sort_keys=True))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(model, path, expected_hash):
    path = Path(path)
    if path.stat().st_size > 64_000_000:
        raise ValueError('Checkpoint exceeds bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Checkpoint hash mismatch')
    payload = json.loads(raw)
    if payload.get('format') != 'r56-json-tensors-v1' or payload.get('scientifically_qualified') is not False:
        raise ValueError('Invalid checkpoint format/status')
    import torch
    state = model.state_dict()
    if set(payload['tensors']) != set(state):
        raise ValueError('Checkpoint keys differ')
    restored = {}
    for name, expected in state.items():
        entry = payload['tensors'][name]
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
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    datasets = json.loads((package / 'datasets.json').read_text())
    freeze_hash = validate_gate(root, manifest, datasets, authorization, qualification, trusted_keys)
    if seed not in manifest['seeds'] or budget not in manifest['budgets'] or control not in manifest['controls']:
        raise PermissionError('Seed/budget outside frozen campaign')
    # All neural imports/construction and optimizer steps are strictly below the gate.
    import torch
    from .neural import R56Model, ModelConfig
    from .native_domains import pack_problem
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if control in ('exact-rule', 'memorization'):
        raise ValueError('Non-neural controls require no optimizer and use benchmarks helpers')
    if control in ('mlp', 'gru'):
        from .benchmarks import pooled_baseline
        model = pooled_baseline(control)
    else:
        model = R56Model(ModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=manifest['learning_rate'])
    rows = datasets[str(seed)]['train'][:budget]
    began, losses = time.monotonic(), []
    try:
        model.train()
        for step in range(manifest['optimizer_steps']):
            row = rows[step % len(rows)]
            graph = pack_problem(row['problem'], device='cpu')
            if control != 'graph-concept-on' and control not in ('mlp', 'gru'):
                from dataclasses import replace
                if control == 'graph-concept-off':
                    graph = replace(graph, concept_ids=graph.concept_ids[:0])
                elif control == 'graph-concept-shuffled':
                    graph = replace(graph, concept_ids=(graph.concept_ids * 17 + 3) % 32)
                elif control == 'graph-concept-ablated':
                    graph = replace(graph, concept_ids=torch.zeros_like(graph.concept_ids))
            prediction = model(graph, stateless=True)
            labels, domain = row['labels'], row['problem']['domain']
            if domain == 'math':
                loss = sum(torch.nn.functional.cross_entropy(prediction[key].reshape(1, -1),
                    torch.tensor([labels[label]], dtype=torch.long)) for key, label in
                    [('math_digit_logits', 'math_digit'), ('math_carry_logits', 'math_carry')])
            elif domain == 'coding':
                loss = torch.nn.functional.cross_entropy(prediction['coding_logits'].reshape(1, -1),
                    torch.tensor([labels['coding_index']], dtype=torch.long))
            else:
                loss = torch.nn.functional.mse_loss(prediction['physics_delta'],
                    torch.tensor(labels['physics_delta'], dtype=torch.float32))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metadata = {'freeze_hash': freeze_hash, 'seed': seed, 'budget': budget, 'control': control,
            'optimizer_steps': len(losses), 'losses': losses,
            'wall_seconds': time.monotonic() - began,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'device': 'cpu', 'torch_version': torch.__version__, 'status': STATUS, 'qualification': 'TRAINED_NOT_INDEPENDENTLY_EVALUATED'}
        return {'checkpoint_sha256': save_checkpoint(model, output, metadata), 'metadata': metadata}
    except Exception as exc:
        Path(str(output) + '.failure.json').write_text(json.dumps({'freeze_hash': freeze_hash,
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


def evaluate_model(model, rows):
    """Original held-out outputs only. No repair, optimizer, or success declaration."""
    import torch
    from .native_domains import pack_problem
    if len(rows) > 4096:
        raise ValueError('Evaluation row budget exceeded')
    began, records = time.monotonic(), []
    model.eval()
    with torch.no_grad():
        for row in rows:
            output = model(pack_problem(row['problem'], device='cpu'), stateless=True)
            labels, domain = row['labels'], row['problem']['domain']
            if domain == 'math':
                prediction = {'math_digit': int(output['math_digit_logits'].argmax()),
                              'math_carry': int(output['math_carry_logits'].argmax())}
                metric = float(all(prediction[k] == labels[k] for k in prediction))
                metric_name = 'exact-domain-accuracy'
            elif domain == 'coding':
                prediction = {'coding_index': int(output['coding_logits'].argmax())}
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
