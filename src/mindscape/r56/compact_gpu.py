"""Separate proposed compact MPS campaign. Historical gates remain untouched.

No preparation/template can authorize optimizer construction. Real deployment and
scientific artifacts must be issued independently outside this worker's authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
SEEDS = (17, 29, 43)

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

CONDITIONS = ('graph-concept-on', 'graph-concept-off', 'matched-gru')
SCOPE = 'r56-compact-mps-9fits-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_json(path, limit=32_000_000):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise PermissionError('Require bounded regular non-symlink artifact')
    return json.loads(path.read_bytes())


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, destination):
    """Nontraining proposal. Evaluator data stays separate from training handoff."""
    from .compact_evaluation import build_protected_evaluation
    from .curriculum import prepare_data, exposure_summary
    from .learning import source_locks
    from .benchmarks import matched_baseline_plan
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    trainer, evaluator = destination / 'trainer', destination / 'evaluator'
    trainer.mkdir(); evaluator.mkdir()
    from .compact_bundle import build_bundle
    runtime_files = build_bundle(root, trainer / 'runtime')
    train_hashes, eval_hashes, counts, domain_updates = {}, {}, {}, {}
    for seed in SEEDS:
        data = prepare_data(seed)
        rows = data['train'][:128]
        write_new(trainer / f'train-{seed}.json', rows)
        protected = build_protected_evaluation(rows, data['dev'], data['test'])
        write_new(evaluator / f'eval-{seed}.json', protected)
        train_hashes[str(seed)] = file_hash(trainer / f'train-{seed}.json')
        eval_hashes[str(seed)] = file_hash(evaluator / f'eval-{seed}.json')
        counts[str(seed)] = exposure_summary(rows)
        domain_updates[str(seed)] = {domain: sum(1 for update in range(2000)
            if rows[update % 128]['problem']['domain'] == domain)
            for domain in ('math', 'coding', 'physics')}
    manifest = {'scope': SCOPE, 'status': 'PROPOSED_UNAPPROVED',
        'source_locks': source_locks(root), 'runtime_files': runtime_files,
        'training_files_sha256': train_hashes,
        'evaluation_references_sha256': eval_hashes, 'exposure_summary': counts,
        'seeds': list(SEEDS), 'conditions': list(CONDITIONS), 'total_records_per_seed': 128,
        'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 1,
        'updates_per_fit': 2000, 'fits': 9, 'total_updates': 18000,
        'workload_rows': [{'seed': seed, 'condition': condition, 'updates': 2000, 'records': 128}
                          for seed in SEEDS for condition in CONDITIONS],
        'update_counts_by_record_index': {str(index): 16 if index < 80 else 15 for index in range(128)},
        'domain_update_counts_per_fit': domain_updates,
        'device': 'mps', 'dtype': 'float32', 'cpu_fallback': False,
        'graph_config': 'PR30-default-R56Model-44962',
        'matched_capacity_plan': matched_baseline_plan(),
        'checkpoint_selection': 'final-update-2000-only-no-heldout-feedback',
        'state_policy': 'stateless-false-reset-memory-per-independent-record',
        'historical_proposal': '72-fits-144000-updates-remains-separate-unexecuted',
        'metrics': ['loss-curve', 'initial-final-fixed-training-mean-by-domain',
                    'math-digit-accuracy', 'math-carry-accuracy', 'math-joint-digit-carry-accuracy',
                    'math-composition-exact-by-depth', 'coding-valid-candidate-accuracy',
                    'coding-accuracy-by-requested-candidate-position-and-structure',
                    'physics-delta-mse', 'physics-accumulated-ood-error', 'all-metrics-per-seed'],
        'candidate_criteria': {'status': 'REQUIRES_INDEPENDENT_REVIEW',
            'classification_target': .90, 'physics_mse_target': .01,
            'advantage_family': '3 domains x (concept-on vs off, concept-on vs matched-GRU); Holm6 alpha.05; paired problem-family intervals and seed variation; 3 seeds not strong reproducibility evidence',
            'loss_decrease': 'actual initial versus final logged loss; domain composition must be reported'},
        'success': 'Results require independent audit; no automatic product qualification.'}
    manifest['package_sha256'] = digest(manifest)
    write_new(destination / 'manifest.json', manifest)
    # Trainer reference manifest has no evaluator targets. Isolation still requires operator.
    write_new(trainer / 'manifest.json', manifest)
    for role in ('authorization', 'qualification', 'readiness', 'deployment'):
        write_new(destination / f'{role}.UNISSUED.json', {
            'scope': SCOPE, 'role': role, 'issued': False,
            'package_sha256': manifest['package_sha256'], 'issuer': '', 'signature_hex': ''})
    return manifest



def stage_training(package, destination, seed):
    """Read-only package copying, no approval or OS-isolation claim. Exact one-seed handoff."""
    import shutil
    package, destination = Path(package), Path(destination)
    manifest = read_json(package / 'manifest.json')
    if seed not in SEEDS:
        raise ValueError('Unsupported compact seed')
    frozen = dict(manifest); identity = frozen.pop('package_sha256', None)
    if digest(frozen) != identity:
        raise ValueError('Package identity mismatch')
    source = package / 'trainer'
    if runtime_locks(source / 'runtime') != manifest['runtime_files']:
        raise ValueError('Prepared runtime differs')
    train = source / f'train-{seed}.json'
    if file_hash(train) != manifest['training_files_sha256'][str(seed)]:
        raise ValueError('Prepared training data differs')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'runtime', destination / 'runtime')
    shutil.copyfile(train, destination / train.name)
    write_new(destination / 'manifest.json', manifest)
    return {'seed': seed, 'package_sha256': identity, 'stage': str(destination.resolve()),
            'status': 'STAGED_NOT_DEPLOYMENT_QUALIFIED', 'optimizer_updates': 0}

def _trusted_config(path):
    """Worker must not own/edit its trust root or any parent directory."""
    path = Path(path).resolve(strict=True)
    for current in (path, *path.parents):
        info = current.stat()
        if info.st_uid == os.getuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError('Trust root must be independently operator-owned and non-writable by worker')
    config = read_json(path, 128_000)
    if config.get('scope') != SCOPE or not config.get('issuers'):
        raise PermissionError('Invalid operator trust configuration')
    return config


def _signed(artifact, role, identity, trust):
    issuer = artifact.get('issuer')
    trusted = trust['issuers'].get(issuer, {})
    if (artifact.get('role') != role or artifact.get('scope') != SCOPE
            or artifact.get('issued') is not True or artifact.get('package_sha256') != identity
            or role not in trusted.get('roles', [])):
        raise PermissionError('Externally issued exact-package artifact required: ' + role)
    body = dict(artifact)
    signature = body.pop('signature_hex', '')
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted['public_key_hex']))
        key.verify(bytes.fromhex(signature), canonical(body))
    except Exception as exc:
        raise PermissionError('External signature invalid or verifier unavailable: ' + role) from exc
    return issuer



def runtime_locks(root):
    """Exact minimal-runtime inventory; never reads scientific/evaluator source."""
    root = Path(root).resolve(strict=True)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise PermissionError('Runtime symlinks forbidden')
        if '__pycache__' in path.parts:
            continue
        if path.is_file():
            result[str(path.relative_to(root))] = file_hash(path)
    return result

def validate_gate(root, manifest, artifacts, trust_path, stage, seed, condition):
    """No torch import, model, optimizer, heldout load, or local trust-root creation."""
    frozen = dict(manifest)
    identity = frozen.pop('package_sha256', None)
    if digest(frozen) != identity or manifest.get('scope') != SCOPE:
        raise PermissionError('Compact package identity mismatch')
    if runtime_locks(root) != manifest.get('runtime_files'):
        raise PermissionError('Minimal training runtime source lock mismatch')
    if Path(root).resolve() != Path(stage).resolve() / 'runtime':
        raise PermissionError('Worker source must be exact staged minimal runtime')
    if (manifest.get('seeds') != list(SEEDS) or manifest.get('conditions') != list(CONDITIONS)
            or manifest.get('fits') != 9 or manifest.get('dtype') != 'float32'
            or manifest.get('updates_per_fit') != 2000 or manifest.get('total_updates') != 18000
            or manifest.get('total_records_per_seed') != 128 or manifest.get('device') != 'mps'
            or manifest.get('cpu_fallback') is not False or manifest.get('batch_size') != 1
            or manifest.get('optimizer') != 'Adam' or manifest.get('learning_rate') != .001
            or manifest.get('state_policy') != 'stateless-false-reset-memory-per-independent-record'):
        raise PermissionError('Compact immutable workload differs')
    if seed not in SEEDS or condition not in CONDITIONS:
        raise PermissionError('Fit outside compact proposal')
    if Path(__file__).resolve() != Path(root).resolve() / 'src/mindscape/r56/compact_gpu.py':
        raise PermissionError('Worker must execute the exact staged minimal source')
    trust = _trusted_config(trust_path)
    issuers = {role: _signed(artifacts.get(role, {}), role, identity, trust)
               for role in ('authorization', 'qualification', 'readiness', 'deployment')}
    if (issuers['authorization'] == issuers['qualification'] or
            trust['issuers'][issuers['authorization']]['public_key_hex'] ==
            trust['issuers'][issuers['qualification']]['public_key_hex']):
        raise PermissionError('Authorization and qualification require distinct independent issuers')
    deployment, readiness = artifacts['deployment'], artifacts['readiness']
    stage = Path(stage).resolve(strict=True)
    expected_receipts = {'deployment_sha256': digest(deployment), 'readiness_sha256': digest(readiness)}
    for role in ('authorization', 'qualification'):
        if any(artifacts[role].get(key) != value for key, value in expected_receipts.items()):
            raise PermissionError('Scientific artifacts do not bind exact deployment/readiness receipts')
    if (deployment.get('worker_uid') != os.getuid() or deployment.get('training_stage') != str(stage)
            or deployment.get('training_only_access_enforced') is not True
            or deployment.get('evaluator_access_denied') is not True
            or deployment.get('operator_verified_os_boundary') is not True
            or deployment.get('operator_evidence_sha256') != trust.get('deployment_evidence_sha256')
            or not trust.get('deployment_evidence_sha256')
            or deployment.get('training_entrypoint') != 'mindscape.r56.compact_gpu:run_fit'):
        raise PermissionError('Actual fitting deployment not independently qualified')
    if (readiness.get('device') != 'mps' or readiness.get('actual_r56_forward_backward') is not True
            or readiness.get('fallback_disabled') is not True
            or readiness.get('environment_receipt_sha256') != trust.get('readiness_evidence_sha256')
            or not trust.get('readiness_evidence_sha256')):
        raise PermissionError('Actual GPU readiness receipt absent')
    allowed = {'manifest.json', f'train-{seed}.json', 'runtime'}
    if {p.name for p in stage.iterdir()} != allowed:
        raise PermissionError('Worker handoff must contain only this seed training file and manifest')
    for entry in stage.iterdir():
        if entry.name == 'runtime' and entry.is_dir() and not entry.is_symlink():
            continue
        if entry.is_symlink() or not entry.is_file():
            raise PermissionError('Unsafe handoff file')
    if file_hash(stage / f'train-{seed}.json') != manifest['training_files_sha256'][str(seed)]:
        raise PermissionError('Training data identity differs')
    if read_json(stage / 'manifest.json') != manifest:
        raise PermissionError('Staged manifest differs')
    return identity


def domain_loss(output, row, torch, device):
    labels, domain = row['labels'], row['problem']['domain']
    if domain == 'math':
        return sum(torch.nn.functional.cross_entropy(output[key][None],
            torch.tensor([labels[label]], dtype=torch.long, device=device))
            for key, label in [('math_digit_logits', 'math_digit'), ('math_carry_logits', 'math_carry')])
    if domain == 'coding':
        count = len(row['problem']['candidates'])
        if not 0 <= labels['coding_index'] < count:
            raise ValueError('Coding label outside production candidate support')
        return torch.nn.functional.cross_entropy(output['coding_logits'][:count][None],
            torch.tensor([labels['coding_index']], dtype=torch.long, device=device))
    if domain != 'physics':
        raise ValueError('Unsupported training domain')
    return torch.nn.functional.mse_loss(output['physics_delta'],
        torch.tensor(labels['physics_delta'], dtype=torch.float32, device=device))



def training_loss_summary(model, graphs, rows, torch, device):
    """Fixed same128 training-only mean at actual initial/final checkpoint weights."""
    losses = {domain: [] for domain in ('math', 'coding', 'physics')}
    previous = model.training
    model.eval()
    with torch.no_grad():
        for graph, row in zip(graphs, rows):
            output = model(graph, memory=None, stateless=False)
            loss = domain_loss(output, row, torch, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite fixed training-set loss')
            losses[row['problem']['domain']].append(float(loss.detach().cpu()))
    model.train(previous)
    return {'per_domain_mean': {domain: sum(values) / len(values) if values else None
                for domain, values in losses.items()},
            'per_domain_count': {domain: len(values) for domain, values in losses.items()},
            'mixed_loss_mean': sum(sum(values) for values in losses.values()) / len(rows),
            'scope': 'Same frozen128 training records; not heldout qualification.'}

def run_fit(root, stage, artifacts, trust_path, output_directory, seed, condition):
    """Actual worker is executable only after external exact-package gate passes."""
    stage = Path(stage)
    manifest = read_json(stage / 'manifest.json')
    identity = validate_gate(root, manifest, artifacts, trust_path, stage, seed, condition)
    output_directory = Path(output_directory).resolve()
    export_root = Path(artifacts['deployment'].get('output_root', '')).resolve()
    if not artifacts['deployment'].get('output_root') or output_directory != export_root / f'{seed}-{condition}':
        raise PermissionError('Output path must match signed deployment exact fit destination')
    output_directory.mkdir(parents=True, exist_ok=False)  # never overwrite any fit/failure
    updates, losses, began = 0, [], time.monotonic()
    try:
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise RuntimeError('MPS CPU fallback must be disabled before torch import')
        import platform
        import torch
        if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not torch.backends.mps.is_available():
            raise RuntimeError('Native arm64 macOS MPS worker required; sandbox failure does not prove absent GPU')
        from .neural import R56Model, ModelConfig
        from dataclasses import asdict
        from .native_domains import pack_problem
        from .benchmarks import matched_baseline, controlled_graph, model_accounting
        from .learning import save_checkpoint
        torch.use_deterministic_algorithms(True)
        import sys
        runtime_base = Path(root).resolve()
        for name, module in tuple(sys.modules.items()):
            if name.startswith('mindscape.r56.') and getattr(module, '__file__', None):
                if not Path(module.__file__).resolve().is_relative_to(runtime_base):
                    raise PermissionError('Full repository module present in worker process')
        torch.manual_seed(seed)
        device = torch.device('mps')
        model = (matched_baseline('gru', 44962, seed) if condition == 'matched-gru' else R56Model()).to(device)
        rows = read_json(stage / f'train-{seed}.json')
        if len(rows) != 128:
            raise ValueError('Training-only exact 128 records required')
        if any(row.get('split') != 'train' or row.get('seed') != seed for row in rows):
            raise ValueError('Nontraining or wrong-seed data in handoff')
        graphs = [controlled_graph(pack_problem(row['problem'], device=device), condition) for row in rows]
        if any(p.device.type != 'mps' or p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('Model parameter device/dtype mismatch')
        if any(graph.node_features.device.type != 'mps' or graph.node_features.dtype != torch.float32
               or graph.action.device.type != 'mps' for graph in graphs):
            raise RuntimeError('Graph device/dtype mismatch')
        initial_dataset_loss = training_loss_summary(model, graphs, rows, torch, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        model.train()
        torch.mps.synchronize()
        fit_start = time.monotonic()
        with (output_directory / 'updates.jsonl').open('x', buffering=1) as log:
            for step in range(2000):
                index = step % 128
                output = model(graphs[index], memory=None, stateless=False)
                loss = domain_loss(output, rows[index], torch, device)
                if loss.device.type != 'mps' or not torch.isfinite(loss):
                    raise RuntimeError('Invalid loss or device')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if any(gradient.device.type != 'mps' for gradient in gradients):
                    raise RuntimeError('Gradient device mismatch')
                if not gradients or not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all():
                    raise RuntimeError('Nonfinite or absent gradients')
                optimizer.step(); updates += 1
                if not torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all():
                    raise RuntimeError('Nonfinite updated parameters')
                value = float(loss.detach().cpu()); losses.append(value)
                log.write(json.dumps({'update': updates, 'loss': value}) + '\n')
                if updates % 100 == 0:
                    log.flush()
        torch.mps.synchronize()
        elapsed = time.monotonic() - fit_start
        final_dataset_loss = training_loss_summary(model, graphs, rows, torch, device)
        metadata = {'initial_dataset_loss': initial_dataset_loss, 'final_dataset_loss': final_dataset_loss,
            'scope': SCOPE, 'package_sha256': identity, 'seed': seed, 'condition': condition,
            'optimizer_updates': updates, 'device': 'mps', 'torch_version': str(torch.__version__),
            'python_version': platform.python_version(), 'platform': platform.platform(),
            'first_online_preupdate_loss': losses[0], 'last_online_preupdate_loss': losses[-1], 'wall_seconds': elapsed,
            'seconds_per_update': elapsed / updates, 'model_accounting': model_accounting(model),
            'optimizer_steps': updates, 'forward_policy': manifest['state_policy'],
            'model_config': ({'kind': 'matched-gru', 'target_parameters': 44962} if condition == 'matched-gru' else asdict(ModelConfig())),
            'source_locks': manifest['source_locks'],
            'scientific_source_sha256': digest(manifest['source_locks']),
            'training_data_sha256': manifest['training_files_sha256'][str(seed)],
            'mps_current_allocated_bytes': torch.mps.current_allocated_memory(),
            'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory(),
            'scientifically_qualified': False, 'status': 'TRAINED_NOT_INDEPENDENTLY_EVALUATED',
            'selection': manifest['checkpoint_selection'], 'cpu_fallback': False}
        import resource
        metadata['process_lifetime_peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        metadata['memory_measurement_scope'] = 'macOS process-lifetime RSS peak; current MPS allocator/driver bytes, not a measured peak of GPU activations'
        checkpoint_hash = save_checkpoint(model, output_directory / 'checkpoint.json', metadata)
        from .learning import load_checkpoint
        reloaded = (matched_baseline('gru', 44962, seed) if condition == 'matched-gru' else R56Model()).to(device)
        load_checkpoint(reloaded, output_directory / 'checkpoint.json', checkpoint_hash)
        model.eval()
        with torch.no_grad():
            before = model(graphs[0], memory=None, stateless=False)
            after = reloaded(graphs[0], memory=None, stateless=False)
            if not all(torch.allclose(before[key], after[key], atol=1e-6, rtol=1e-5) for key in
                    ('math_digit_logits', 'math_carry_logits', 'coding_logits', 'physics_delta')):
                raise RuntimeError('Safe checkpoint reload prediction mismatch')
        metadata['reload_verified'] = True
        metadata['reload_tolerance'] = {'atol': 1e-6, 'rtol': 1e-5}
        write_new(output_directory / 'fit.json', {**metadata, 'checkpoint_sha256': checkpoint_hash})
        return {**metadata, 'checkpoint_sha256': checkpoint_hash}
    except BaseException as exc:
        write_new(output_directory / 'failure.json', {'package_sha256': identity,
            'optimizer_updates': updates, 'error_type': type(exc).__name__,
            'error': str(exc),
            'wall_seconds': time.monotonic() - began, 'status': 'FAILED'})
        raise


def main():
    parser = argparse.ArgumentParser(description='Compact MPS proposal/externally gated worker')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--root', required=True); prep.add_argument('--output', required=True)
    staging = sub.add_parser('stage')
    staging.add_argument('--package', required=True); staging.add_argument('--output', required=True)
    staging.add_argument('--seed', type=int, required=True)
    worker = sub.add_parser('fit')
    for name in ('root', 'stage', 'artifacts', 'trust', 'output', 'condition'):
        worker.add_argument('--' + name, required=True)
    worker.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.output)
    elif args.command == 'stage':
        result = stage_training(args.package, args.output, args.seed)
    else:
        result = run_fit(args.root, args.stage, read_json(args.artifacts), args.trust, args.output, args.seed, args.condition)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
