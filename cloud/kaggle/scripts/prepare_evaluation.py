"""Prepare a private CPU evaluator in two phases without running inference locally.

Stage runs a CPU launcher that verifies its frozen core without scoring any cases.
Submit it with normal Kaggle push; CLI 2.2.4 has no --no-run flag. Finalize accepts controller
records from authenticated provider pulls; the operator must verify those records.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import zipfile

SEEDS = (17, 29, 43)
CONDITIONS = ('graph-concept-on', 'graph-concept-off', 'matched-gru')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def isolation_module(repo):
    spec = importlib.util.spec_from_file_location('staged_isolation', Path(repo) / 'cloud/kaggle/evaluation/isolation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_checkpoint_index(entries):
    if isinstance(entries, dict):
        entries = entries['checkpoints']
    expected = {(s, c) for s in SEEDS for c in CONDITIONS}
    if len(entries) != 9 or {(e['seed'], e['condition']) for e in entries} != expected:
        raise ValueError('Exactly nine finalized approved seed/condition checkpoints required')
    if len({e['kernel_id'] for e in entries}) != 9:
        raise ValueError('Nine distinct one-shot trainer kernels required')
    checked = []
    for entry in entries:
        if str(entry['version']) != '1' or not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]+', entry['kernel_id']):
            raise ValueError('Unique trainer kernel version 1 required')
        path, receipt_path = Path(entry['path']), Path(entry['receipt_path'])
        if sha(path) != entry['sha256']:
            raise ValueError('Finalized local checkpoint hash mismatch')
        receipt = load(receipt_path)
        checkpoint = load(path)
        if (receipt.get('status') != 'COMPLETED_EXPLORATORY'
            or receipt.get('optimizer_updates') != 2000
            or receipt.get('optimizer_steps') != 2000
            or receipt.get('step_post_hook_count') != 2000
            or receipt.get('checkpoint_sha256') != entry['sha256']
            or (receipt.get('seed'), receipt.get('condition')) != (entry['seed'], entry['condition'])
            or checkpoint.get('metadata', {}).get('optimizer_steps') != 2000):
            raise ValueError('Finalized checkpoint/receipt identity or optimizer count mismatch')
        checked.append({k: entry[k] for k in ('seed', 'condition', 'sha256', 'kernel_id', 'version')} |
                       {'version': '1', 'receipt_sha256': sha(receipt_path), 'version_label': 'R57c'})
    return sorted(checked, key=lambda e: (e['seed'], e['condition']))


def validate_checkpoint_dataset(receipt_path, checked_entries):
    """Validate a controller's private, provider-downloaded version-1 transport."""
    receipt = load(receipt_path)
    dataset_id = receipt.get('dataset_id', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]+', dataset_id) or type(receipt.get('version')) is not int or receipt['version'] != 1:
        raise ValueError('Private checkpoint dataset must pin owner/slug version 1')
    if receipt.get('is_private') is not True or receipt.get('download_verified') is not True:
        raise ValueError('Private provider-downloaded checkpoint dataset required')
    if not re.fullmatch('[0-9a-f]{64}', receipt.get('provider_metadata_sha256', '')):
        raise ValueError('Provider dataset metadata hash required')
    expected = {(e['seed'], e['condition']): e for e in checked_entries}
    supplied = receipt.get('checkpoints', [])
    if len(checked_entries) != 9 or len(expected) != 9 or len(supplied) != 9 or {(e['seed'], e['condition']) for e in supplied} != set(expected):
        raise ValueError('Checkpoint dataset receipt must cover exactly nine finalized fits')
    expected_files, augmented = {}, []
    for entry in supplied:
        original = expected[entry['seed'], entry['condition']]
        if any(str(entry.get(key)) != str(original[key]) for key in ('sha256', 'kernel_id', 'version')):
            raise ValueError('Dataset checkpoint identity differs from finalized export')
    for entry in checked_entries:
        stem = f"{entry['seed']}-{entry['condition']}"
        input_filename, receipt_filename = stem + '-final.json', stem + '-result.json'
        expected_files[input_filename] = entry['sha256']
        expected_files[receipt_filename] = entry['receipt_sha256']
        augmented.append({**entry, 'input_filename': input_filename, 'receipt_filename': receipt_filename})
    if receipt.get('files_sha256') != expected_files:
        raise ValueError('Checkpoint dataset must contain exactly the 18 pinned checkpoint/receipt files')
    return {'kind': 'private-checkpoint-dataset', 'dataset_id': dataset_id, 'version': 1,
            'reference': dataset_id + '/1', 'is_private': True, 'download_verified': True,
            'provider_metadata_sha256': receipt['provider_metadata_sha256'],
            'receipt_sha256': sha(receipt_path), 'files_sha256': expected_files,
            'checkpoints': augmented}


def stage(repo, training_payload, data_root, checkpoint_index, owner, slug, output, checkpoint_dataset_receipt=None):
    repo, payload, data_root, out = map(Path, (repo, training_payload, data_root, output))
    checkpoint_entries = validate_checkpoint_index(load(checkpoint_index))
    transport = validate_checkpoint_dataset(checkpoint_dataset_receipt, checkpoint_entries) if checkpoint_dataset_receipt else None
    if transport:
        checkpoint_entries = transport['checkpoints']
    training_manifest = load(payload / 'manifest.json')
    if not str(training_manifest.get('source_commit', '')).startswith('d54ff8f'):
        raise ValueError('Frozen R5.7c trainer source identity required')
    frozen_data = load(data_root / 'manifest.json')
    contents = {}
    for relative, expected in training_manifest['source_files_sha256'].items():
        source = payload / 'source' / relative
        if source.is_symlink() or not source.resolve().is_relative_to((payload / 'source').resolve()) or sha(source) != expected:
            raise ValueError('Frozen source mismatch')
        contents[relative] = source.read_bytes()
    for filename in ('__init__.py', 'worker.py', 'isolation.py'):
        relative = 'cloud/kaggle/evaluation/' + filename
        contents[relative] = (repo / relative).read_bytes()
    data_files, jobs = [], []
    for seed in SEEDS:
        for old, new in ((f'train/{seed}.json', f'data/train/{seed}.json'),
                         (f'evaluator-only/final-{seed}.json', f'data/final/{seed}.json')):
            source = data_root / old
            if sha(source) != frozen_data['files'][old]:
                raise ValueError('Historical frozen dataset changed: ' + old)
            contents[new] = source.read_bytes()
            data_files.append({'path': new, 'sha256': sha(source)})
        jobs.append({'seed': seed, 'training_path': f'data/train/{seed}.json',
                     'evaluation_path': f'data/final/{seed}.json',
                     'distribution_id': 'r57c-data-v1-historical-final',
                     'historical_evaluation_reuse': True})
    source_files = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items() if name.endswith('.py')}
    out.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(contents.items()):
            archive.writestr(name, value)
    (out / 'evaluator-core.zip').write_bytes(buffer.getvalue())
    kernel_id = f'{owner}/{slug}'
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]+', kernel_id):
        raise ValueError('Invalid evaluator kernel identity')
    metadata = {'id': kernel_id, 'title': slug, 'code_file': 'kernel.py', 'language': 'python',
                'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
                'enable_tpu': False, 'enable_internet': False,
                'dataset_sources': [transport['reference']] if transport else [],
                'competition_sources': [], 'model_sources': [],
                'kernel_sources': [] if transport else sorted(e['kernel_id'] for e in checkpoint_entries)}
    blueprint = {'schema_version': 1, 'kernel_id': kernel_id, 'core_zip_sha256': sha(out / 'evaluator-core.zip'),
                 'core_source_files': source_files, 'data_files': data_files, 'evaluations': jobs,
                 'checkpoints': checkpoint_entries, 'source_commit': training_manifest['source_commit'],
                 'source_manifest_sha256': sha(payload / 'manifest.json'),
                 'historical_evaluation_reuse': True,
                 'protection_scope': 'frozen-runtime-isolation', 'controller_trusted': True}
    if transport:
        blueprint['checkpoint_transport'] = transport
    write(out / 'kernel-metadata.json', metadata)
    write(out / 'evaluation-blueprint.json', blueprint)
    (out / 'kernel.py').write_text(launcher(out, None))
    write(out / 'stage-receipt.json', {'kernel_sha256': sha(out / 'kernel.py'),
          'metadata_sha256': sha(out / 'kernel-metadata.json'), 'blueprint_sha256': sha(out / 'evaluation-blueprint.json'),
          'core_zip_sha256': blueprint['core_zip_sha256'], 'status': 'STAGED_NO_SCORING'})
    return blueprint


def launcher(package, evidence):
    package = Path(package); blueprint = load(package / 'evaluation-blueprint.json')
    encoded = base64.b64encode((package / 'evaluator-core.zip').read_bytes()).decode('ascii')
    preamble = f'''# Private CPU-only evaluator; generated by trusted account-owner controller.
import base64, hashlib, io, json, os, pathlib, sys, zipfile
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') != 'Batch':
    raise RuntimeError('Separate Kaggle Batch execution required')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
core_bytes = base64.b64decode({encoded!r})
if hashlib.sha256(core_bytes).hexdigest() != {blueprint['core_zip_sha256']!r}:
    raise RuntimeError('Evaluator core package hash mismatch')
core = pathlib.Path('/kaggle/working/evaluator-package')
core.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(core_bytes)) as archive:
    for member in archive.namelist():
        if not (core / member).resolve().is_relative_to(core.resolve()):
            raise RuntimeError('Unsafe core archive path')
    archive.extractall(core)
sys.path.insert(0, str(core))
sys.path.insert(0, str(core / 'src'))
os.chdir(core)
blueprint = json.loads({json.dumps(blueprint, sort_keys=True)!r})
for relative, expected in blueprint['core_source_files'].items():
    if hashlib.sha256((core / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Extracted core source changed')
'''
    if evidence is None:
        return preamble + "print(json.dumps({'status': 'STAGED_NO_SCORING', 'core_zip_sha256': blueprint['core_zip_sha256'], 'core_source_files': blueprint['core_source_files']}, sort_keys=True))\n"
    evidence_text = json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False)
    evidence_sha = hashlib.sha256(evidence_text.encode()).hexdigest()
    return preamble + f'''
from cloud.kaggle.evaluation.isolation import kaggle_verifier
from cloud.kaggle.evaluation.worker import run
input_root = pathlib.Path('/kaggle/input')
checkpoints = []
for entry in blueprint['checkpoints']:
    matches = [p for p in input_root.rglob(entry.get('input_filename', 'final.json')) if hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']]
    if len(matches) != 1:
        raise RuntimeError('Each finalized checkpoint must match exactly one mounted immutable input')
    checkpoint = matches[0]
    receipt_path = checkpoint.parent / entry.get('receipt_filename', 'result.json')
    if not receipt_path.is_file() or hashlib.sha256(receipt_path.read_bytes()).hexdigest() != entry['receipt_sha256']:
        raise RuntimeError('Mounted final receipt hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('optimizer_updates') != 2000 or receipt.get('checkpoint_sha256') != entry['sha256'] or (receipt.get('seed'), receipt.get('condition')) != (entry['seed'], entry['condition']):
        raise RuntimeError('Mounted receipt identity/update mismatch')
    checkpoints.append({{'path': str(checkpoint), 'sha256': entry['sha256'], 'seed': entry['seed'], 'condition': entry['condition'], 'version': 'R57c'}})
evidence = json.loads({evidence_text!r})
evidence_path = core / 'isolation-evidence.json'
evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2))
manifest = {{'schema': 'mindscape-independent-evaluation-v1', 'source_files': [{{'path': name, 'sha256': value}} for name, value in blueprint['core_source_files'].items() if name.startswith('src/')],
            'data_files': blueprint['data_files'], 'evaluations': blueprint['evaluations'], 'checkpoints': checkpoints,
            'isolation_evidence': {{'path': 'isolation-evidence.json', 'sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest()}}}}
manifest_path = core / 'evaluation-manifest.json'
manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
verifier = kaggle_verifier({evidence_sha!r}, [e['path'] for e in checkpoints], core_root=core)
result = run(manifest_path, '/kaggle/working/evaluation-output', trusted_control_plane_verifier=verifier)
print(json.dumps({{'status': result['status'], 'evaluation_optimizer_updates': result['evaluation_optimizer_updates'], 'historical_evaluation_reuse': True}}, sort_keys=True))
'''


def finalize(package, evidence):
    package, evidence_path = Path(package), Path(evidence)
    stage_receipt = load(package / 'stage-receipt.json')
    for filename, field in (('kernel-metadata.json', 'metadata_sha256'), ('evaluation-blueprint.json', 'blueprint_sha256'), ('evaluator-core.zip', 'core_zip_sha256')):
        if sha(package / filename) != stage_receipt[field]:
            raise ValueError('Staged metadata or evaluator core changed')
    blueprint, evidence = load(package / 'evaluation-blueprint.json'), load(evidence_path)
    # Load exact staged adapter, never an updated local implementation.
    with zipfile.ZipFile(package / 'evaluator-core.zip') as archive:
        namespace = {'__name__': 'staged_isolation'}
        exec(archive.read('cloud/kaggle/evaluation/isolation.py'), namespace)
    namespace['validate_controller_evidence'](evidence)
    evaluator = evidence['evaluator']
    if evaluator['kernel_id'] != blueprint['kernel_id'] or evaluator['core_source_files'] != blueprint['core_source_files']:
        raise ValueError('Provider-evidenced evaluator core identity differs')
    # Kaggle pull renames code_file from kernel.py to the provider slug.py.
    # Filename equality is not source identity; the controller independently
    # checks the exact downloaded source bytes and records their digest.
    remote_code_file = evaluator['metadata'].get('code_file', '')
    if not isinstance(remote_code_file, str) or Path(remote_code_file).name != remote_code_file or not remote_code_file.endswith('.py'):
        raise ValueError('Pulled evaluator code_file must name a Python source file')
    for field, expected_value in load(package / 'kernel-metadata.json').items():
        if field == 'code_file':
            continue
        observed_value = evaluator['metadata'].get(field)
        if field in ('kernel_sources', 'dataset_sources', 'competition_sources', 'model_sources'):
            if not isinstance(observed_value, list) or len(observed_value) != len(set(observed_value)):
                raise ValueError('Pulled evaluator source references differ: ' + field)
            if field == 'dataset_sources':
                # Kaggle source pulls omit the version-1 suffix. Input content hashes
                # remain authoritative; this is not provider metadata version proof.
                canonical = {item[:-2] if item.endswith('/1') and item.count('/') == 2 else item for item in expected_value}
                normalized = {item[:-2] if item in expected_value and item.endswith('/1') and item.count('/') == 2 else item for item in observed_value}
                if len(normalized) != len(observed_value) or normalized != canonical:
                    raise ValueError('Pulled evaluator source references differ: ' + field)
            elif set(observed_value) != set(expected_value):
                raise ValueError('Pulled evaluator source references differ: ' + field)
        elif observed_value != expected_value:
            raise ValueError('Pulled evaluator metadata differs from staged metadata: ' + field)
    if blueprint.get('checkpoint_transport'):
        if evidence.get('checkpoint_transport') != blueprint['checkpoint_transport']:
            raise ValueError('Controller checkpoint transport differs from staged dataset')
        if blueprint['checkpoint_transport']['reference'] not in evidence.get('checkpoint_input_version', []):
            raise ValueError('Controller evidence omits version-pinned checkpoint dataset')
    trainers = evidence.get('trainers', [])
    expected = {e['kernel_id'] for e in blueprint['checkpoints']}
    if len(trainers) != 9 or {e['kernel_id'] for e in trainers} != expected:
        raise ValueError('Provider evidence must cover all nine trainers')
    for trainer in trainers:
        if str(trainer['version']) != '1':
            raise ValueError('Trainer kernel must be immutable unique version 1')
        namespace['validate_controller_evidence']({**evidence, 'trainer': trainer})
    with (package / 'finalize-reservation.json').open('x') as stream:
        json.dump({'status': 'FINALIZED_ONCE', 'evidence_sha256': namespace['evidence_digest'](evidence)}, stream)
    write(package / 'controller-evidence.json', evidence)
    (package / 'kernel.py').write_text(launcher(package, evidence))
    result = {'status': 'FINALIZED_READY_FOR_PRIVATE_CPU_SUBMISSION', 'kernel_sha256': sha(package / 'kernel.py'),
              'evidence_sha256': namespace['evidence_digest'](evidence), 'core_zip_sha256': blueprint['core_zip_sha256'],
              'note': 'Controller must pull final provider source/metadata after execution and verify full launcher SHA.'}
    if blueprint.get('checkpoint_transport'):
        result['dataset_version_metadata_limitation'] = 'Kaggle source pulls may omit dataset version; exact checkpoint and receipt content hashes establish input identity, not metadata version proof.'
    write(package / 'finalized-receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('stage')
    for name in ('repo', 'training-payload', 'data-root', 'checkpoint-index', 'owner', 'slug', 'output'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--checkpoint-dataset-receipt')
    final = sub.add_parser('finalize'); final.add_argument('--package', required=True); final.add_argument('--evidence', required=True)
    values = vars(parser.parse_args()); action = values.pop('action')
    print(json.dumps(stage(**values) if action == 'stage' else finalize(**values), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
