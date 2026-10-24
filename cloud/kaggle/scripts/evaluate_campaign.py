"""Two-phase, one-shot private Kaggle CPU evaluation controller. No local inference."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

DOWNLOAD_PATTERN = r'^evaluation-output/|^evaluator-package/(evaluation-manifest|isolation-evidence)\.json$'


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def event(name, **values):
    print(json.dumps({'event': name, **values}, sort_keys=True), flush=True)


def cli(arguments):
    executable = os.environ.get('KAGGLE_BIN') or shutil.which('kaggle')
    if not executable:
        raise RuntimeError('Set KAGGLE_BIN to the authenticated Kaggle CLI')
    result = subprocess.run([executable, *arguments], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Kaggle CLI operation failed; inspect privately (credential-bearing stderr is not printed)')
    return result.stdout


def reserve_push(package, phase, expected_version):
    package = Path(package)
    with (package / f'{phase}-submission-reservation.json').open('x') as stream:
        json.dump({'status': 'ATTEMPTED_ONCE', 'expected_version': expected_version,
                   'kernel_sha256': sha(package / 'kernel.py'), 'reserved_unix': time.time()}, stream)
    response = cli(['kernels', 'push', '-p', str(package)])
    (package / f'{phase}-submission-response.txt').write_text(response)
    match = re.search(r'Kernel version (\d+) successfully pushed', response)
    if not match or int(match.group(1)) != expected_version:
        raise RuntimeError('Provider push did not confirm expected unique kernel version; no resubmission permitted')
    event('evaluator_submitted', phase=phase, version=expected_version)


def wait_complete(kernel, record, timeout=7200):
    started = time.monotonic(); previous = None
    while True:
        state = cli(['kernels', 'status', kernel]).strip()
        Path(record).write_text(state + '\n')
        if state != previous:
            event('evaluator_status', kernel=kernel, state=state); previous = state
        if 'KernelWorkerStatus.COMPLETE' in state:
            return
        if any(x in state for x in ('KernelWorkerStatus.ERROR', 'KernelWorkerStatus.CANCEL', 'KernelWorkerStatus.FAIL')):
            raise RuntimeError('Evaluator cloud execution failed; preserve run and do not resubmit')
        if time.monotonic() - started > timeout:
            raise TimeoutError('Evaluator remains reserved; inspect cloud state without resubmitting')
        time.sleep(20)


def validate_metadata(actual, expected):
    for field, wanted in expected.items():
        observed = actual.get(field)
        if field == 'code_file':
            if not isinstance(observed, str) or Path(observed).name != observed or not observed.endswith('.py'):
                raise ValueError('Provider code_file must name Python source')
        elif field in ('kernel_sources', 'dataset_sources', 'competition_sources', 'model_sources'):
            if not isinstance(observed, list) or len(observed) != len(set(observed)):
                raise ValueError('Provider attachment references differ: ' + field)
            if field == 'dataset_sources':
                # Provider pulls omit /1. Only that exact normalization is accepted;
                # frozen dataset bytes, rather than metadata, prove input identity.
                canonical = {item[:-2] if item.endswith('/1') and item.count('/') == 2 else item for item in wanted}
                normalized = {item[:-2] if item in wanted and item.endswith('/1') and item.count('/') == 2 else item for item in observed}
                if len(normalized) != len(observed) or normalized != canonical:
                    raise ValueError('Provider attachment references differ: ' + field)
            elif set(observed) != set(wanted):
                raise ValueError('Provider attachment references differ: ' + field)
        elif observed != wanted:
            raise ValueError('Provider metadata differs: ' + field)
    if actual.get('is_private') is not True or any(actual.get(field) is not False for field in ('enable_gpu', 'enable_tpu', 'enable_internet')):
        raise ValueError('Private offline CPU provider configuration required')


def pull_verified(kernel, destination, expected_source_sha256, expected_metadata):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    # This provider rejects versioned source pulls; unique ID plus exact bytes pins identity.
    cli(['kernels', 'pull', kernel, '-p', str(destination), '-m'])
    metadata_path = destination / 'kernel-metadata.json'; metadata = read(metadata_path)
    validate_metadata(metadata, expected_metadata)
    source_path = destination / metadata['code_file']
    if not source_path.is_file() or sha(source_path) != expected_source_sha256:
        raise ValueError('Provider-pulled launcher bytes differ from submitted launcher')
    source = source_path.read_text()
    if any(token in source for token in ('UserSecretsClient', 'kaggle_secrets', '.get_secret(')):
        raise PermissionError('Provider launcher requests user secrets')
    return {'metadata': metadata, 'remote_metadata_sha256': sha(metadata_path),
            'launcher_sha256': sha(source_path), 'source_file': str(source_path),
            'metadata_file': str(metadata_path)}


def verify_download(root, blueprint, expected_evidence_digest):
    root = Path(root); result_root = root / 'evaluation-output'
    manifest_path = root / 'evaluator-package/evaluation-manifest.json'
    evidence_path = root / 'evaluator-package/isolation-evidence.json'
    receipt_path = result_root / 'evaluation_receipt.json'
    receipt, manifest, evidence = read(receipt_path), read(manifest_path), read(evidence_path)
    evidence_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if evidence_digest != expected_evidence_digest:
        raise ValueError('Downloaded runtime controller evidence differs')
    if receipt.get('status') != 'COMPLETED_CLOUD_EVALUATION' or receipt.get('device') != 'cpu' or receipt.get('evaluation_optimizer_updates') != 0:
        raise ValueError('Completed CPU evaluation with zero optimizer updates required')
    if receipt.get('manifest_sha256') != sha(manifest_path) or receipt.get('isolation_evidence_sha256') != sha(evidence_path):
        raise ValueError('Runtime manifest/evidence hash mismatch')
    if manifest.get('isolation_evidence', {}).get('sha256') != sha(evidence_path):
        raise ValueError('Runtime evidence pin mismatch')
    expected_sources = {name: value for name, value in blueprint['core_source_files'].items() if name.startswith('src/')}
    if {entry['path']: entry['sha256'] for entry in manifest.get('source_files', [])} != expected_sources:
        raise ValueError('Runtime evaluator model source pins differ')
    if manifest.get('data_files') != blueprint['data_files'] or manifest.get('evaluations') != blueprint['evaluations']:
        raise ValueError('Runtime frozen evaluation protocol/data identities differ')
    expected = {(e['seed'], e['condition']): e['sha256'] for e in blueprint['checkpoints']}
    pins = {(e['seed'], e['condition']): e['sha256'] for e in manifest.get('checkpoints', [])}
    if len(manifest.get('checkpoints', [])) != 9 or pins != expected:
        raise ValueError('Runtime checkpoint inventory differs from finalized exports')
    runs = [run for group in receipt.get('results', []) for run in group.get('runs', [])]
    if len(runs) != 9 or {(run['seed'], run['condition']) for run in runs} != set(expected):
        raise ValueError('Exactly nine distinct evaluated model artifacts required')
    artifacts = {}
    for run in runs:
        name = run['artifact']; path = result_root / name
        if Path(name).name != name or not path.is_file() or sha(path) != run['sha256']:
            raise ValueError('Evaluation result artifact hash mismatch')
        result = read(path)
        key = run['seed'], run['condition']
        if result.get('checkpoint_sha256') != expected[key] or (result.get('seed'), result.get('condition')) != key:
            raise ValueError('Evaluation result model/checkpoint identity differs')
        artifacts[str(path.relative_to(root))] = sha(path)
    audit = receipt.get('runtime_isolation_audit') or {}
    protected = audit.get('checkpoint_write_open_denied_paths', [])
    if len(protected) != 9 or len(set(protected)) != 9 or not all(p.startswith('/kaggle/input/') for p in protected):
        raise ValueError('Nine actual provider read-only checkpoint checks required')
    for flag in ('provider_batch_execution', 'known_credential_environment_absent', 'known_credential_files_absent'):
        if audit.get(flag) is not True:
            raise ValueError('Runtime isolation audit incomplete: ' + flag)
    if audit.get('controller_evidence_sha256') != expected_evidence_digest:
        raise ValueError('Runtime audit evidence digest differs')
    if {str(e['path']) for e in manifest['checkpoints']} != set(protected):
        raise ValueError('Runtime read-only checks do not cover exact evaluated checkpoints')
    artifacts.update({str(path.relative_to(root)): sha(path) for path in (receipt_path, manifest_path, evidence_path)})
    return {'status': 'VERIFIED_CLOUD_EVALUATION', 'evaluated_fits': 9,
            'evaluation_optimizer_updates': 0, 'artifact_sha256': artifacts,
            'runtime_isolation_audit': audit, 'historical_evaluation_reuse': True,
            'protection_scope': 'frozen-runtime-isolation', 'controller_trusted': True}


def download_verified(kernel, output, blueprint, evidence_digest):
    output = Path(output)
    for attempt in range(1, 4):
        target = output / f'download-attempt-{attempt}'; target.mkdir(exist_ok=False)
        try:
            cli(['kernels', 'output', kernel + '/2', '-p', str(target), '--file-pattern', DOWNLOAD_PATTERN])
            verified = verify_download(target, blueprint, evidence_digest)
        except (RuntimeError, ValueError, KeyError, FileNotFoundError) as error:
            event('evaluation_download_verification_failed', attempt=attempt, error_type=type(error).__name__)
            if attempt == 3:
                raise
            time.sleep(20)
            continue
        for directory in ('evaluation-output', 'evaluator-package'):
            shutil.copytree(target / directory, output / directory)
        return verified


def execute(repo, training_payload, data_root, checkpoint_index, trainer_evidence, owner, slug, package, output, checkpoint_dataset_receipt=None):
    repo, package, output = Path(repo).resolve(), Path(package).resolve(), Path(output).resolve()
    spec = importlib.util.spec_from_file_location('evaluation_builder', repo / 'cloud/kaggle/scripts/prepare_evaluation.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    output.mkdir(parents=True, exist_ok=False)
    blueprint = builder.stage(repo, training_payload, data_root, checkpoint_index, owner, slug, package, checkpoint_dataset_receipt)
    trainers = read(trainer_evidence); trainers = trainers['trainers'] if isinstance(trainers, dict) else trainers
    expected_trainers = {entry['kernel_id'] for entry in blueprint['checkpoints']}
    if len(trainers) != 9 or {role['kernel_id'] for role in trainers} != expected_trainers:
        raise ValueError('Controller trainer evidence must cover nine finalized exports')
    for role in trainers:
        if role.get('control_plane_source_verified') is not True or role.get('credential_access_code_absent') is not True or str(role.get('version')) != '1':
            raise ValueError('Authenticated reviewed immutable trainer evidence required')
    with zipfile.ZipFile(package / 'evaluator-core.zip') as archive:
        for name in archive.namelist():
            if name.endswith('.py') and any(token in archive.read(name).decode() for token in ('UserSecretsClient', 'kaggle_secrets', '.get_secret(')):
                raise PermissionError('Evaluator core requests user secrets')
    expected_metadata = read(package / 'kernel-metadata.json'); kernel = blueprint['kernel_id']
    stage_sha = sha(package / 'kernel.py')
    reserve_push(package, 'stage', 1)
    wait_complete(kernel, output / 'stage-status.txt')
    staged = pull_verified(kernel, output / 'provider-stage-v1', stage_sha, expected_metadata)
    worker_path = 'cloud/kaggle/evaluation/worker.py'
    evaluator = {'kernel_id': kernel, 'version': '2', 'provider_identity_stage_version': '1',
        'source_sha256': blueprint['core_source_files'][worker_path],
        'source_path': worker_path, 'source_identity_scope': 'evaluator-core-file',
        'core_source_files': blueprint['core_source_files'], 'remote_metadata_sha256': staged['remote_metadata_sha256'],
        'metadata': staged['metadata'], 'credential_access_code_absent': True,
        'control_plane_source_verified': True, 'control_plane_source_verified_scope': 'CLI-pulled stage-v1 launcher and immutable core; final-v2 full launcher separately verified after run',
        'stage_launcher_sha256': stage_sha}
    evidence = {'provider': 'kaggle', 'protection_scope': 'frozen-runtime-isolation', 'controller_trusted': True,
        'checkpoint_export_hash_verified': True,
        'checkpoint_input_version': [entry['kernel_id'] + '/1' for entry in blueprint['checkpoints']],
        'trainer': trainers[0], 'trainers': trainers, 'evaluator': evaluator,
        'limitations': ['Historically reused final cases', 'Trusted account owner can change future runs',
                       'Provider source pull by unique slug; exact source hashes pin one-shot trainer identities',
                       'Provider metadata may omit dataset version; exact checkpoint and receipt hashes establish data identity rather than metadata version proof']}
    if blueprint.get('checkpoint_transport'):
        evidence['checkpoint_transport'] = blueprint['checkpoint_transport']
        evidence['checkpoint_input_version'].insert(0, blueprint['checkpoint_transport']['reference'])
    evidence_path = output / 'controller-evidence.json'; write(evidence_path, evidence)
    finalized = builder.finalize(package, evidence_path)
    reserve_push(package, 'final', 2)
    wait_complete(kernel, output / 'final-status.txt')
    final_provider = pull_verified(kernel, output / 'provider-final-v2', finalized['kernel_sha256'], expected_metadata)
    verification = download_verified(kernel, output, blueprint, finalized['evidence_sha256'])
    verification.update(kernel_id=kernel, provider_version='2', provider_launcher_sha256=final_provider['launcher_sha256'],
                        provider_remote_metadata_sha256=final_provider['remote_metadata_sha256'],
                        stage_launcher_sha256=stage_sha, core_zip_sha256=blueprint['core_zip_sha256'],
                        controller_evidence_digest=finalized['evidence_sha256'])
    if blueprint.get('checkpoint_transport'):
        verification['dataset_version_metadata_limitation'] = 'Kaggle source pulls may omit dataset version; exact checkpoint and receipt content hashes establish input identity, not metadata version proof.'
    write(output / 'externalpostrun-controller-verification.json', verification)
    event('independent_cloud_evaluation_verified', kernel=kernel, evaluated_fits=9, evaluation_optimizer_updates=0)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'training-payload', 'data-root', 'checkpoint-index', 'trainer-evidence', 'owner', 'slug', 'package', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checkpoint-dataset-receipt')
    execute(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
