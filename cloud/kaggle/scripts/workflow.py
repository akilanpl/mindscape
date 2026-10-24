"""Private, one-shot Kaggle orchestration; never executes neural code locally."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
IDENTIFIER = re.compile(r'^[a-zA-Z0-9_-]+/[a-z0-9-]+$')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def executable():
    value = os.environ.get('KAGGLE_BIN') or shutil.which('kaggle')
    if not value:
        raise RuntimeError('Kaggle CLI missing; set KAGGLE_BIN to the installed executable')
    return value


def invoke(args):
    # Never echo CLI stderr: authentication backends may include credential material.
    result = subprocess.run([executable(), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Kaggle command failed (authentication, account activation, network, or service error); inspect privately with Kaggle CLI')
    return result.stdout


def verify_account(output):
    listing = invoke(['kernels', 'list', '--mine', '--format', 'json'])
    quota_raw = invoke(['quota', '--format', 'json'])
    try:
        quota = json.loads(quota_raw)
    except json.JSONDecodeError:
        raise RuntimeError('Kaggle quota was not machine-readable; training remains blocked')
    # Listing itself is deliberately not persisted: it may contain unrelated private work.
    result = {'schema_version': 1, 'status': 'verified', 'checked_unix': time.time(),
              'kernel_listing_authenticated': bool(listing.strip()), 'quota': quota,
              'note': 'Quota is a snapshot, not a guarantee of GPU allocation.'}
    write_json(output, result)
    return result


def make_metadata(owner, slug, kind):
    kernel_id = f'{owner}/{slug}'
    if not IDENTIFIER.fullmatch(kernel_id):
        raise ValueError('Expected valid Kaggle owner and lowercase hyphenated slug')
    return {'id': kernel_id, 'title': slug, 'code_file': 'kernel.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [],
            'model_sources': [], 'keywords': []}


def zip_payload(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(payload.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlinks forbidden in cloud payload')
            if path.is_file():
                relative = path.relative_to(payload)
                if path.name.startswith('.') or path.suffix in ('.pem', '.key') or path.name == 'kaggle.json':
                    raise ValueError('Potential credential file in payload')
                if '__pycache__' not in relative.parts:
                    archive.writestr(str(relative), path.read_bytes())
    return buffer.getvalue()


def prepare(kind, owner, slug, output, payload=None, seed=None, condition=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = make_metadata(owner, slug, kind)
    if kind == 'probe':
        script = (Path(__file__).parent / 'cuda_probe.py').read_text()
    else:
        if seed not in (17, 29, 43) or condition not in ('graph-concept-on', 'graph-concept-off', 'matched-gru'):
            raise ValueError('Training requires one approved seed and condition')
        payload = Path(payload)
        manifest = json.loads((payload / 'manifest.json').read_text())
        if not (payload / 'source/cloud/kaggle/training/worker.py').is_file():
            raise ValueError('Missing training worker in frozen payload')
        # Refuse evaluation data and unrelated files; only the reproducibility builder creates this layout.
        if set(p.name for p in payload.iterdir()) - {'source', 'train', 'manifest.json'}:
            raise ValueError('Unexpected files in trainer package')
        if set(p.name for p in (payload / 'train').iterdir()) != {'17.json', '29.json', '43.json'}:
            raise ValueError('Only frozen training splits may be embedded')
        worker_path = Path(__file__).parents[1] / 'training/worker.py'
        spec = importlib.util.spec_from_file_location('cloud_worker_validation', worker_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        worker.validate_package(payload, seed, condition)
        expected = {'manifest.json'} | {f'train/{n}.json' for n in (17, 29, 43)}
        for field in ('source_files_sha256', 'infrastructure_files_sha256'):
            for relative, expected_sha in manifest.get(field, {}).items():
                target = payload / 'source' / relative
                if not target.resolve().is_relative_to((payload / 'source').resolve()) or target.is_symlink() or digest(target) != expected_sha:
                    raise ValueError('Payload source/infrastructure identity changed')
                expected.add('source/' + relative)
        actual = {str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if actual != expected:
            raise ValueError('Payload contains unmanifested files or omits declared files')
        data = zip_payload(payload)
        encoded = base64.b64encode(data).decode('ascii')
        data_sha = hashlib.sha256(data).hexdigest()
        script = f'''# Generated private immutable single-fit Kaggle entrypoint.
import base64, hashlib, io, os, pathlib, runpy, sys, zipfile
if not pathlib.Path('/kaggle/working').is_dir() or not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    raise RuntimeError('Kaggle cloud execution required')
raw = base64.b64decode({encoded!r})
assert hashlib.sha256(raw).hexdigest() == {data_sha!r}
package = pathlib.Path('/kaggle/working/input-package')
package.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        target = (package / name).resolve()
        if not target.is_relative_to(package.resolve()):
            raise RuntimeError('Unsafe archive path')
    archive.extractall(package)
sys.path.insert(0, str(package / 'source/src'))
sys.argv = ['worker.py', '--package', str(package), '--output', '/kaggle/working/fit-output', '--seed', {str(seed)!r}, '--condition', {condition!r}]
runpy.run_path(str(package / 'source/cloud/kaggle/training/worker.py'), run_name='__main__')
'''
        write_json(out / 'payload-identity.json', {'zip_sha256': data_sha, 'manifest_sha256': digest(payload / 'manifest.json'), 'seed': seed, 'condition': condition,
            'fit_identity': hashlib.sha256(json.dumps({'source': manifest['source_files_sha256'], 'training': manifest['training_files_sha256'], 'seed': seed, 'condition': condition}, sort_keys=True).encode()).hexdigest()})
    (out / 'kernel.py').write_text(script)
    write_json(out / 'kernel-metadata.json', metadata)
    write_json(out / 'package-receipt.json', {'schema_version': 1, 'kind': kind,
               'kernel_id': metadata['id'], 'created_unix': time.time(),
               'kernel_sha256': digest(out / 'kernel.py'),
               'metadata_sha256': digest(out / 'kernel-metadata.json'),
               'payload_identity_sha256': digest(out / 'payload-identity.json') if kind == 'training' else None})
    return out


def verify_probe(path):
    receipt_path = Path(path)
    receipt = json.loads(receipt_path.read_text())
    if not (receipt.get('status') == 'passed' and receipt.get('cuda_available') is True
            and receipt.get('optimizer_updates') == 5 and receipt.get('weights_changed') is True
            and receipt.get('reload_succeeded') is True and receipt.get('device') == 'cuda:0'
            and receipt.get('kaggle_run_type')):
        raise ValueError('Verified genuine Kaggle five-update CUDA probe required')
    for filename, expected in receipt.get('artifacts', {}).items():
        artifact = receipt_path.parent / filename
        if artifact.resolve().parent != receipt_path.parent.resolve() or digest(artifact) != expected:
            raise ValueError('Probe artifact hash mismatch')
    if not receipt.get('artifacts'):
        raise ValueError('Probe has no verified artifacts')
    return receipt


def gpu_quota_remaining(account):
    # CLI 2.2.4 quota_view_cli emits resource='GPU', remaining='N.NNh'.
    rows = account.get('quota', [])
    if isinstance(rows, dict):
        rows = rows.get('data', rows.get('rows', []))
    for row in rows if isinstance(rows, list) else []:
        normalized = {re.sub('[^a-z]', '', str(k).lower()): v for k, v in row.items()}
        if str(normalized.get('resource', normalized.get('accelerator', normalized.get('type', '')))).upper() == 'GPU':
            for field in ('remainingseconds', 'remaining'):
                try:
                    value = str(normalized[field])
                    return float(value[:-1]) * 3600 if value.endswith('h') else float(value)
                except (KeyError, TypeError, ValueError):
                    pass
    raise ValueError('Cannot determine account GPU quota; inspect account quota before training')


def push(package, probe_receipt=None, account_receipt=None, accelerator='NvidiaTeslaT4', ledger=None):
    folder = Path(package)
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    receipt = json.loads((folder / 'package-receipt.json').read_text())
    if receipt.get('kind') not in ('probe', 'training') or metadata.get('id') != receipt.get('kernel_id'):
        raise ValueError('Invalid package identity')
    if metadata.get('is_private') is not True or metadata.get('enable_internet') is not False:
        raise ValueError('Private offline kernel required')
    if digest(folder / 'kernel.py') != receipt['kernel_sha256'] or digest(folder / 'kernel-metadata.json') != receipt['metadata_sha256']:
        raise ValueError('Prepared package changed; rebuild using a fresh slug')
    if accelerator != 'NvidiaTeslaT4':
        raise ValueError('Only the approved NvidiaTeslaT4 accelerator is supported')
    if not account_receipt:
        raise ValueError('Fresh account/quota verification required')
    account = json.loads(Path(account_receipt).read_text())
    age = time.time() - account.get('checked_unix', 0)
    if account.get('status') != 'verified' or not 0 <= age <= 3600:
        raise ValueError('Fresh account/quota verification required')
    if gpu_quota_remaining(account) <= 0:
        raise ValueError('Account GPU quota exhausted')
    if receipt['kind'] == 'training':
        verify_probe(probe_receipt)
        if not ledger:
            raise ValueError('Training requires a stable campaign --ledger directory')
    if receipt['kind'] == 'training':
        if digest(folder / 'payload-identity.json') != receipt.get('payload_identity_sha256'):
            raise ValueError('Payload identity changed')
        identity = json.loads((folder / 'payload-identity.json').read_text())
        if not re.fullmatch('[0-9a-f]{64}', identity.get('fit_identity', '')):
            raise ValueError('Invalid fit identity')
        ledger = Path(ledger)
        ledger.mkdir(parents=True, exist_ok=True)
        with (ledger / (identity['fit_identity'] + '.json')).open('x') as stream:
            json.dump({'kernel_id': metadata['id'], 'reserved_unix': time.time(), 'status': 'attempted-once', **identity}, stream, indent=2)
    # Reserve BEFORE the network call. Any failure requires manual remote inspection; never auto retry.
    with (folder / 'submission-reservation.json').open('x') as stream:
        json.dump({'kernel_id': metadata['id'], 'reserved_unix': time.time(),
                   'status': 'attempted-once', 'kind': receipt['kind']}, stream, indent=2)
    command = ['kernels', 'push', '-p', str(folder)]
    if accelerator:
        command += ['--accelerator', accelerator]
    output = invoke(command)
    (folder / 'submission-response.txt').write_text(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify-account'); verify.add_argument('--output', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--kind', choices=('probe', 'training'), required=True)
    for name in ('owner', 'slug', 'output'): prep.add_argument('--' + name, required=True)
    prep.add_argument('--payload'); prep.add_argument('--seed', type=int); prep.add_argument('--condition')
    submit = commands.add_parser('push'); submit.add_argument('--package', required=True)
    submit.add_argument('--probe-receipt'); submit.add_argument('--account-receipt'); submit.add_argument('--accelerator', default='NvidiaTeslaT4'); submit.add_argument('--ledger')
    status = commands.add_parser('status'); status.add_argument('kernel')
    download = commands.add_parser('download'); download.add_argument('kernel'); download.add_argument('--output', required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'verify-account': print(json.dumps(verify_account(**args), indent=2))
    elif command == 'prepare': print(prepare(**args))
    elif command == 'push': print(push(**args))
    else:
        if not IDENTIFIER.fullmatch(args['kernel']): raise ValueError('Invalid kernel ID')
        if command == 'status': print(invoke(['kernels', 'status', args['kernel']]))
        else:
            target = Path(args['output']); target.mkdir(parents=True, exist_ok=False)
            print(invoke(['kernels', 'output', args['kernel'], '-p', str(target)]))
            invalid = []
            for artifact in sorted(target.rglob('*')):
                if not artifact.is_file():
                    continue
                prefix = artifact.read_bytes()[:512].lstrip().lower()
                if artifact.suffix in ('.json', '.pt', '.jsonl') and (prefix.startswith(b'<html') or prefix.startswith(b'<!doctype')):
                    invalid.append(str(artifact.relative_to(target)))
                elif artifact.suffix == '.json':
                    try:
                        json.loads(artifact.read_text())
                    except (ValueError, UnicodeError):
                        invalid.append(str(artifact.relative_to(target)))
            write_json(target / 'download-index.json', {'verified_transport': not invalid,
                'invalid_artifacts': invalid, 'files': {str(p.relative_to(target)): digest(p)
                for p in sorted(target.rglob('*')) if p.is_file() and p.name != 'download-index.json'}})
            if invalid:
                raise RuntimeError('Downloaded response is not the expected artifact; files preserved; retry download to a fresh directory, never rerun fitting')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, FileExistsError, TypeError) as error:
        raise SystemExit(str(error))
