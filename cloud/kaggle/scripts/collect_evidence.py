"""Collect fresh authenticated provider evidence for nine finalized CUDA fits.

Read-only cloud operations; no fitting, submissions, source execution, or cached
provider claims. KAGGLE_BIN selects the authenticated CLI outside this repository.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cloud.kaggle.reproducibility import SEEDS, CONDITIONS, FROZEN_COMMIT, verify_export
from cloud.kaggle.scripts.evaluate_campaign import cli, event


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_prepared(package):
    package = Path(package); receipt = read(package / 'package-receipt.json')
    metadata = read(package / 'kernel-metadata.json'); identity = read(package / 'payload-identity.json')
    if receipt.get('kind') != 'training' or receipt.get('kernel_id') != metadata.get('id'):
        raise ValueError('Prepared training package identity differs')
    for filename, field in (('kernel.py', 'kernel_sha256'), ('kernel-metadata.json', 'metadata_sha256'),
                            ('payload-identity.json', 'payload_identity_sha256')):
        if sha(package / filename) != receipt.get(field):
            raise ValueError('Prepared package hash differs: ' + filename)
    response = (package / 'submission-response.txt').read_text()
    if re.findall(r'Kernel version (\d+) successfully pushed', response) != ['1']:
        raise ValueError('Exactly one confirmed unique version-1 submission required')
    source = (package / 'kernel.py').read_text()
    tree = ast.parse(source)
    encoded = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'base64'
            and node.func.attr == 'b64decode' and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            encoded.append(node.args[0].value)
    if len(encoded) != 1:
        raise ValueError('Exactly one statically readable frozen trainer payload required')
    raw = base64.b64decode(encoded[0], validate=True)
    if hashlib.sha256(raw).hexdigest() != identity.get('zip_sha256'):
        raise ValueError('Embedded trainer ZIP differs from prepared identity')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate payload archive entries')
        manifest_bytes = archive.read('manifest.json'); manifest = json.loads(manifest_bytes)
        if hashlib.sha256(manifest_bytes).hexdigest() != identity.get('manifest_sha256'):
            raise ValueError('Embedded manifest differs from prepared identity')
        if manifest.get('source_commit') != FROZEN_COMMIT or manifest.get('evaluation_data_included') is not False:
            raise ValueError('Frozen train-only R5.7c source required')
        if manifest.get('protocol', {}).get('optimizer_updates') != 2000:
            raise ValueError('Frozen 2,000-update training protocol required')
        expected = {'manifest.json'}
        for field in ('source_files_sha256', 'infrastructure_files_sha256'):
            for relative, wanted in manifest[field].items():
                if not relative.endswith('.py') or Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('Unexpected trainer source path')
                name = 'source/' + relative; expected.add(name)
                if name.startswith('source/cloud/kaggle/evaluation/') or hashlib.sha256(archive.read(name)).hexdigest() != wanted:
                    raise ValueError('Trainer source hash/isolation differs')
        for seed in SEEDS:
            name = f'train/{seed}.json'; expected.add(name); value = archive.read(name)
            if hashlib.sha256(value).hexdigest() != manifest['training_files_sha256'][str(seed)]:
                raise ValueError('Embedded frozen training data hash differs')
            rows = json.loads(value)
            if len(rows) != 128 or any(row.get('seed') != seed or row.get('split') != 'train' for row in rows):
                raise ValueError('Only exact 128-record training splits allowed')
        if set(archive.namelist()) != expected:
            raise ValueError('Trainer payload contains unmanifested or evaluation material')
        executed_sources = [source] + [archive.read(name).decode() for name in expected if name.endswith('.py')]
        if any(token in text for text in executed_sources for token in ('UserSecretsClient', 'kaggle_secrets', '.get_secret(')):
            raise PermissionError('Trainer sources request user secrets')
    return metadata, {**identity, 'validated_training_sha256': manifest['training_files_sha256'][str(identity['seed'])]}, receipt


def find_packages(outputs, owner):
    matches = {}
    for receipt_path in sorted(Path(outputs).rglob('package-receipt.json')):
        receipt = read(receipt_path)
        if receipt.get('kind') != 'training':
            continue
        package = receipt_path.parent
        metadata_path, identity_path = package / 'kernel-metadata.json', package / 'payload-identity.json'
        if not metadata_path.is_file() or not identity_path.is_file():
            continue
        metadata, identity = read(metadata_path), read(identity_path)
        if metadata.get('id', '').split('/', 1)[0] != owner:
            continue
        key = identity.get('seed'), identity.get('condition')
        if key not in {(s, c) for s in SEEDS for c in CONDITIONS}:
            continue
        if key in matches:
            raise ValueError('More than one prepared package for owner/seed/condition; review campaign identity')
        matches[key] = package
    if len(matches) != 9:
        raise ValueError('Exactly nine unique owner-matching prepared training packages required')
    return matches


def verify_remote(directory, kernel_id, expected_source_sha256):
    directory = Path(directory); metadata_path = directory / 'kernel-metadata.json'; metadata = read(metadata_path)
    if (metadata.get('id') != kernel_id or metadata.get('is_private') is not True
        or metadata.get('enable_gpu') is not True or metadata.get('enable_tpu') is not False
        or metadata.get('enable_internet') is not False):
        raise ValueError('Provider trainer privacy/offline/GPU configuration differs')
    if any(metadata.get(field) != [] for field in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources')):
        raise ValueError('Provider trainer has unexpected attached inputs')
    filename = metadata.get('code_file', '')
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.py'):
        raise ValueError('Provider trainer source filename is unsafe')
    source = directory / filename
    if sha(source) != expected_source_sha256:
        raise ValueError('Fresh provider trainer source differs from prepared bytes')
    return {'kernel_id': kernel_id, 'version': '1', 'source_sha256': sha(source),
            'remote_metadata_sha256': sha(metadata_path), 'metadata': metadata,
            'control_plane_source_verified': True, 'credential_access_code_absent': True,
            'payload_evaluation_records': 0, 'evaluation_inputs_attached': False}


def collect(outputs, owner, provider_dir):
    outputs = Path(outputs).resolve(); provider_dir = Path(provider_dir).resolve()
    packages = find_packages(outputs, owner)
    # Validate all nine local exports and submitted source identities before the first provider operation.
    validated = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            fit = outputs / f'verified-fits/{seed}-{condition}'
            result = verify_export(fit)
            if (result['seed'], result['condition']) != (seed, condition):
                raise ValueError('Canonical finalized output directory identity differs')
            package = packages[seed, condition]
            metadata, identity, receipt = validate_prepared(package)
            if (identity['seed'], identity['condition']) != (seed, condition):
                raise ValueError('Prepared fit identity differs')
            if result.get('manifest_sha256') != identity['manifest_sha256'] or result.get('training_sha256') != identity['validated_training_sha256']:
                raise ValueError('Export receipt does not bind the exact prepared manifest/training bytes')
            validated.append((seed, condition, fit, result, metadata, receipt))
    if len({metadata['id'] for _, _, _, _, metadata, _ in validated}) != 9:
        raise ValueError('Nine distinct unique trainer kernels required')
    provider_dir.mkdir(parents=True, exist_ok=False)
    entries, roles = [], []
    for seed, condition, fit, result, metadata, receipt in validated:
        remote = provider_dir / f'{seed}-{condition}'; remote.mkdir()
        # Version source URLs are unsupported by this provider; unique kernel plus exact submitted bytes pins source.
        cli(['kernels', 'pull', metadata['id'], '-p', str(remote), '-m'])
        role = verify_remote(remote, metadata['id'], receipt['kernel_sha256'])
        roles.append(role)
        entries.append({'seed': seed, 'condition': condition, 'path': str(fit / 'final.json'),
            'sha256': result['checkpoint_sha256'], 'kernel_id': metadata['id'], 'version': '1',
            'receipt_path': str(fit / 'result.json')})
        event('trainer_provider_source_verified', seed=seed, condition=condition, kernel=metadata['id'])
    evidence = outputs / 'evidence'; evidence.mkdir(parents=True, exist_ok=True)
    # Replace only these derived controller records after all authenticated pulls succeed.
    for name, value in (('checkpoint-index.json', entries), ('trainer-control-plane-evidence.json', roles)):
        temporary = evidence / (name + '.new')
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
        temporary.replace(evidence / name)
    result = {'verified_fits': 9, 'verified_optimizer_updates': 18000,
              'provider_dir': str(provider_dir), 'checkpoint_index_sha256': sha(evidence / 'checkpoint-index.json'),
              'trainer_evidence_sha256': sha(evidence / 'trainer-control-plane-evidence.json')}
    (provider_dir / 'collection-receipt.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    event('campaign_evidence_collected', **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('outputs', 'owner', 'provider-dir'):
        parser.add_argument('--' + name, required=True)
    collect(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
