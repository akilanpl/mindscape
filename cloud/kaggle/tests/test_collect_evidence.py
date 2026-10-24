"""Hermetic evidence-collection checks with synthetic train-only package bytes."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile
import pytest

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('collect_evidence', REPO / 'cloud/kaggle/scripts/collect_evidence.py')
collector = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(collector)


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def package(tmp_path, extra=False):
    tmp_path.mkdir()
    source = b'# safe model source\n'
    contents = {'source/src/mindscape/__init__.py': source}
    hashes = {}
    for seed in collector.SEEDS:
        value = json.dumps([{'seed': seed, 'split': 'train'} for _ in range(128)]).encode()
        contents[f'train/{seed}.json'] = value
        hashes[str(seed)] = hashlib.sha256(value).hexdigest()
    manifest = {'source_commit': collector.FROZEN_COMMIT, 'evaluation_data_included': False,
                'protocol': {'optimizer_updates': 2000}, 'training_files_sha256': hashes,
                'source_files_sha256': {'src/mindscape/__init__.py': hashlib.sha256(source).hexdigest()},
                'infrastructure_files_sha256': {}}
    contents['manifest.json'] = json.dumps(manifest).encode()
    if extra:
        contents['evaluator-only/final-17.json'] = b'[]'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, value in contents.items(): archive.writestr(name, value)
    raw = buffer.getvalue()
    (tmp_path / 'kernel.py').write_text('import base64\nraw = base64.b64decode(' + repr(base64.b64encode(raw).decode()) + ')\n')
    write(tmp_path / 'kernel-metadata.json', {'id': 'owner/trainer'})
    write(tmp_path / 'payload-identity.json', {'seed': 17, 'condition': 'graph-concept-on',
        'zip_sha256': hashlib.sha256(raw).hexdigest(),
        'manifest_sha256': hashlib.sha256(contents['manifest.json']).hexdigest()})
    receipt = {'kind': 'training', 'kernel_id': 'owner/trainer',
        'kernel_sha256': collector.sha(tmp_path / 'kernel.py'),
        'metadata_sha256': collector.sha(tmp_path / 'kernel-metadata.json'),
        'payload_identity_sha256': collector.sha(tmp_path / 'payload-identity.json')}
    write(tmp_path / 'package-receipt.json', receipt)
    (tmp_path / 'submission-response.txt').write_text('Kernel version 1 successfully pushed. Please check progress at provider')
    return tmp_path


def test_static_embedded_train_only_inventory_verified(tmp_path):
    path = package(tmp_path / 'package')
    metadata, identity, receipt = collector.validate_prepared(path)
    assert metadata['id'] == 'owner/trainer' and identity['seed'] == 17


def test_evaluation_material_rejected_even_if_zip_identity_updated(tmp_path):
    path = package(tmp_path / 'package', extra=True)
    with pytest.raises(ValueError, match='unmanifested or evaluation material'):
        collector.validate_prepared(path)


def test_local_metadata_tamper_rejected(tmp_path):
    path = package(tmp_path / 'package')
    write(path / 'kernel-metadata.json', {'id': 'owner/public-or-different'})
    with pytest.raises(ValueError): collector.validate_prepared(path)


def test_version_two_training_package_rejected(tmp_path):
    path = package(tmp_path / 'package')
    (path / 'submission-response.txt').write_text('Kernel version 2 successfully pushed')
    with pytest.raises(ValueError, match='version-1 submission'): collector.validate_prepared(path)


def test_remote_provider_pin_and_input_acl_checks(tmp_path):
    source = tmp_path / 'renamed-trainer.py'; source.write_text('# actual pulled source')
    metadata = {'id': 'owner/trainer', 'is_private': True, 'enable_gpu': True,
        'enable_tpu': False, 'enable_internet': False, 'code_file': source.name,
        'dataset_sources': [], 'kernel_sources': [], 'model_sources': [], 'competition_sources': []}
    write(tmp_path / 'kernel-metadata.json', metadata)
    role = collector.verify_remote(tmp_path, 'owner/trainer', collector.sha(source))
    assert role['control_plane_source_verified'] is True
    metadata['kernel_sources'] = ['owner/evaluator']; write(tmp_path / 'kernel-metadata.json', metadata)
    with pytest.raises(ValueError, match='unexpected attached inputs'):
        collector.verify_remote(tmp_path, 'owner/trainer', collector.sha(source))


def test_fresh_pulled_source_must_match_exact_prepared_bytes(tmp_path):
    source = tmp_path / 'renamed-trainer.py'; source.write_text('# altered remote source')
    write(tmp_path / 'kernel-metadata.json', {'id': 'owner/trainer', 'is_private': True,
        'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'code_file': source.name,
        'dataset_sources': [], 'kernel_sources': [], 'model_sources': [], 'competition_sources': []})
    with pytest.raises(ValueError, match='differs from prepared bytes'):
        collector.verify_remote(tmp_path, 'owner/trainer', 'f' * 64)
