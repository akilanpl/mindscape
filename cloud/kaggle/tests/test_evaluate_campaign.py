"""Controller verification only, using explicit synthetic JSON fixtures; no cloud jobs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('evaluate_campaign', REPO / 'cloud/kaggle/scripts/evaluate_campaign.py')
controller = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(controller)


def fake_download(tmp_path):
    result_root = tmp_path / 'evaluation-output'; result_root.mkdir()
    core = tmp_path / 'evaluator-package'; core.mkdir()
    blueprint = {'checkpoints': [], 'core_source_files': {'src/mindscape/__init__.py': '1' * 64},
                 'data_files': [], 'evaluations': [], 'test_fixture_only': True}
    runs = []; pins = []; protected = []
    for seed in (17, 29, 43):
        for condition in ('graph-concept-on', 'graph-concept-off', 'matched-gru'):
            checkpoint_sha = hashlib.sha256(f'{seed}-{condition}'.encode()).hexdigest()
            blueprint['checkpoints'].append({'seed': seed, 'condition': condition, 'sha256': checkpoint_sha})
            mount = f'/kaggle/input/{seed}-{condition}/fit-output/final.json'; protected.append(mount)
            pins.append({'path': mount, 'sha256': checkpoint_sha, 'seed': seed, 'condition': condition})
            artifact = f'{seed}-{condition}-R57c.json'
            controller.write(result_root / artifact, {'seed': seed, 'condition': condition, 'checkpoint_sha256': checkpoint_sha})
            runs.append({'seed': seed, 'condition': condition, 'artifact': artifact, 'sha256': controller.sha(result_root / artifact)})
    evidence = {'test_fixture_only': True}; controller.write(core / 'isolation-evidence.json', evidence)
    evidence_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = {'checkpoints': pins, 'data_files': [], 'evaluations': [],
                'source_files': [{'path': name, 'sha256': value} for name, value in blueprint['core_source_files'].items()],
                'isolation_evidence': {'sha256': controller.sha(core / 'isolation-evidence.json')}}
    controller.write(core / 'evaluation-manifest.json', manifest)
    receipt = {'status': 'COMPLETED_CLOUD_EVALUATION', 'device': 'cpu', 'evaluation_optimizer_updates': 0,
               'manifest_sha256': controller.sha(core / 'evaluation-manifest.json'),
               'isolation_evidence_sha256': controller.sha(core / 'isolation-evidence.json'),
               'results': [{'runs': runs}], 'runtime_isolation_audit': {
                   'checkpoint_write_open_denied_paths': protected, 'provider_batch_execution': True,
                   'known_credential_environment_absent': True, 'known_credential_files_absent': True,
                   'controller_evidence_sha256': evidence_digest}}
    controller.write(result_root / 'evaluation_receipt.json', receipt)
    return blueprint, evidence_digest


def test_completed_download_requires_nine_verified_artifacts(tmp_path):
    blueprint, evidence = fake_download(tmp_path)
    result = controller.verify_download(tmp_path, blueprint, evidence)
    assert result['evaluated_fits'] == 9 and result['evaluation_optimizer_updates'] == 0
    assert len(result['artifact_sha256']) == 12


def test_result_tampering_rejected(tmp_path):
    blueprint, evidence = fake_download(tmp_path)
    (tmp_path / 'evaluation-output/17-matched-gru-R57c.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        controller.verify_download(tmp_path, blueprint, evidence)


def test_missing_runtime_readonly_evidence_rejected(tmp_path):
    blueprint, evidence = fake_download(tmp_path)
    path = tmp_path / 'evaluation-output/evaluation_receipt.json'; receipt = controller.read(path)
    receipt['runtime_isolation_audit']['checkpoint_write_open_denied_paths'].pop()
    controller.write(path, receipt)
    with pytest.raises(ValueError, match='read-only checkpoint checks'):
        controller.verify_download(tmp_path, blueprint, evidence)


def test_provider_renamed_filename_and_ref_order_allowed_but_gpu_rejected():
    expected = {'id': 'owner/evaluator', 'code_file': 'kernel.py', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
        'kernel_sources': ['owner/a', 'owner/b']}
    actual = {**expected, 'code_file': 'evaluator.py', 'kernel_sources': ['owner/b', 'owner/a']}
    controller.validate_metadata(actual, expected)
    with pytest.raises(ValueError, match='metadata differs'):
        controller.validate_metadata({**actual, 'enable_gpu': True}, expected)
    with pytest.raises(ValueError, match='attachment references differ'):
        controller.validate_metadata({**actual, 'kernel_sources': ['owner/a', 'owner/unexpected']}, expected)


def test_failed_push_reserved_no_retry(tmp_path, monkeypatch):
    (tmp_path / 'kernel.py').write_text('# no code')
    calls = []
    def failed(args):
        calls.append(args); raise RuntimeError('transport failure')
    monkeypatch.setattr(controller, 'cli', failed)
    with pytest.raises(RuntimeError): controller.reserve_push(tmp_path, 'stage', 1)
    with pytest.raises(FileExistsError): controller.reserve_push(tmp_path, 'stage', 1)
    assert len(calls) == 1


def test_provider_dataset_version1_suffix_omission_is_content_pin_only():
    expected = {'id': 'owner/evaluator', 'code_file': 'kernel.py', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
        'dataset_sources': ['owner/checkpoints/1'], 'kernel_sources': []}
    actual = {**expected, 'dataset_sources': ['owner/checkpoints']}
    controller.validate_metadata(actual, expected)
    for refs in (['owner/checkpoints/2'], ['owner/other'], ['owner/checkpoints', 'owner/checkpoints/1']):
        with pytest.raises(ValueError, match='attachment references differ'):
            controller.validate_metadata({**actual, 'dataset_sources': refs}, expected)
