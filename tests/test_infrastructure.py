"""Orchestration tests only: no local optimizer or fitting operations."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import pytest

BASE = Path(__file__).resolve().parents[1] / 'cloud/kaggle/scripts'

def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

wf = load('workflow')


def test_probe_refuses_local_before_torch_import(tmp_path, monkeypatch):
    probe = load('cuda_probe')
    monkeypatch.delenv('KAGGLE_KERNEL_RUN_TYPE', raising=False)
    monkeypatch.delenv('KAGGLE_KERNEL_INTEGRATIONS', raising=False)
    with pytest.raises(RuntimeError, match='Real Kaggle'):
        probe.main(tmp_path / 'must-not-exist')
    assert not (tmp_path / 'must-not-exist').exists()


def test_prepare_probe_private_no_internet(tmp_path):
    folder = wf.prepare('probe', 'owner', 'probe-unique-123', tmp_path / 'probe')
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    assert meta['is_private'] is True
    assert meta['enable_internet'] is False
    assert meta['enable_gpu'] is True
    assert meta['dataset_sources'] == []
    assert wf.digest(folder / 'kernel.py') == json.loads((folder / 'package-receipt.json').read_text())['kernel_sha256']


def test_publication_rejected(tmp_path, monkeypatch):
    folder = wf.prepare('probe', 'owner', 'probe-unique-123', tmp_path / 'probe')
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    meta['is_private'] = False
    wf.write_json(folder / 'kernel-metadata.json', meta)
    monkeypatch.setattr(wf, 'invoke', lambda _: pytest.fail('Must not invoke network'))
    with pytest.raises(ValueError, match='Private offline'):
        wf.push(folder)


def test_edited_package_rejected(tmp_path, monkeypatch):
    folder = wf.prepare('probe', 'owner', 'probe-unique-123', tmp_path / 'probe')
    (folder / 'kernel.py').write_text('changed')
    monkeypatch.setattr(wf, 'invoke', lambda _: pytest.fail('Must not invoke network'))
    with pytest.raises(ValueError, match='changed'):
        wf.push(folder)


def test_submission_reserved_even_on_failure(tmp_path, monkeypatch):
    folder = wf.prepare('probe', 'owner', 'probe-unique-123', tmp_path / 'probe')
    account = tmp_path / 'account.json'
    wf.write_json(account, {'status': 'verified', 'checked_unix': time.time(), 'quota': [{'resource': 'GPU', 'remaining': '1.00h'}]})
    calls = []
    def failed(args):
        calls.append(args)
        raise RuntimeError('Network failed')
    monkeypatch.setattr(wf, 'invoke', failed)
    with pytest.raises(RuntimeError): wf.push(folder, account_receipt=account)
    assert (folder / 'submission-reservation.json').exists()
    with pytest.raises(FileExistsError): wf.push(folder, account_receipt=account)
    assert len(calls) == 1


def test_quota_matches_installed_cli_format():
    assert wf.gpu_quota_remaining({'quota': [{'resource': 'GPU', 'remaining': '2.50h'}]}) == 9000
    assert wf.gpu_quota_remaining({'quota': [{'resource': 'GPU', 'remaining': '0.00h'}]}) == 0
    with pytest.raises(ValueError): wf.gpu_quota_remaining({'quota': []})


def test_probe_verification_requires_artifacts(tmp_path):
    receipt = tmp_path / 'cuda_probe_receipt.json'
    wf.write_json(receipt, {'status': 'passed', 'cuda_available': True, 'optimizer_updates': 5,
        'weights_changed': True, 'reload_succeeded': True, 'device': 'cuda:0',
        'kaggle_run_type': 'Batch', 'artifacts': {}})
    with pytest.raises(ValueError, match='no verified artifacts'): wf.verify_probe(receipt)


def test_zip_payload_rejects_credentials(tmp_path):
    (tmp_path / 'kaggle.json').write_text('not-real-secret')
    with pytest.raises(ValueError, match='credential'): wf.zip_payload(tmp_path)


def test_submission_requires_real_quota_snapshot(tmp_path, monkeypatch):
    folder = wf.prepare('probe', 'owner', 'probe-unique-123', tmp_path / 'probe')
    monkeypatch.setattr(wf, 'invoke', lambda _: pytest.fail('Must not invoke network'))
    with pytest.raises(ValueError, match='Fresh account'): wf.push(folder)


def test_unapproved_accelerator_refused(tmp_path, monkeypatch):
    folder = wf.prepare('probe', 'owner', 'probe-unique-123', tmp_path / 'probe')
    monkeypatch.setattr(wf, 'invoke', lambda _: pytest.fail('Must not invoke network'))
    with pytest.raises(ValueError, match='approved'): wf.push(folder, accelerator='NvidiaH100')


def test_campaign_ledger_refuses_same_fit_in_new_package(tmp_path, monkeypatch):
    account = tmp_path / 'account.json'
    wf.write_json(account, {'status': 'verified', 'checked_unix': time.time(), 'quota': [{'resource': 'GPU', 'remaining': '1.00h'}]})
    packages = []
    for n in (1, 2):
        folder = wf.prepare('probe', 'owner', f'fit-unique-{n}', tmp_path / f'fit{n}')
        identity = {'fit_identity': 'f' * 64}
        wf.write_json(folder / 'payload-identity.json', identity)
        receipt = json.loads((folder / 'package-receipt.json').read_text())
        receipt.update(kind='training', payload_identity_sha256=wf.digest(folder / 'payload-identity.json'))
        wf.write_json(folder / 'package-receipt.json', receipt)
        packages.append(folder)
    monkeypatch.setattr(wf, 'verify_probe', lambda _: {'status': 'passed'})
    calls = []
    monkeypatch.setattr(wf, 'invoke', lambda args: calls.append(args) or 'submitted')
    wf.push(packages[0], probe_receipt='cloud-receipt', account_receipt=account, ledger=tmp_path / 'ledger')
    with pytest.raises(FileExistsError):
        wf.push(packages[1], probe_receipt='cloud-receipt', account_receipt=account, ledger=tmp_path / 'ledger')
    assert len(calls) == 1


def test_download_rejects_http_error_saved_as_receipt(tmp_path, monkeypatch):
    target = tmp_path / 'download'
    monkeypatch.setattr(sys, 'argv', ['workflow.py', 'download', 'owner/probe', '--output', str(target)])
    def downloaded(_):
        (target / 'cuda_probe_receipt.json').write_text('<html><title>502 Server Error</title></html>')
        return 'CLI reported download success'
    monkeypatch.setattr(wf, 'invoke', downloaded)
    with pytest.raises(RuntimeError, match='not the expected artifact'):
        wf.main()
    index = json.loads((target / 'download-index.json').read_text())
    assert index['verified_transport'] is False
    assert index['invalid_artifacts'] == ['cuda_probe_receipt.json']
    assert (target / 'cuda_probe_receipt.json').is_file()
