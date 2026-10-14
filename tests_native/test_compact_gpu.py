"""No optimizer creation/steps: proposal and fail-closed worker contracts."""
import json
import os
from pathlib import Path
import pytest
from mindscape.r56 import compact_gpu as campaign


def test_compact_proposal_separate_and_exact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = campaign.prepare(root, tmp_path / 'proposal')
    assert manifest['fits'] == 9 and manifest['total_updates'] == 18000
    assert manifest['status'] == 'PROPOSED_UNAPPROVED'
    assert manifest['total_records_per_seed'] == 128
    assert len(manifest['conditions']) == 3
    assert manifest['state_policy'] == 'stateless-false-reset-memory-per-independent-record'
    train = json.loads((tmp_path / 'proposal/trainer/train-17.json').read_text())
    assert len(train) == 128 and {r['split'] for r in train} == {'train'}
    assert manifest['exposure_summary']['17']['total_exposures'] == 128
    assert not any('eval' in p.name for p in (tmp_path / 'proposal/trainer').iterdir())
    for role in ('authorization', 'qualification', 'readiness', 'deployment'):
        assert not campaign.read_json(tmp_path / f'proposal/{role}.UNISSUED.json')['issued']


def test_worker_rejects_before_torch_import_or_output(tmp_path, monkeypatch):
    import builtins
    root = Path(__file__).resolve().parents[1]
    manifest = campaign.prepare(root, tmp_path / 'proposal')
    stage = tmp_path / 'stage'; stage.mkdir()
    runtime = stage / 'runtime'; runtime.mkdir()
    manifest['runtime_files'] = {}
    body = dict(manifest); body.pop('package_sha256')
    manifest['package_sha256'] = campaign.digest(body)
    campaign.write_new(stage / 'manifest.json', manifest)
    original = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == 'torch':
            raise AssertionError('Torch before external gate')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    trust = tmp_path / 'self-owned-trust.json'
    trust.write_text(json.dumps({'scope': campaign.SCOPE, 'issuers': {'self': {}}}))
    with pytest.raises(PermissionError, match='staged minimal source'):
        campaign.run_fit(runtime, stage, {}, trust, tmp_path / 'fit', 17, 'graph-concept-on')
    assert not (tmp_path / 'fit').exists()
    with pytest.raises(PermissionError, match='operator-owned'):
        campaign._trusted_config(trust)


def test_identity_and_training_path_refuse_without_heldout_read(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    manifest = campaign.prepare(root, tmp_path / 'proposal')
    manifest['total_updates'] = 1
    with pytest.raises(PermissionError, match='identity'):
        campaign.validate_gate(root, manifest, {}, 'unused', tmp_path, 17, 'graph-concept-on')
    # Proposal cannot be used as valid signed artifact.
    with pytest.raises(PermissionError):
        campaign._signed({'scope': campaign.SCOPE, 'role': 'authorization',
            'issued': False, 'issuer': '', 'package_sha256': 'a' * 64},
            'authorization', 'a' * 64, {'issuers': {}})


def test_candidate_support_loss_and_no_optimizer(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No optimizer')))
    output = {'coding_logits': torch.tensor([0., 1., 999.], requires_grad=True)}
    row = {'problem': {'domain': 'coding', 'candidates': ['a', 'b']}, 'labels': {'coding_index': 1}}
    loss = campaign.domain_loss(output, row, torch, torch.device('cpu'))
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(torch.tensor([[0., 1.]]), torch.tensor([1])))
    loss.backward()
    assert output['coding_logits'].grad[2] == 0
    row['labels']['coding_index'] = 2
    with pytest.raises(ValueError, match='support'):
        campaign.domain_loss(output, row, torch, torch.device('cpu'))


def test_new_evidence_refuses_overwrite(tmp_path):
    path = tmp_path / 'failure.json'
    campaign.write_new(path, {'completed': 0})
    with pytest.raises(FileExistsError):
        campaign.write_new(path, {'completed': 2000})
    assert campaign.read_json(path) == {'completed': 0}


def test_training_summary_fixed_means_no_updates(monkeypatch):
    torch = pytest.importorskip('torch')
    class Model:
        training = True
        def eval(self):
            self.training = False
        def train(self, previous=True):
            self.training = previous
        def __call__(self, graph, memory=None, stateless=True):
            assert memory is None and stateless is False
            return {'physics_delta': torch.tensor(graph)}
    rows = [{'problem': {'domain': 'physics', 'mass': 2., 'force': [2., 4.], 'dt': 1.},
             'labels': {'physics_delta': [1., 2.]}}] * 2
    summary = campaign.training_loss_summary(Model(), [[1., 2.], [2., 3.]], rows, torch, torch.device('cpu'))
    assert summary['mixed_loss_mean'] == .5
    assert summary['per_domain_mean']['physics'] == .5
    assert summary['per_domain_count']['physics'] == 2


def test_exact_stage_and_update_exposures(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = campaign.prepare(root, tmp_path / 'proposal')
    receipt = campaign.stage_training(tmp_path / 'proposal', tmp_path / 'stage', 17)
    assert receipt['optimizer_updates'] == 0
    assert set(p.name for p in (tmp_path / 'stage').iterdir()) == {'runtime', 'train-17.json', 'manifest.json'}
    assert campaign.runtime_locks(tmp_path / 'stage/runtime') == manifest['runtime_files']
    assert sum(manifest['update_counts_by_record_index'].values()) == 2000
    assert len(manifest['workload_rows']) == 9
    assert sum(manifest['domain_update_counts_per_fit']['17'].values()) == 2000
    with pytest.raises(FileExistsError):
        campaign.stage_training(tmp_path / 'proposal', tmp_path / 'stage', 17)
