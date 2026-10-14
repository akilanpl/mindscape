"""Synthetic storage fixtures only; no optimizer or neural evaluation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'research/alpha/r54_checkpoint_guard.py'
SPEC = importlib.util.spec_from_file_location('r54_checkpoint_guard', SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def fixture():
    return {'model': {'first': torch.zeros(2, 3), 'second': torch.zeros(2)},
            'optimizer': {'param_groups': [{'params': [0, 1], 'lr': .003,
                'weight_decay': .0001, 'betas': (.9, .999), 'eps': 1e-8,
                'amsgrad': False, 'maximize': False}],
                'state': {i: {'step': torch.tensor(800.), 'exp_avg': torch.zeros(shape),
                             'exp_avg_sq': torch.zeros(shape)}
                          for i, shape in enumerate(((2, 3), (2,)))}}}


def test_synthetic_valid_storage_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No optimizer construction allowed')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    assert guard.validate_optimizer(fixture())['exact_updates'] == 800


@pytest.mark.parametrize('mutation', ['fractional', 'missing', 'duplicate', 'routing',
    'shape', 'float64', 'nan', 'negative', 'amsgrad', 'epsilon', 'step_dtype', 'extra'])
def test_adversarial_storage_rejected(mutation):
    payload = copy.deepcopy(fixture())
    state = payload['optimizer']['state'][0]
    group = payload['optimizer']['param_groups'][0]
    if mutation == 'fractional': state['step'] = torch.tensor(800.5)
    elif mutation == 'missing': del payload['optimizer']['state'][1]
    elif mutation == 'duplicate': group['params'] = [0, 0]
    elif mutation == 'routing': group['params'] = [1, 0]
    elif mutation == 'shape': state['exp_avg'] = torch.zeros(3, 2)
    elif mutation == 'float64': state['exp_avg'] = state['exp_avg'].double()
    elif mutation == 'nan': state['exp_avg'][0, 0] = float('nan')
    elif mutation == 'negative': state['exp_avg_sq'][0, 0] = -1
    elif mutation == 'amsgrad': group['amsgrad'] = True
    elif mutation == 'epsilon': group['eps'] = .1
    elif mutation == 'step_dtype': state['step'] = torch.tensor(800, dtype=torch.int64)
    elif mutation == 'extra': state['unexpected'] = torch.tensor(0.)
    with pytest.raises(ValueError):
        guard.validate_optimizer(payload)


def test_unrun_campaign_rejected_without_loading_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Tensor inspection before completion'))
    with pytest.raises(ValueError, match='Completed'):
        guard.audit_checkpoint_storage(tmp_path)


def synthetic_campaign(directory):
    entries = []
    for kind in ('markov', 'recurrent'):
        for seed in (11, 23, 47):
            identity = f'r54-{kind}-neural-seed-{seed}-u800'
            payload = fixture()
            payload['updates'] = 800
            path = directory / (identity + '.pt')
            torch.save(payload, path)
            entries.append({'model_id': identity, 'kind': kind, 'seed': seed,
                            'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    seal = {'outcomes_opened': False, 'checkpoints': entries}
    (directory / 'PREDICTIONS_SEALED.json').write_text(json.dumps(seal))
    (directory / 'COMPLETE.json').write_text('{}')
    return seal


def test_six_checkpoint_storage_roundtrip_synthetic(tmp_path):
    synthetic_campaign(tmp_path)
    result = guard.audit_checkpoint_storage(tmp_path)
    assert len(result['checks']) == 6
    assert result['neural_forward_calls'] == result['optimizer_calls'] == 0


@pytest.mark.parametrize('mutation', ['count', 'order', 'seal', 'hash', 'failed', 'symlink'])
def test_campaign_storage_boundaries(tmp_path, mutation):
    seal = synthetic_campaign(tmp_path)
    if mutation == 'count': seal['checkpoints'].pop()
    elif mutation == 'order': seal['checkpoints'].reverse()
    elif mutation == 'seal': seal['outcomes_opened'] = True
    elif mutation == 'hash': seal['checkpoints'][0]['sha256'] = 'substituted'
    elif mutation == 'failed': (tmp_path / 'FAILED.json').write_text('{}')
    elif mutation == 'symlink':
        first = tmp_path / seal['checkpoints'][0]['path']
        first.unlink()
        first.symlink_to(tmp_path / seal['checkpoints'][1]['path'])
    (tmp_path / 'PREDICTIONS_SEALED.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        guard.audit_checkpoint_storage(tmp_path)


@pytest.mark.parametrize('review', [
    {'mode': 'EXECUTION', 'scientific_optimizer_updates': 0, 'neural_ood_predictions': 0},
    {'mode': 'NONTRAINING_REVIEW_ONLY', 'scientific_optimizer_updates': 1, 'neural_ood_predictions': 0},
    {'mode': 'NONTRAINING_REVIEW_ONLY', 'scientific_optimizer_updates': 0, 'neural_ood_predictions': 1},
])
def test_readiness_refuses_scientific_boundary(tmp_path, monkeypatch, review):
    spec = importlib.util.spec_from_file_location('r54_prepare', SOURCE.with_name('r54_prepare.py'))
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    def review_process(command, **kwargs):
        assert '--execute-scientific-training' not in command
        assert '--gate-adoption' not in command and '--scientific-authorization' not in command
        return SimpleNamespace(stdout=json.dumps(review))
    monkeypatch.setattr(preparation.subprocess, 'run', review_process)
    with pytest.raises(ValueError, match='Read-only'):
        preparation.prepare(tmp_path)
