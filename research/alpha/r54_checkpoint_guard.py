"""Supplemental storage-only audit; no model, optimizer, fit or prediction calls.

The frozen v1 verifier coerces optimizer counters with int(), which can hide a
fractional counter. This additive guard preserves that verifier byte-for-byte.
It authenticates exact AdamW routing and moments before scientific scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_optimizer(payload, expected_updates=800):
    import torch

    weights = list(payload['model'].values())
    optimizer = payload['optimizer']
    groups = optimizer['param_groups']
    require(len(groups) == 1, 'Exactly one frozen optimizer group required')
    group = groups[0]
    require(group['lr'] == .003 and group['weight_decay'] == .0001
            and tuple(group['betas']) == (.9, .999), 'Frozen optimizer policy changed')
    require(group.get('eps') == 1e-8 and group.get('amsgrad') is False
            and group.get('maximize') is False, 'Frozen AdamW defaults changed')
    ids = group['params']
    require(all(type(i) is int for i in ids) and len(ids) == len(weights)
            and len(set(ids)) == len(ids), 'Duplicate or incomplete parameter routing')
    require(set(optimizer['state']) == set(ids), 'Optimizer state routing mismatch')
    for index, (identity, weight) in enumerate(zip(ids, weights)):
        state = optimizer['state'][identity]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected optimizer state')
        step = state['step']
        require(isinstance(step, torch.Tensor) and step.numel() == 1
                and step.dtype == torch.float32 and torch.isfinite(step).all().item()
                and step.item() == expected_updates, 'Exact optimizer update count required')
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = state[name]
            require(isinstance(moment, torch.Tensor) and moment.dtype == torch.float32
                    and moment.shape == weight.shape and torch.isfinite(moment).all().item(),
                    f'Invalid {name} tensor for parameter {index}')
        require((state['exp_avg_sq'] >= 0).all().item(), 'Negative second optimizer moment')
    return {'parameters': len(weights), 'exact_updates': expected_updates,
            'routing': 'VERIFIED', 'moments': 'VERIFIED'}


def audit_checkpoint_storage(destination):
    import torch

    destination = Path(destination)
    require((destination / 'COMPLETE.json').is_file(), 'Completed campaign required')
    require(not (destination / 'FAILED.json').exists(), 'Failed campaign cannot qualify')
    seal = json.loads((destination / 'PREDICTIONS_SEALED.json').read_text())
    require(seal['outcomes_opened'] is False, 'Predictions must precede reference scoring')
    expected = [(kind, seed) for kind in ('markov', 'recurrent') for seed in (11, 23, 47)]
    require(len(seal['checkpoints']) == 6, 'Exactly six checkpoints required')
    checks = []
    for entry, (kind, seed) in zip(seal['checkpoints'], expected):
        identity = f'r54-{kind}-neural-seed-{seed}-u800'
        require((entry['model_id'], entry['kind'], entry['seed'], entry['path'])
                == (identity, kind, seed, identity + '.pt'), 'Frozen checkpoint attribution changed')
        path = destination / entry['path']
        require(not path.is_symlink() and path.is_file(), 'Regular checkpoint file required')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'Checkpoint hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(type(payload['updates']) is int and payload['updates'] == 800,
                'Exact checkpoint update attribution required')
        checks.append({'model_id': identity, **validate_optimizer(payload)})
    return {'status': 'VERIFIED_R54_SUPPLEMENTAL_CHECKPOINT_STORAGE', 'checks': checks,
            'neural_forward_calls': 0, 'optimizer_calls': 0,
            'limitation': 'Storage consistency is not proof of actual training or model quality.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_checkpoint_storage(args.campaign), indent=2, sort_keys=True))
