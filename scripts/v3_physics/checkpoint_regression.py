"""Serialization-only inspection of genuine native MPS continuation checkpoints.

Loading tensor bytes on CPU here performs no neural forward/backward execution.
"""
import argparse
import json
from pathlib import Path

import torch

from mindscape.v3.physics_world.storage import atomic_json, file_sha


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert type(left) is type(right) and left == right


def plain(value):
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            plain(item)
    elif isinstance(value, list):
        for item in value:
            plain(item)
    else:
        assert type(value) in (str, int, float, bool, type(None)), type(value)


def inspect(root, output):
    root = Path(root)
    states = {}
    for pointer in root.rglob('checkpoint-index.json'):
        index = json.loads(pointer.read_text())
        assert Path(index['file']).name == index['file']
        path = pointer.parent / index['file']
        assert file_sha(path) == index['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        plain(saved['identity'])
        assert saved['identity'] == index['identity']
        assert saved['step'] == index['step']
        assert saved['optimizer']['state']
        states[pointer.parent] = saved
    comparisons = []
    for family in ('motion', 'interaction'):
        parent = root / 'resume-integration' / family
        resumed = states[parent / 'resumed']
        uninterrupted = states[parent / 'uninterrupted']
        for field in ('state', 'optimizer', 'step', 'identity', 'memory_keys',
                      'memory_residuals', 'memory_active'):
            equal(resumed[field], uninterrupted[field])
        assert resumed['step'] == 12
        comparisons.append({'family': family, 'parameters_optimizer_provenance_exact': True,
                            'step': resumed['step']})
    receipt = {'passed': True, 'weights_only': True, 'checkpoints_loaded': len(states),
               'native_continuation_comparisons': comparisons, 'neural_cpu_execution': False}
    atomic_json(output, receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inspect(args.pilot, args.output)
