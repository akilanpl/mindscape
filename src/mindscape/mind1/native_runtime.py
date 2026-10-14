"""Native-only inference boundary. Untrained demonstrations require explicit opt-in."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path

STATUS = 'R5.6 EXPERIMENTAL — SCIENTIFIC GENERALIZATION UNVERIFIED'


class NativeMindscapeRuntime:
    def __init__(self, runner):
        self.runner = runner

    def capability(self):
        path = os.getenv('MIND1_NATIVE_CHECKPOINT', '')
        return {'engine': 'Mindscape R5.6', 'torch_installed': importlib.util.find_spec('torch') is not None,
                'checkpoint_configured': bool(path), 'trained_checkpoint_qualified': False,
                'status': STATUS, 'external_inference': False,
                'engineering_demo': 'Explicit request opt-in only; random weights are not learned intelligence',
                'language_interpretation': 'deterministic bounded grammar; learned interpreter unqualified',
                'registry': [
                    {'family': 'decimal addition / local carry and borrow', 'native': 'experimental', 'validated_learning': False},
                    {'family': 'Python AST rename candidate selection', 'native': 'experimental', 'validated_learning': False},
                    {'family': '2D constant force / impulse velocity update', 'native': 'experimental', 'validated_learning': False},
                    {'family': 'symbolic calculus, numerical models, WASI Python', 'native': False, 'support': 'independently attributed tools'},
                    {'family': 'unrestricted free-form advanced reasoning', 'native': 'unsupported', 'validated_learning': False}]}

    def run(self, problem, *, engineering_demo=False, cancel=None, timeout=15):
        if not engineering_demo and not os.getenv('MIND1_NATIVE_CHECKPOINT'):
            return {'available': False, 'status': 'unavailable', 'reason': 'No qualified native checkpoint exists. Enable the untrained engineering demo only to inspect model wiring.', 'scientific_status': STATUS}
        return self.runner('native_transition', {'problem': problem, 'engineering_demo': engineering_demo}, cancel=cancel, timeout=timeout)


def predict_native(problem, engineering_demo=False):
    """Runs only in the bounded trusted worker. No HTTP or pretrained-model code."""
    if type(engineering_demo) is not bool or not isinstance(problem, dict):
        raise ValueError('Typed problem and boolean engineering_demo required')
    import sys
    def deny_network(event, args):
        if event == 'socket.connect':
            raise PermissionError('Native inference has no network capability')
    sys.addaudithook(deny_network)
    import torch
    from mindscape.r56.neural import R56Model
    torch.set_num_threads(1)
    checkpoint = os.getenv('MIND1_NATIVE_CHECKPOINT', '')
    if checkpoint:
        # Local operator-specified checkpoint, never a user-supplied arbitrary path.
        from mindscape.r56.learning import load_checkpoint
        expected_hash = os.getenv('MIND1_NATIVE_CHECKPOINT_SHA256', '')
        if len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash):
            raise ValueError('A locally reviewed checkpoint SHA256 is required')
        model = R56Model()
        metadata = load_checkpoint(model, Path(checkpoint), expected_hash)
        checkpoint_id = expected_hash
        trained = bool(metadata.get('optimizer_steps', 0) > 0)
        if not engineering_demo:
            # A serialized trained flag is insufficient scientific qualification.
            return {'available': False, 'status': 'unqualified', 'checkpoint_sha256': checkpoint_id,
                    'reason': 'Checkpoint loading is implemented; independent scientific qualification is not issued. Use engineering demo to inspect predictions.', 'scientific_status': STATUS}
    else:
        if not engineering_demo:
            return {'available': False, 'status': 'unavailable', 'scientific_status': STATUS}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = R56Model()
        trained = False
        checkpoint_id = 'untrained-initialization-seed-17'
    model.cpu().eval()
    weights_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights_digest.update(name.encode()); weights_digest.update(tensor.detach().cpu().numpy().tobytes())
    receipt = model.predict(problem)
    return {'available': True, 'status': 'engineering_demo', **receipt,
            'scientific_status': STATUS,
            'checkpoint': {'identity': checkpoint_id, 'weights_sha256': weights_digest.hexdigest(),
                           'trained': trained, 'scientifically_qualified': False},
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'attribution': {'interpreter': 'deterministic grammar', 'transition': 'original R5.6 neural prediction',
                            'controller': 'supplied bounded schedule', 'verifier': 'independent deterministic reference'},
            'warning': 'ENGINEERING DEMONSTRATION — unqualified weights; no learned competence or generalization claim'}
