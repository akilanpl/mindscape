"""Isolated-process guard regression and real native MPS regression (never CPU)."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def execute(code):
    return subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                          env={**os.environ, 'PYTORCH_ENABLE_MPS_FALLBACK': '0'},
                          capture_output=True, text=True)


def test_guard_preserves_classes_blocks_construction_and_step():
    result = execute('''import torch
from scripts.native.mps_preflight import block_optimizers
base=torch.optim.Optimizer
classes={n:v for n,v in vars(torch.optim).items() if isinstance(v,type) and issubclass(v,base)}
weight=torch.nn.Parameter(torch.tensor([2.]))
before=weight.detach().clone()
block_optimizers(torch)
for name,cls in classes.items():
 assert getattr(torch.optim,name) is cls and isinstance(cls,type)
 assert (cls | None) is not None
 try: cls([weight])
 except RuntimeError as e: assert 'forbidden' in str(e)
 else: raise AssertionError('Construction permitted')
 # Exercise the guarded step without constructing an optimizer.
 try: cls.step(object())
 except RuntimeError as e: assert 'forbidden' in str(e)
 else: raise AssertionError('Update permitted')
assert torch.equal(before,weight.detach())
from mindscape.r56.neural import R56Model
assert isinstance(R56Model(),torch.nn.Module)
''')
    assert result.returncode == 0, result.stderr


def test_native_mps_full_preflight_unchanged_weights():
    # No mocking availability or CPU substitution: run this test in native Terminal.
    result = execute('''import torch,sys
if not torch.backends.mps.is_available(): sys.exit(77)
from scripts.native.mps_preflight import run
r=run()
assert r['status']=='ENGINEERING_PASSED_NOT_SCIENTIFICALLY_QUALIFIED'
assert r['optimizer_updates']==0 and r['weights_unchanged'] is True
assert len(r['measurements'])==9
assert {m['control'] for m in r['measurements']}=={'graph-concept-on','graph-concept-off','matched-gru'}
for m in r['measurements']:
 assert m['device']=='mps' and m['loss_device'].startswith('mps')
 assert m['gradient_devices'] and all(d.startswith('mps') for d in m['gradient_devices'])
''')
    if result.returncode == 77:
        pytest.skip('Native MPS unavailable in this execution context; native Terminal receipt required')
    assert result.returncode == 0, result.stderr
