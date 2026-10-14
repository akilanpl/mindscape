"""Safe local CI equivalent; fails if any selected test constructs an optimizer."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'src'))
import torch
import pytest

def forbidden(*args,**kwargs):
    raise RuntimeError('Nontraining qualification forbids optimizer construction')

for name in ('Adam','AdamW','SGD','RMSprop','Adagrad','Adadelta','Rprop','ASGD','LBFGS','NAdam','RAdam','SparseAdam','Adamax'):
    if hasattr(torch.optim,name):setattr(torch.optim,name,forbidden)
if __name__=='__main__':
    raise SystemExit(pytest.main([str(root/'tests_native'),str(root/'tests_mind1'),str(root/'tests_r56'),'-q',*sys.argv[1:]]))
