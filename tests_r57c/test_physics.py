import copy
import pytest
import torch
from mindscape.r57.model import make_model
from mindscape.r57c.physics_audit import diagnostic_grid,exact,predict,gate_audit,baselines


def test_new_diagnostic_axes_and_transparent_analytic_reference():
 grid=diagnostic_grid();assert len(grid)==12
 for p in grid:
  reversed={**p,'force':[-f for f in p['force']]}
  assert exact(reversed)==pytest.approx([-v for v in exact(p)])
  assert exact({**p,'mass':p['mass']*2})==pytest.approx([v/2 for v in exact(p)])


def test_readonly_gate_probe_and_permutation_invariance(monkeypatch):
 def forbidden(*a,**k):raise AssertionError('No optimizer allowed')
 monkeypatch.setattr(torch.optim,'Adam',forbidden)
 model=make_model('graph-concept-on');before={k:v.clone()for k,v in model.state_dict().items()}
 p=diagnostic_grid()[0];report=gate_audit(model,p)
 assert report['trained']['gate_gradient'] is not None
 assert all(torch.equal(before[k],v)for k,v in model.state_dict().items())
 assert predict(model,p)==pytest.approx(predict(model,p,'permuted'),abs=1e-6)


def test_off_context_is_suppressed_and_linear_fit_uses_only_training():
 model=make_model('graph-concept-off');probe=gate_audit(model,diagnostic_grid()[0])
 assert all(r['prediction']==probe['trained']['prediction'] for r in probe.values())
 rows=[{'problem':p,'labels':{'physics_delta':exact(p)}}for p in diagnostic_grid()]
 result=baselines(rows,diagnostic_grid())
 assert result['optimizer_updates']==0
 assert result['records'][0]['analytic']==exact(diagnostic_grid()[0])
