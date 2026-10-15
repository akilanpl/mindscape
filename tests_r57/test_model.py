import pytest
import torch
from mindscape.r57.model import make_model


def coding(names=None,requested='value'):
 return {'domain':'coding','source':'x=1\nprint(x)','old_name':'x','candidates':names or ['value','count','total','result'],'requested_name':requested,'concept_ids':[3]}


def test_common_graph_initialization_and_zero_gated_match():
 on=make_model('graph-concept-on',17);off=make_model('graph-concept-off',17)
 assert all(torch.equal(v,off.state_dict()[k])for k,v in on.state_dict().items())
 p={'domain':'physics','mass':2.,'force':[2.,-3.],'dt':.5,'velocity':[0.,0.],'concept_ids':[2]}
 assert torch.equal(on.predict(p)['physics_delta'],off.predict(p)['physics_delta'])


@pytest.mark.parametrize('condition',['graph-concept-on','graph-concept-off','matched-gru'])
def test_candidate_permutation_equivariance_and_label_rejection(condition):
 model=make_model(condition).eval();p=coding();base=model.predict(p)['coding_logits'];order=[2,0,3,1]
 changed=coding([p['candidates'][i] for i in order]);actual=model.predict(changed)['coding_logits']
 assert torch.allclose(actual,base[order],atol=1e-7)
 with pytest.raises(ValueError):model.prepare({**p,'target_name':'value'})
 with pytest.raises(ValueError):model.prepare({**p,'labels':{'index':0}})


def test_numeric_sensitivity_gradients_and_safe_state_reload():
 for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
  model=make_model(condition);p={'domain':'math','operation':'add_digit','left':2,'right':3,'incoming':0,'concept_ids':[0]}
  a=model.predict(p)['math_digit_logits'];b=model.predict({**p,'left':4})['math_digit_logits']
  assert not torch.equal(a,b)
  a.square().mean().backward()
  assert model.math_head[0].weight.grad.abs().sum()>0
  clone=make_model(condition);clone.load_state_dict(model.state_dict())
  assert torch.equal(model.predict(p)['math_digit_logits'],clone.predict(p)['math_digit_logits'])
  p={'domain':'physics','mass':2.,'force':[2.,-3.],'dt':.5,'velocity':[0.,0.],'concept_ids':[2]}
  assert not torch.equal(model.predict(p)['physics_delta'],model.predict({**p,'force':[-2.,-3.]})['physics_delta'])


def test_names_use_learned_characters_not_equality_solver():
 model=make_model('graph-concept-on');out=model.predict(coding())
 out['coding_logits'].square().mean().backward()
 assert model.name_encoder.embedding.weight.grad.abs().sum()>0
 # Alpha renaming is not mathematically enforced; measured difference is retained.
 changed=coding(['a','b','c','d'],requested='a')
 assert not torch.equal(out['coding_logits'],model.predict(changed)['coding_logits'])
