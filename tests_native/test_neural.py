"""Engineering checks only: no optimizer construction, updates or fitted claims."""
import copy
import pytest

torch=pytest.importorskip('torch')
from mindscape.r56.neural import GraphInput, ModelConfig, R56Model
from mindscape.r56.native_domains import pack_problem, infer_problem, verify_prediction


def math_problem(**extra):
    return {'domain':'math','operation':'add_digit','left':7,'right':8,'incoming':1,'concept_ids':[0,2],**extra}


def test_graph_model_backward_connects_original_components():
    torch.manual_seed(17)
    model=R56Model(ModelConfig(width=24))
    out=model(pack_problem(math_problem()))
    loss=sum(out[key].square().mean() for key in ('math_digit_logits','math_carry_logits','coding_logits','physics_delta','action_scores'))
    loss.backward()
    for name in ('features','types','relations','concepts','domains','action','message','update','working_memory','state_fusion','math_digit','math_carry','coding_choice','physics_effect','action_score'):
        parameters=list(getattr(model,name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),name
        assert sum(p.grad.abs().sum().item() for p in parameters)>0,name


def test_variable_graph_edges_action_and_concepts_affect_predictions():
    torch.manual_seed(7); model=R56Model(ModelConfig(width=24)).eval()
    graph=pack_problem(math_problem())
    base=model(graph)['math_digit_logits']
    changed=copy.deepcopy(graph); changed.relation_types=(changed.relation_types+1)%8
    assert not torch.allclose(base,model(changed)['math_digit_logits'])
    changed=copy.deepcopy(graph); changed.action=torch.tensor([0.,1.,0.,0.])
    assert not torch.allclose(base,model(changed)['math_digit_logits'])
    changed=copy.deepcopy(graph); changed.concept_ids=torch.tensor([3,4])
    assert not torch.allclose(base,model(changed)['math_digit_logits'])
    changed=copy.deepcopy(graph)
    changed.node_features=torch.cat([graph.node_features,torch.zeros(1,8)])
    changed.node_types=torch.cat([graph.node_types,torch.tensor([3])])
    assert model(changed)['node_states'].shape==(4,24)


def test_memory_bounded_and_stateless_ignores_prior():
    model=R56Model(ModelConfig(width=24)).eval(); graph=pack_problem(math_problem())
    first=model(graph)
    second=model(graph,first['memory'])
    assert second['memory'].abs().max()<=1
    assert not torch.allclose(first['math_digit_logits'],second['math_digit_logits'])
    a=model(graph,torch.ones(24),stateless=True)
    b=model(graph,-torch.ones(24),stateless=True)
    assert torch.equal(a['math_digit_logits'],b['math_digit_logits'])


def test_verifier_separate_and_predictions_never_corrected(monkeypatch):
    model=R56Model(ModelConfig(width=24))
    with torch.no_grad():
        model.math_digit.weight.zero_(); model.math_digit.bias.zero_(); model.math_digit.bias[3]=10
        model.math_carry.weight.zero_(); model.math_carry.bias[:]=torch.tensor([10.,0.])
    problem=math_problem(); original=copy.deepcopy(problem)
    result=infer_problem(model,problem)
    assert result['prediction']['digit']==3 and result['prediction']['outgoing']==0
    assert not result['verification']['passed']
    assert problem==original and model.training
    full=infer_problem(model,{'domain':'math','operation':'add','left':'99','right':'99'})
    assert full['prediction']['answer']=='33'
    assert full['prediction']['transitions'][1]['outgoing']==0
    assert not full['verification']['passed']


def test_borrow_reference_and_carry_alias():
    problem=math_problem(operation='borrow_digit',left=2,right=8,incoming=1)
    assert verify_prediction(problem,{'digit':3,'outgoing':1})['passed']
    p=math_problem(); p.pop('incoming'); p['carry_in']=1
    assert torch.equal(pack_problem(p).node_features,pack_problem(math_problem()).node_features)


def test_coding_graph_has_no_target_labels_and_ast_candidate_application():
    p={'domain':'coding','source':'def f(x):\n return x+1','old_name':'x','candidates':['value','count'],'target_name':'value','requested_name':'value'}
    other={**p,'target_name':'count'}
    a,b=pack_problem(p),pack_problem(other)
    assert torch.equal(a.node_features,b.node_features)
    assert torch.equal(a.edges,b.edges)
    check=verify_prediction(p,{'name':'count','candidate_index':1})
    assert not check['passed'] and 'return count + 1' in check['renamed_source']
    assert len(a.node_features)>3
    requested=pack_problem({**p,'requested_name':'count'})
    assert not torch.equal(a.node_features,requested.node_features)


def test_physics_learned_output_and_external_reference():
    p={'domain':'physics','velocity':[1,2],'mass':2,'force':[4,0],'impulse':[0,2],'dt':.5}
    assert verify_prediction(p,{'velocity':[2,3]})['passed']
    model=R56Model(ModelConfig(width=24))
    with torch.no_grad(): model.physics_effect.weight.zero_(); model.physics_effect.bias.zero_()
    r=infer_problem(model,p)
    assert r['prediction']['velocity']==[1,2]
    assert not r['verification']['passed']


def test_validation_limits_and_no_optional_training():
    model=R56Model(ModelConfig(width=24))
    graph=pack_problem(math_problem())
    graph.edges[0,0]=99
    with pytest.raises(ValueError): model(graph)
    with pytest.raises(ValueError): pack_problem(math_problem(concept_ids=[32]))
    with pytest.raises(ValueError): pack_problem({'domain':'physics','mass':0,'velocity':[0,0]})
    with pytest.raises(ValueError): infer_problem(model,{'domain':'math','operation':'add','left':'-1','right':'2'})


def test_inference_never_constructs_optimizer_or_uses_verifier_target(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Inference must not construct an optimizer')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    torch.manual_seed(17)
    model=R56Model(ModelConfig(width=24))
    problem={'domain':'coding','source':'x = 1\nprint(x)','old_name':'x',
             'candidates':['value','count'],'requested_name':'value','target_name':'value'}
    first=infer_problem(model,problem)
    second=infer_problem(model,{**problem,'target_name':'count'})
    assert first['prediction']==second['prediction']
    assert first['verification']['passed'] != second['verification']['passed']
    assert first['learned_claim'] is False
