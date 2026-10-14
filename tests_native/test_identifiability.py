import copy
import pytest

torch=pytest.importorskip('torch')
from mindscape.r56.identifiability import (ConceptProbe,DelayedCueProbe,concept_curriculum,
 pack_concept,delayed_cue_curriculum,pack_memory,run_memory_episode,supplemental_protocol)


def test_concept_necessary_beyond_operation_and_split_identity():
    train=concept_curriculum(); held=concept_curriculum(split='heldout')
    trainstates={tuple(r['observations']['state']) for r in train}
    assert not trainstates & {tuple(r['observations']['state']) for r in held}
    group=[r for r in train if r['observations']['state']==[0,1]]
    assert len(group)==4 and len({r['labels']['class'] for r in group})==4
    inputs=[pack_concept(r['observations']) for r in group]
    assert all(torch.equal(inputs[0].state,p.state) and torch.equal(inputs[0].action,p.action) and torch.equal(inputs[0].domain,p.domain) for p in inputs)
    assert len({tuple(p.context.tolist()) for p in inputs})==4
    for concept in range(4):
        corruptions={r['corrupted_context_id'] for r in train if r['observations']['concept_id']==concept}
        assert len(corruptions)>1  # not a globally learnable ID remapping


def test_target_cannot_reach_allowlisted_features():
    row=concept_curriculum()[0]; original=pack_concept(row['observations'])
    altered=copy.deepcopy(row); altered['labels']['class']=999
    assert torch.equal(original.state,pack_concept(altered['observations']).state)
    with pytest.raises(ValueError): pack_concept(row)
    with pytest.raises(ValueError): pack_concept({**row['observations'],'target':1})
    with pytest.raises(ValueError): pack_memory({'phase':'query','value':1,'target':2})


def test_matched_concept_information_and_gradient_no_optimizer(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('No optimizer allowed')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    model=ConceptProbe(); before={k:v.detach().clone() for k,v in model.state_dict().items()}
    inputs=pack_concept(concept_curriculum()[0]['observations'])
    embedding=model(inputs,representation='embedding'); context=model(inputs,representation='context-vector')
    assert torch.equal(embedding,context)
    embedding.square().mean().backward()
    assert model.context.weight.grad.abs().sum()>0 and model.observed.weight.grad.abs().sum()>0
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_delayed_query_identifiable_only_from_past():
    episodes=delayed_cue_curriculum()
    paired=[e for e in episodes if e['observations'][-1]['value']==2]
    assert len({e['labels']['query_class'] for e in paired})==4
    assert all(torch.equal(pack_memory(paired[0]['observations'][-1]),pack_memory(e['observations'][-1])) for e in paired)
    assert all(e['observations'][1:]==paired[0]['observations'][1:] for e in paired)
    assert len({tuple(pack_memory(e['observations'][0]).tolist()) for e in paired})==4


def test_memory_controls_same_parameters_and_fusion_paths(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('No optimizer allowed')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    torch.manual_seed(7); model=DelayedCueProbe(); episodes=delayed_cue_curriculum()
    first=episodes[0]['observations']; other=episodes[4]['observations']
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    retained=run_memory_episode(model,first)
    other_retained=run_memory_episode(model,other)
    reset=run_memory_episode(model,first,control='reset')
    other_reset=run_memory_episode(model,other,control='reset')
    assert torch.equal(reset[-1]['logits'],other_reset[-1]['logits'])
    assert not torch.allclose(retained[-1]['logits'],other_retained[-1]['logits'])
    donor=[torch.zeros(model.width),other_retained[0]['memory'],other_retained[1]['memory']]
    shuffled=run_memory_episode(model,first,control='shuffled-past',shuffled_past=donor)
    assert not torch.allclose(shuffled[-1]['logits'],retained[-1]['logits'])
    (retained[-1]['logits'].square().mean()+reset[-1]['logits'].square().mean()).backward()
    for component in (model.encoder,model.memory,model.fusion,model.head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(r['memory'].abs().max()<=1 for r in retained)
    again=run_memory_episode(model,first)
    assert torch.equal(again[-1]['logits'],retained[-1]['logits']) # boundary reset
    with pytest.raises(ValueError): run_memory_episode(model,first,control='shuffled-past')


def test_supplement_never_changes_primary_grid():
    protocol=supplemental_protocol()
    assert protocol['primary_grid_modified'] is False
    assert protocol['primary_fits']==72 and protocol['primary_optimizer_steps']==144000
    assert protocol['authorization']=='UNISSUED' and protocol['qualification']=='UNISSUED'
