import pytest
from mindscape.r57.failure_audit import entropy, gradient_audit, signal_probe, bounded_sampling_probe
from mindscape.r57.data import split_rows


def test_entropy_and_update_ceiling():
    assert entropy([0,0,1,1])==1
    with pytest.raises(ValueError): bounded_sampling_probe([],301)


def test_real_gradient_margins_and_signal_no_updates():
    torch=pytest.importorskip('torch')
    from mindscape.r56.neural import R56Model
    torch.manual_seed(17); model=R56Model()
    rows=split_rows(17,'dev',set())
    selected=[rows[0],rows[80],rows[160]]
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=gradient_audit(model,selected)
    assert result['optimizer_updates']==0
    assert all(r['gradient_l2']>0 for r in result['records'])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    p=rows[80]['problem']; q={**p,'requested_name':p['candidates'][1]}
    probe=signal_probe(model,p,q)
    assert probe['encoded_l2']>0 and probe['coding_logits_l2']>0
