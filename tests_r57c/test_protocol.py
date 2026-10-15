import json,hashlib
from pathlib import Path
import pytest,torch
from mindscape.r57c.ordinal_model import make_model
from mindscape.r57c import runner


def test_ordinal_decoder_static_reduction_matches_masked_reference():
    for condition in runner.CONDITIONS:
        model=make_model(condition,17)
        for operation in range(2):
            for value in (-9.8,-.5,.1,5.4,9.9,18.8):
                joint=-(torch.tensor(value)-model.ordinal_centers[operation]).square()
                digit=torch.stack([torch.logsumexp(joint[model.ordinal_digits[operation]==i],0) for i in range(10)])
                carry=torch.stack([torch.logsumexp(joint[model.ordinal_carries[operation]==i],0) for i in range(2)])
                torch.testing.assert_close(digit,joint.reshape(2,10).logsumexp(0))
                actual=joint.reshape(2,10).logsumexp(1)
                torch.testing.assert_close(carry,actual.flip(0) if operation else actual)


def test_native_refusal_before_model_and_output(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'verify_identity',lambda out:{})
    def reject():raise RuntimeError('NATIVE_MPS_EXECUTION_REQUIRED')
    monkeypatch.setattr(runner,'require_mps',reject)
    with pytest.raises(RuntimeError,match='NATIVE_MPS_EXECUTION_REQUIRED'):runner.fit(tmp_path,17,'graph-concept-on')
    assert not list(tmp_path.iterdir())


def test_parameter_exposure_counts_distinguish_split_routes():
    from mindscape.r57c.ordinal_model import domain_loss
    model=make_model('graph-concept-on',17);opt=torch.optim.Adam(model.parameters(),lr=.001)
    rows=[{'problem':{'domain':'math','operation':'borrow_digit','left':2,'right':8,'incoming':1},'labels':{'math_digit':3,'math_carry':1}},
          {'problem':{'domain':'physics','mass':2.,'force':[1.,-2.],'velocity':[0.,0.],'dt':.5},'labels':{'physics_delta':[.25,-.5]}}]
    calls=[0];hook=opt.register_step_post_hook(lambda *args:calls.__setitem__(0,calls[0]+1));touches={}
    for row in rows:
        opt.zero_grad(set_to_none=True);loss=domain_loss(model.predict(row['problem']),row,torch,'cpu');loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None:touches[name]=touches.get(name,0)+1
        opt.step()
    hook.remove()
    actual={n:int(opt.state[p]['step'].item()) for n,p in model.named_parameters() if 'step' in opt.state[p]}
    assert actual==touches and calls[0]==2
    assert actual['ordinal_weights']==1 and actual['core.features.weight']==1


def test_completed_refuses_inconsistent_adam_exposure(tmp_path,monkeypatch):
    run=tmp_path/'checkpoints/17-graph-concept-on';run.mkdir(parents=True)
    monkeypatch.setattr(runner,'verify_identity',lambda out:{})
    (run/'result.json').write_text(json.dumps({'status':'COMPLETED_EXPLORATORY','optimizer_updates':2000,'optimizer_steps':2000,'step_post_hook_count':2000,
        'adam_steps_by_parameter':{'ordinal_weights':672},'gradient_exposures_by_parameter':{'ordinal_weights':2000}}))
    with pytest.raises(RuntimeError,match='Adam exposure evidence differs'):runner.completed(run)
