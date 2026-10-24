import copy,os
import pytest
import torch
from torch import nn
from research.r6.architectures.model import make_model as baseline,predict
from research.r6.common import encode_batch
from research.r61.model import summarize,make_model

def test_early_input_information_survives_summary_regression():
    # A recurrent encoder may retain a distinguishing early state while its
    # final state is equal: the exact information-loss mechanism under test.
    outputs=torch.zeros(2,4,3,requires_grad=True)
    changed=outputs.clone();changed[1,0,0]=1
    packed=nn.utils.rnn.pack_padded_sequence(changed,torch.tensor([4,4]),batch_first=True,enforce_sorted=False)
    state=torch.zeros(1,2,3)
    old=summarize(packed,state,torch.tensor([4,4]),'last')
    fixed=summarize(packed,state,torch.tensor([4,4]),os.environ.get('R61_REGRESSION_MODE','masked-mean'))
    assert torch.equal(old[0],old[1])
    assert not torch.equal(fixed[0],fixed[1])
    fixed.sum().backward()
    assert outputs.grad[0,0,0]>0

def test_padding_does_not_change_pooling():
    data=torch.tensor([[[2.],[4.],[999.]],[[1.],[3.],[5.]]])
    lengths=torch.tensor([2,3]);packed=nn.utils.rnn.pack_padded_sequence(data,lengths,batch_first=True,enforce_sorted=False)
    pooled=summarize(packed,torch.zeros(1,2,1),lengths,'masked-mean')
    assert torch.equal(pooled,torch.tensor([[3.],[3.]]))

def test_matched_parameters_and_baseline_exact():
    config={'width':48,'seed':61117};old=baseline(config);same=make_model(config);fixed=make_model({**config,'sequence_pooling':'masked-mean'})
    assert set(old.state_dict())==set(fixed.state_dict())
    assert all(torch.equal(v,fixed.state_dict()[k]) for k,v in old.state_dict().items())
    rows=[{'input':'Calculate (12) + (7).','domain':'mathematics','features':[],'target':{'answer':'19'}}, {'input':'def f(n):\n return n+2\nr=f(4)','domain':'coding','features':[],'target':{'answer':'6'}}]
    assert predict(old,rows)==predict(same,rows)
    b=encode_batch(rows,'cpu');c=fixed.encode(b['tokens'],b['lengths'],b['features'],b['domains'])
    assert c.shape==(2,48)
    assert not fixed.sequence._forward_hooks

def test_target_blind_inference_and_order():
    model=make_model({'width':48,'seed':61117,'sequence_pooling':'masked-mean'})
    rows=[{'input':t,'domain':'mathematics','features':[],'target':{'answer':str(i)}} for i,t in enumerate(['Calculate (12) + (7).','Calculate (21) - (8).'])]
    mutated=copy.deepcopy(rows)
    for row in mutated:row['target']={'unavailable':True}
    assert predict(model,rows)==predict(model,mutated)
    assert predict(model,rows)==list(reversed(predict(model,list(reversed(rows)))))

def test_no_approval_no_optimizer(tmp_path):
    from research.r61.pilot_worker import validate_approval
    config=tmp_path/'config.json';config.write_text('{}')
    with pytest.raises(RuntimeError,match='APPROVAL_REQUIRED'):
        validate_approval(config,None)

def test_local_cloud_guard():
    from research.r61.pilot_worker import require_cloud
    with pytest.raises(RuntimeError,match='KAGGLE_CUDA_BATCH_REQUIRED'):
        require_cloud()

def test_trainer_config_excludes_protected_information():
    from research.r61.prepare import trainer_config
    c={'seed':61117,'train_sha256':'train','dev_sha256':'dev','data':{'seeds':{'final_iid':63149},'records':['hidden']},'source_sha256':{'research/r6/datasets/coding.py':'generator'}}
    assert trainer_config(c)=={'seed':61117,'train_sha256':'train','dev_sha256':'dev'}

def test_approval_is_bound_to_budget_and_config(tmp_path):
    import hashlib,json
    from research.r61.pilot_worker import validate_approval
    c={'updates_per_fit_max':1200,'total_optimizer_update_ceiling':2400,'wall_seconds_per_fit_max':300,'total_cuda_fit_seconds_ceiling':600,'seed':61117}
    config=tmp_path/'c.json';config.write_text(json.dumps(c));approval=tmp_path/'a.json'
    approval.write_text(json.dumps({'approved':True,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'user_approval_text':'test fixture only; does not authorize a real execution'}))
    assert validate_approval(config,approval)==c
    c['total_optimizer_update_ceiling']=2401;config.write_text(json.dumps(c))
    with pytest.raises(RuntimeError,match='APPROVAL_NOT_BOUND'):
        validate_approval(config,approval)
