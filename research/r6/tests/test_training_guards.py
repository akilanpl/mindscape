import importlib
from collections import Counter
from unittest.mock import patch
import pytest
from research.r6.training.worker import require_cloud, validate_config, BalancedSampler


def test_mac_rejects_before_torch_or_optimizer_import():
    original=__import__
    def guarded(name,*args,**kwargs):
        if name=='torch':raise AssertionError('Torch imported before host rejection')
        return original(name,*args,**kwargs)
    with patch('platform.system',return_value='Darwin'),patch('builtins.__import__',side_effect=guarded):
        with pytest.raises(RuntimeError,match='no optimizer constructed'):require_cloud()


def test_bounded_config_rejects_budget_expansion():
    config={'fit_id':'test','seed':61017,'updates':600,'batch_per_domain':8,'optimizer':'adam','lr':.001,'clip_norm':1.,'max_seconds':3600,
            'physics_target_mean':[0]*4,'physics_target_scale':[1]*4}
    validate_config(config)
    for key,value in [('updates',601),('batch_per_domain',9),('optimizer','sgd'),('physics_target_scale',[0]*4),('max_seconds',3601)]:
        with pytest.raises(ValueError):validate_config({**config,key:value})


def test_balanced_sampler_reproducibility_without_torch():
    rows=[{'record_id':f'{d}-{i}','domain':d} for d in ('mathematics','coding','physics') for i in range(10)]
    a=BalancedSampler(rows,61017);b=BalancedSampler(rows,61017)
    for _ in range(20):
        batch=a.next();assert batch==b.next()
        assert Counter(r['domain'] for r in batch)=={'mathematics':8,'coding':8,'physics':8}


def test_mixed_loss_masks_and_gradient_connectivity_without_fitting():
    import torch
    from research.r6.training.worker import _loss
    logits=torch.zeros(3,2,14,requires_grad=True)
    physics=torch.zeros(3,4,requires_grad=True)
    batch={'physics_mask':torch.tensor([False,True,False]),'text_mask':torch.tensor([True,False,True]),
           'target_output':torch.tensor([[4,2],[2,0],[5,2]]),'physics_target':torch.tensor([[0.]*4,[2.]*4,[0.]*4])}
    total,text,physical=_loss(torch,{'text_logits':logits,'physics':physics},batch)
    assert float(physical.detach())==4.
    assert float(text.detach())==pytest.approx(2.6390573)
    total.backward()
    assert not logits.grad[1].any() and physics.grad[1].abs().sum()>0
    assert not physics.grad[[0,2]].any()
