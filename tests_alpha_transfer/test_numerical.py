import unittest
from unittest.mock import patch
import torch
from mindscape.alpha_transfer.numerical import balanced_loss, snapshot, GraphTransition, route_core, NativeCallbacks, target_tensors

class NumericalTests(unittest.TestCase):
    def test_architecture_backward_frozen_aux(self):
        model=GraphTransition(11); self.assertEqual(sum(p.numel() for p in model.parameters()),6967)
        opt=torch.optim.Adam(model.parameters())
        with patch('mindscape.v5.model.verify_placement'):
            with route_core(model,'domain_only','mathematics'):
                x=torch.zeros(2,7,6); masks=torch.ones_like(x); active=torch.ones(2,7)
                effect,valid=model('mathematics',x,masks,torch.ones(2,7,7,3),active,torch.zeros(2,8))
                loss=balanced_loss(effect,valid,torch.ones_like(effect),torch.ones(2),active); loss.backward()
                state=snapshot(model,opt)
                self.assertIsNone(state['core']['gradient'])
                self.assertGreater(state['encoder.mathematics']['gradient'],0)
                self.assertIsNone(state['encoder.coding']['gradient'])
        self.assertFalse(opt.state)
    def test_unauthorized_no_initialization(self):
        adapter=NativeCallbacks(None,None,None)
        with self.assertRaises(ValueError): adapter.initialize(11,'shared',{})
    def test_invalid_effect_shape(self):
        with self.assertRaises(ValueError): target_tensors([dict(domain='coding',target=dict(delta=[[1,2]],valid=1))],'cpu')
    def test_nonfinite_loss(self):
        with self.assertRaises(ValueError): balanced_loss(torch.tensor([[[float('nan')]]]),torch.ones(1),torch.ones(1,1,1),torch.ones(1),torch.ones(1,1))
    def test_adam_corrupt_step(self):
        model=GraphTransition(11); opt=torch.optim.Adam(model.parameters()); p=next(model.parameters())
        opt.state[p]=dict(step=torch.tensor(.5),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        with self.assertRaises(ValueError): snapshot(model,opt)
    def test_adam_corrupt_shape(self):
        model=GraphTransition(11); opt=torch.optim.Adam(model.parameters()); p=next(model.parameters())
        opt.state[p]=dict(step=torch.tensor(1.),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros_like(p))
        with self.assertRaises(ValueError): snapshot(model,opt)
    def test_raw_prediction_rejection(self):
        from mindscape.alpha_transfer.numerical_scoring import task_metrics
        target=dict(delta=[[1.]],valid=1)
        for record in (dict(delta=[[float('nan')]],valid_probability=.5),dict(delta=[[1.]],valid_probability=True),dict(delta=[[1.,2.]],valid_probability=.5)):
            with self.assertRaises(ValueError): task_metrics(record,target)
    def test_balanced_strata_exact(self):
        loss=balanced_loss(torch.zeros(1,3,1),torch.zeros(1),torch.tensor([[[2.],[0.],[0.]]]),torch.ones(1),torch.ones(1,3))
        self.assertAlmostEqual(float(loss),2+.1*__import__('math').log(2),places=6)
    def test_source_retains_native_guard(self):
        import inspect
        self.assertIn('require_mps()',inspect.getsource(NativeCallbacks.initialize))
