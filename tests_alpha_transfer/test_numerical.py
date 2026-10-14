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
    def test_loss_rejects_fractional_active_or_label(self):
        for active,valid in [(torch.tensor([[.5]]),torch.ones(1)),(torch.ones(1,1),torch.tensor([.5]))]:
            with self.assertRaises(ValueError): balanced_loss(torch.ones(1,1,1),torch.zeros(1),torch.zeros(1,1,1),valid,active)
    def test_adam_rejects_boolean_step_negative_variance(self):
        for step,variance in [(torch.tensor(True),0.),(torch.tensor(1.),-1.)]:
            model=GraphTransition(11); opt=torch.optim.Adam(model.parameters()); p=next(model.parameters())
            opt.state[p]=dict(step=step,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.full_like(p,variance))
            with self.assertRaises(ValueError): snapshot(model,opt)
    def test_external_signatures_purpose_content_and_key(self):
        # Ephemeral TEST key only; no scientific authorizations persisted.
        import base64,json
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
        from mindscape.alpha_transfer.numerical_authority import verify_signed
        key=Ed25519PrivateKey.generate(); public=key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
        payload=dict(purpose='TEST_ONLY',binding={'synthetic':'fixture'},signer='TEST',decision_id='TEST',sequence=1,decision='AUTHORIZE')
        receipt=dict(payload=payload,signature=base64.b64encode(key.sign(json.dumps(payload,sort_keys=True,separators=(',',':')).encode())).decode())
        verify_signed(receipt,'TEST_ONLY',payload['binding'],{('TEST_ONLY','TEST'):public})
        for purpose,binding,keys in [('training',payload['binding'],{}),('TEST_ONLY',{'synthetic':'changed'},{('TEST_ONLY','TEST'):public}),('TEST_ONLY',payload['binding'],{})]:
            with self.assertRaises(ValueError): verify_signed(receipt,purpose,binding,keys)
    def test_raw_clone_cannot_follow_mutation(self):
        from mindscape.alpha_transfer.numerical import raw_state
        model=GraphTransition(11); opt=torch.optim.Adam(model.parameters()); raw=raw_state(model,opt)
        name,p=next(iter(model.named_parameters())); before=raw['parameters'][name].clone()
        with torch.no_grad(): p.add_(1)
        self.assertTrue(torch.equal(before,raw['parameters'][name]))
    def test_source_lock_corruption_omission_traversal(self):
        from pathlib import Path
        import hashlib
        from mindscape.alpha_transfer.numerical_locks import MANDATORY,verify_source,checked_path
        manifest={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in MANDATORY}
        verify_source('.',manifest)
        broken=manifest.copy(); broken[MANDATORY[0]]='0'*64
        with self.assertRaises(ValueError): verify_source('.',broken)
        missing=manifest.copy(); missing.pop(MANDATORY[0])
        with self.assertRaises(ValueError): verify_source('.',missing)
        with self.assertRaises(ValueError): checked_path('.','../outside')
    def test_source_lock_symlink(self):
        import tempfile
        from pathlib import Path
        from mindscape.alpha_transfer.numerical_locks import checked_path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'actual').write_text('fixture'); (root/'alias').symlink_to(root/'actual')
            with self.assertRaises(ValueError): checked_path(root,'alias')
    def test_verdict_empty_or_forged_status_rejected(self):
        from mindscape.alpha_transfer.numerical_authority import verify_verdict
        report=dict(status='GATES_PASS_PENDING_INDEPENDENT_AUDIT',gates={})
        with self.assertRaises(ValueError): verify_verdict(report,dict(bindings=dict(source='0'*64)),[],{}, {})
