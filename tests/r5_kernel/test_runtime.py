import copy,json,sys,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mindscape.r5_kernel import Kernel,Request
from mindscape.r5_kernel.examples import mathematics,coding,physics,probes
from mindscape.r5_kernel.shared import encode
from mindscape.r5_kernel.evidence import asset,registry,sha
from mindscape.r5_kernel.resources import Resources
from mindscape.r5_kernel.__main__ import comparable

def available(name):
    from mindscape.r5_kernel.evidence import BASE
    path=Path(registry()['assets'][name]['path'])
    return (path if path.is_absolute() else BASE/path).is_file()

class RuntimeTests(unittest.TestCase):
    def setUp(self):self.kernel=Kernel()
    def tearDown(self):self.kernel.close()
    def test_exact_symbolic_integer_and_direct_parity(self):
        request=mathematics()[0];response=self.kernel.run(request);direct=self.kernel.direct(request)
        self.assertEqual(response.prediction['value'],'-432');self.assertTrue(response.verification['success'])
        self.assertEqual(response.prediction,direct['prediction']);self.assertFalse(response.provenance['learned_transfer'])
        self.assertEqual([t['stage'] for t in response.trace],['observation','state','action','prediction','execution','verification'])
        self.assertIsNone(self.kernel.resources.key)
    def test_all_bounded_math_families(self):
        fixtures=[('division',{'a':-17,'b':-5},{'quotient':'4','remainder':'3'}),('fractions',{'a':[2,3],'b':[5,6],'operation':'add'},'3/2'),('algebra',{'coefficients':[1,2,3],'x':2},'17'),('equation',{'a':3,'b':2,'c':11},'3'),('matrix',{'a':[[1,2],[3,4]],'b':[[2,0],[0,2]]},[['2','4'],['6','8']]),('geometry',{'a':[0,0],'b':[3,4]},{'distance_squared':'25','distance':'5'}),('calculus',{'coefficients':[1,2,3],'lower':0,'upper':2},{'derivative':'6*x + 2','definite_integral':'14'}),('statistics',{'values':[1,2,3]},{'mean':'2','population_variance':'2/3'}),('probability',{'successes':2,'trials':5},'2/5')]
        for op,obs,expected in fixtures:
            with self.subTest(op=op):
                r=self.kernel.run(Request('mathematics',op,obs));self.assertEqual(r.prediction['value'],expected);self.assertTrue(r.verification['success']);self.assertEqual(r.provenance['method'],'symbolic_tool')
    def test_precision_and_json_reproducibility(self):
        req=Request('mathematics','integer',{'operands':[10**20,-1],'operations':['add']});a=self.kernel.run(req);b=self.kernel.run(Request(**json.loads(json.dumps(asdict(req)))))
        self.assertEqual(a.prediction['value'],str(10**20-1));self.assertEqual(a.record(),b.record())
    def test_unsupported_private_and_invalid_requests(self):
        fixtures=[Request('mathematics','fractions',{'a':[1,0],'b':[1,2],'operation':'add'}),Request('mathematics','unknown',{}),Request('mathematics','integer',{'operands':[1,2],'operations':['add'],'expected':3})]
        for req in fixtures:self.assertEqual(self.kernel.handle(asdict(req))['status'],'REJECTED')
        with self.assertRaises(ValueError):Request('physics','forecast',{'x':float('nan')})
        with self.assertRaises(ValueError):Request('mathematics','integer',{},version='future')
    def test_observation_mutation_is_copied(self):
        obs={'operands':[1,2],'operations':['add']};r=Request('mathematics','integer',obs);obs['operands'][0]=100
        self.assertEqual(self.kernel.run(r).prediction['value'],'3')
    def test_bad_neural_prediction_is_preserved(self):
        class Wrong:
            def reset(self):pass
            def sign(self,*args):return 0,{'kind':'test_double'}
            def local(self,state):return 0,0,{'kind':'test_double'}
        with patch.object(self.kernel.resources,'math',return_value=(Wrong(),{'method':'test_double','neural':False,'test_double':True})):
            r=self.kernel.run(Request('mathematics','integer',{'operands':[7,8],'operations':['add']},mode='experimental_neural'))
        self.assertEqual(r.prediction['value']['answer'],'0');self.assertFalse(r.verification['success']);self.assertEqual(r.verification['details']['expected'],'15')
        self.assertEqual(r.prediction['value'],r.execution['value'])
    def test_physics_executed_parity_and_unverified_forecast(self):
        req,ctx=physics();r=self.kernel.run(req,ctx);direct=self.kernel.direct(req,ctx)
        self.assertEqual(r.prediction,direct['prediction']);self.assertTrue(r.verification['execution_verified']);self.assertGreater(r.verification['prediction_mse'],0)
        self.assertEqual(r.prediction['kind'],'procedural_prediction');self.assertIn('uncalibrated',r.provenance['uncertainty'])
        unverified=self.kernel.run(req);self.assertIsNone(unverified.verification['success']);self.assertEqual(unverified.execution['kind'],'not_executed')
    def test_private_physics_never_enters_predictor(self):
        req,ctx=physics()
        from mindscape.r5_kernel.physics_adapter import predict
        with patch('mindscape.r5_kernel.physics_adapter.predict',wraps=predict) as spy:
            self.kernel.run(req,ctx)
        self.assertEqual(len(spy.call_args.args),4);self.assertNotIn('scene',spy.call_args.args[0])
        bad=copy.deepcopy(req.observation);bad['future_states']=[]
        self.assertEqual(self.kernel.handle(asdict(Request('physics','forecast',bad,req.action)),ctx)['status'],'REJECTED')
        ctx['scene']['gravity']=8
        self.assertEqual(self.kernel.handle(asdict(req),ctx)['status'],'REJECTED')
    @unittest.skipUnless(available('wasi') and available('wasi_compiled'),'Preserved local WASI artifacts required')
    def test_coding_real_wasi_parity_and_isolation(self):
        req,_=coding();a=self.kernel.run(req);b=self.kernel.direct(req)
        self.assertTrue(a.verification['success']);self.assertTrue(a.execution['value']['actual_vm_call']);self.assertEqual(comparable(a.execution),comparable(b['execution']))
        obs=copy.deepcopy(req.observation);obs['repository']={'solution.py':'import os\ndef total(x):\n return 0\n'}
        self.assertEqual(self.kernel.handle(asdict(Request('coding','execute',obs)))['status'],'REJECTED')
        self.assertIsNone(self.kernel.resources.key)
    def test_coding_private_context_not_request(self):
        req,ctx=coding(True);obs=copy.deepcopy(req.observation);obs['private_tests']=ctx['private_tests']
        self.assertEqual(self.kernel.handle(asdict(Request('coding','repair',obs)),ctx)['status'],'REJECTED')
        self.assertEqual(self.kernel.handle(asdict(req))['status'],'REJECTED')
    def test_shared_adapters_and_hidden_fields(self):
        for domain,public in probes().items():
            g=encode(domain,public);self.assertEqual(g['domain'],domain);self.assertEqual(len(g['action']),8)
            with self.assertRaises(ValueError):encode(domain,{**public,'target':123})
        g=encode('coding',probes()['coding']);self.assertEqual(g['observed'][-1][0],0);self.assertEqual(g['entities'][-1][0],0)
    @unittest.skipUnless(all(available(n) for n in ('shared_core','math_neural','physics_neural')),'Preserved native checkpoints required')
    def test_checkpoint_cpu_inventory_and_tamper_rejection(self):
        import torch
        for name,key,count in [('shared_core','model',6967),('math_neural','state',None),('physics_neural','state',None)]:
            path,row=asset(name);saved=torch.load(path,map_location='cpu',weights_only=True)
            self.assertTrue(all(v.device.type=='cpu' and torch.isfinite(v).all() for v in saved[key].values()))
            if count:self.assertEqual(sum(v.numel() for v in saved[key].values()),count)
        modified=copy.deepcopy(registry());modified['assets']['shared_core']['sha256']='0'*64
        with patch('mindscape.r5_kernel.evidence.registry',return_value=modified):
            with self.assertRaises(ValueError):asset('shared_core')
    def test_hidden_shared_values_reject_before_model_loading(self):
        g=encode('coding',probes()['coding']);g['entities'][-1][0]=99
        req=Request('coding','transition',{k:g[k] for k in ('entities','observed','relations')},{'encoding':g['action']},'experimental_shared')
        with patch.object(self.kernel.resources,'shared') as loader:
            self.assertEqual(self.kernel.handle(asdict(req))['status'],'REJECTED');loader.assert_not_called()
    def test_proposal_is_frozen_and_has_no_training_endpoint(self):
        from mindscape.r5_kernel.contract import digest
        p=Path('configs/r5_kernel');proposal=json.loads((p/'shared-study-proposal.json').read_text());freeze=json.loads((p/'proposal-freeze.json').read_text())
        self.assertEqual(digest(proposal),freeze['proposal_hash']);self.assertEqual(freeze['authorization'],'NOT_GRANTED');self.assertEqual(freeze['optimizer_steps'],0)
        for d in ('mathematics','coding','physics'):
            r=self.kernel.handle(asdict(Request(d,'train',{})))
            self.assertEqual(r['status'],'REJECTED')
        self.assertIsNone(self.kernel.resources.value)
    def test_one_model_slot_releases_prior_asset(self):
        r=Resources();a=r.slot('a',lambda:object());self.assertIs(a,r.slot('a',lambda:None));b=r.slot('b',lambda:object());self.assertIsNot(a,b);self.assertEqual(r.loads,['a','b']);r.close();self.assertIsNone(r.value)

if __name__=='__main__':unittest.main()
