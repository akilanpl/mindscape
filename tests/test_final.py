import tempfile, unittest
from pathlib import Path
import numpy as np
from mindscape.models.final_policy import FinalPolicy
from mindscape.models.numpy_backend import NumpyMLP
from mindscape.memory.concept import ConceptMemory
from mindscape.environments.multiplication.claims import expected_value

class FinalTests(unittest.TestCase):
 def test_final_cli_help(self):
  import subprocess,sys
  root=Path(__file__).resolve().parents[1]
  for script in ['run_final_study.py','evaluate_final.py','summarize_final_costs.py','freeze_final_artifacts.py','build_final_demo.py']:
   result=subprocess.run([sys.executable,str(root/'scripts'/script),'--help'],cwd=root,capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('usage:',result.stdout)
 def test_environment_interface_domain_neutral(self):
  from mindscape.environments.base import Environment
  from mindscape.environments.multiplication.claims import ClaimsEnvironment
  from mindscape.environments.multiplication.environment import MultiplicationEnvironment
  self.assertIsInstance(ClaimsEnvironment(),Environment);self.assertIsInstance(MultiplicationEnvironment(),Environment)
 def test_concept_memory_descriptive(self):
  m=ConceptMemory();m.store('cursor','Select a numerical claim');self.assertEqual(m.retrieve('cursor'),'Select a numerical claim');self.assertIsNone(m.retrieve('missing'))
 def test_primary_has_no_singleton_or_oracle_feedback(self):
  model=FinalPolicy(NumpyMLP([10]*8+[2],seed=2),'trajectory',dream=False)
  p=model.predict({'observation':{'operands':[12,3],'source':'user_input','kind':'observation'}})
  self.assertFalse(p.diagnostics['oracle_feedback_visible'])
  for d in p.diagnostics['decisions']:
   self.assertEqual(d['total_candidate_actions'],100);self.assertEqual(len(d['valid_actions']),100);self.assertNotIn('correct_action',d);self.assertFalse(d['hypothetical']);self.assertFalse(d['actual_result']['hypothetical']);self.assertEqual(d['selected_value'],d['actual_result']['value'])
 def test_open_loop_only_hypothetical(self):
  m=FinalPolicy(NumpyMLP([10]*8+[2]),'trajectory',dream=False,no_interaction=True);p=m.predict({'observation':{'operands':[2,3],'source':'user_input','kind':'observation'}})
  self.assertIsNone(p.trajectory);self.assertTrue(all(d['hypothetical'] for d in p.diagnostics['decisions']))
  from mindscape.data.claims_backend import ClaimsBackend
  ClaimsBackend().annotate(None,p.diagnostics)
  self.assertTrue(all('correct_action' in d and d['transition_valid'] is None for d in p.diagnostics['decisions']))
 def test_real_backend_reload_and_active_loss(self):
  try:import torch
  except ImportError:self.skipTest('optional torch dependency')
  from mindscape.models.final_backend import FinalBackend
  class Encoder:
   size=52;parameter_count=10;path='test';revision='test'
   def encode(self,x):return np.asarray(x,dtype=np.float32)
  e=Encoder();b=FinalBackend(e,0);x=np.eye(52,dtype=np.float32)[:3];y=np.zeros((3,9),dtype=int)
  before=b.logits(x);inactive=b.head[4].weight.detach().numpy()[20:].copy();stats=b.fit(x,y,3,0,2);self.assertEqual(stats['active_heads'],2);self.assertFalse(np.array_equal(before,b.logits(x)));self.assertTrue(np.array_equal(inactive,b.head[4].weight.detach().numpy()[20:]))
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'model';b.save(p);loaded=FinalBackend.load(p,e);self.assertTrue(np.array_equal(b.logits(x),loaded.logits(x)))
   import json
   meta=json.loads((p/'backend.json').read_text());meta['backend']='unknown';(p/'backend.json').write_text(json.dumps(meta))
   with self.assertRaises(ValueError):FinalBackend.load(p,e)
if __name__=='__main__':unittest.main()
