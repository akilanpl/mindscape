import copy
import unittest
from mindscape.r5_2_recovery.__main__ import reconstruct, replay_record, diagnostics
from mindscape.r5_2_math.analysis import analyze, replay
from mindscape.r5_2_math.arithmetic import run
from mindscape.r5_2_math.verification import Teacher,verify

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.task={'identity':'fresh-fixture','split':'ood','family':'length','operands':[12,3],'operations':['add']}
        result=run(self.task,Teacher()); score=verify(self.task,result)
        score['context_diagnostics']=diagnostics(result,score,{},set())
        self.record={**{k:self.task[k] for k in ('identity','split','family')},'condition':'carry','seed':11,'result':result,'verification':score}
    def test_original_shadowing_and_zero_division(self):
        for name in ['recurrent']:
            for name,target in [('digit',1),('carry',0)]: pass
        self.assertEqual(name,'carry')
        with self.assertRaises(ZeroDivisionError): analyze([self.record],{'optimizer_seeds':[11],'families':['length']})
    def test_replay_extra_diagnostics_failure_and_repair(self):
        with self.assertRaisesRegex(ValueError,'Replay outcome changed'): replay({'tasks':[self.task]},[self.record])
        replay_record(self.task,self.record,{},set())
    def test_order_and_count_fail_closed(self):
        records=[copy.deepcopy(self.record) for _ in range(5)]
        restored=reconstruct(records,[self.task],11)
        self.assertEqual([r['condition'] for r in restored],['recurrent','reset_hidden','memory_only','zero_weight','zero_weight_memory'])
        self.assertTrue(all(r['result']==self.record['result'] for r in restored))
        self.assertTrue(all(r['condition']=='carry' for r in records))
        with self.assertRaises(ValueError): reconstruct(records[:-1],[self.task],11)
        records[2]['identity']='wrong'
        with self.assertRaises(ValueError): reconstruct(records,[self.task],11)
    def test_diagnostics_and_verifier_tampering(self):
        for key in ('context_diagnostics','exact'):
            r=copy.deepcopy(self.record)
            if key=='exact': r['verification'][key]=False
            else: r['verification'][key][0]['memory_seen']=True
            with self.assertRaises(ValueError): replay_record(self.task,r,{},set())

if __name__=='__main__': unittest.main()
