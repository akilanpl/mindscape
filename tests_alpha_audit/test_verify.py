import tempfile
import unittest
from pathlib import Path
from mindscape.alpha_audit.verify import *

class AuditTests(unittest.TestCase):
    def test_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve(); p = Path(d)/'source'; p.write_text('frozen')
            inventory = {'source': file_hash(p)}
            self.assertEqual(verify_inventory(d, inventory)['files'],1)
            p.write_text('changed')
            with self.assertRaises(AuditError): verify_inventory(d,inventory)
            (Path(d)/'link').symlink_to(p)
            with self.assertRaises(AuditError): verify_inventory(d,{'link':file_hash(p)})
            with self.assertRaises(AuditError): verify_inventory(d,{'../source':'0'*64})

    def test_disjoint(self):
        self.assertEqual(verify_disjoint({'train':['a'],'test':['b']},['old']),{'train':1,'test':1})
        for splits in ({'train':['a','a']},{'train':['a'],'test':['a']},{'test':['old']}):
            with self.assertRaises(AuditError): verify_disjoint(splits,['old'])

    def test_matching(self):
        fits = [dict(seed=s,condition=c,updates=240,target_presentations=1920,checkpoint_rule='final',
                initial_state_sha256=digest(s),target_order_sha256=digest([s]),optimizer_config_sha256=digest('Adam'))
                for s in (11,23,47) for c in ('domain_only','shared','randomized')]
        check=lambda data: verify_fit_matching(data,[11,23,47],['domain_only','shared','randomized'],240,1920)
        self.assertEqual(check(fits)['fits'],9)
        with self.assertRaises(AuditError): check(fits[:-1])
        fits[1]['target_order_sha256']=digest('changed')
        with self.assertRaises(AuditError): check(fits)
        fits[1]['target_order_sha256']=fits[0]['target_order_sha256']; fits[1]['updates']=241
        with self.assertRaises(AuditError): check(fits)

    def test_sealing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'predictions';p.write_text('synthetic')
            def chain(kinds):
                events=[]
                for i,kind in enumerate(kinds):
                    event=dict(sequence=i,previous_sha256=events[-1]['sha256'] if events else '0'*64,event=kind)
                    if kind=='PREDICTIONS_SEALED':event['prediction_sha256']=file_hash(p)
                    event['sha256']=digest(event);events.append(event)
                return events
            events=chain(['PREDICTING','PREDICTIONS_SEALED','REFERENCES_OPENED','SCORED'])
            self.assertEqual(verify_event_chain(events,p)['status'],'VERIFIED')
            for kinds in (['SCORED','PREDICTIONS_SEALED'],['PREDICTIONS_SEALED','TRAINING','SCORED'],['PREDICTIONS_SEALED']):
                with self.assertRaises(AuditError):verify_event_chain(chain(kinds),p)
            p.write_text('changed')
            with self.assertRaises(AuditError):verify_event_chain(events,p)

    def test_partial_accuracy(self):
        result=exact_metrics({'a':dict(answer=0,states=[1,2,9])},{'a':dict(answer=3,states=[1,2,3])})
        self.assertEqual(result['whole_answer_exactness'],0)
        self.assertAlmostEqual(result['intermediate_accuracy'],2/3)
        with self.assertRaises(AuditError):exact_metrics({}, {'a':dict(answer=3,states=[1])})

    def test_wilson(self):
        self.assertAlmostEqual(wilson(0,100)[1],0.03699349820698568)
        self.assertAlmostEqual(wilson(50,100)[0],0.4038315303659956)
        with self.assertRaises(AuditError):wilson(1,0)

    def test_status(self):
        self.assertEqual(scientific_status(executed=False,complete=True,independent=True,gates={'engineering':True},mandatory_gates=['engineering']),'UNRUN')
        self.assertEqual(scientific_status(executed=True,complete=False,independent=True,gates={'x':True},mandatory_gates=['x']),'FAILED_CAMPAIGN')
        self.assertEqual(scientific_status(executed=True,complete=True,independent=False,gates={'x':True},mandatory_gates=['x']),'UNVERIFIED')
        self.assertEqual(scientific_status(executed=True,complete=True,independent=True,gates={'x':False},mandatory_gates=['x']),'VERIFIED_NO_GO')

    def test_missing_frozen_gate(self):
        with self.assertRaises(AuditError): scientific_status(executed=True,complete=True,independent=True,gates={'x':True},mandatory_gates=['x','y'])

if __name__=='__main__':unittest.main()
