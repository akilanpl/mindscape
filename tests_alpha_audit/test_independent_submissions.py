"""Independent adversarial receipt fixtures; no scientific task generation or models."""
import tempfile
import unittest
from pathlib import Path
from mindscape.alpha_audit.verify import (AuditError, digest, exact_metrics, file_hash,
    scientific_status, verify_event_chain, verify_fit_matching, verify_inventory, wilson)


class IndependentReviewTests(unittest.TestCase):
    def test_empty_and_duplicate_fit_designs_rejected(self):
        for seeds, conditions in (([], ['x']), ([11], []), ([11, 11], ['x']), ([11], ['x', 'x'])):
            with self.assertRaises(AuditError):
                verify_fit_matching([], seeds, conditions, 240, 1920)
        for updates, exposure in ((0,1920),(240,False),(240,1920.0)):
            with self.assertRaises(AuditError):
                verify_fit_matching([], [11], ['x'], updates, exposure)

    def test_extra_trace_and_boolean_state_rejected(self):
        truth = {'invented': {'answer': 8, 'states': [3, 8]}}
        for states in ([3], [3, 8, 99], [True, 8]):
            with self.assertRaises(AuditError):
                exact_metrics({'invented': {'answer': 8, 'states': states}}, truth)

    def test_release_and_scoring_order_adversarial(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory)/'receipt'; prediction.write_text('invented fixture')
            def chain(kinds):
                events=[]
                for i, kind in enumerate(kinds):
                    event={'sequence':i,'event':kind,'previous_sha256':events[-1]['sha256'] if events else '0'*64}
                    if kind=='PREDICTIONS_SEALED': event['prediction_sha256']=file_hash(prediction)
                    event['sha256']=digest(event);events.append(event)
                return events
            valid=['PREDICTIONS_SEALED','REFERENCES_OPENED','SCORED']
            self.assertEqual(verify_event_chain(chain(valid),prediction)['status'],'VERIFIED')
            for kinds in (['PREDICTIONS_SEALED','SCORED'], valid+['SCORED'],
                          valid+['REFERENCES_OPENED'], valid+['PREDICTING']):
                with self.assertRaises(AuditError): verify_event_chain(chain(kinds),prediction)

    def test_inventory_rejects_symlink_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); (root/'real').mkdir(); (root/'real/file').write_text('fixture')
            (root/'alias').symlink_to(root/'real',target_is_directory=True)
            with self.assertRaises(AuditError): verify_inventory(root,{'alias/file':file_hash(root/'real/file')})

    def test_engineering_cannot_promote_unrun(self):
        self.assertEqual(scientific_status(executed=False,complete=True,independent=True,
            gates={'engineering':True},mandatory_gates=['engineering']),'UNRUN')
        with self.assertRaises(AuditError): wilson(True,3)
        self.assertLess(wilson(3,3)[0], .95)


if __name__ == '__main__': unittest.main()
