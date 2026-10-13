"""Nonneural regressions for native float32 versus JSON float64 scoring."""
import ast,json,unittest
from pathlib import Path
import numpy as np
from mindscape.r5_3_physics.uncertainty import interval_score
from mindscape.r5_3_recovery.numerics import native_bounds,score_tolerance,acceptable
from mindscape.r5_3_recovery.__main__ import compare_tree

class RecoveryTests(unittest.TestCase):
    def test_exact_equality_fails_lossless_json_roundtrip(self):
        lower=np.array([.1],np.float32);upper=np.array([.3],np.float32);truth=np.array([.2],np.float64)
        saved=float(interval_score(truth,lower,upper).mean())
        decoded=json.loads(json.dumps({'lo':lower.tolist(),'hi':upper.tolist()}));lo=np.asarray(decoded['lo']);hi=np.asarray(decoded['hi'])
        self.assertEqual(lo.dtype,np.dtype('float64'))
        self.assertTrue(np.array_equal(lo,lower))
        recomputed=float(interval_score(truth,lo,hi).mean())
        self.assertNotEqual(saved,recomputed)
        tol=score_tolerance(truth,lo,hi)
        self.assertTrue(acceptable(saved,recomputed,tol))
        nlo,nhi=native_bounds(lo,hi)
        self.assertEqual(float(interval_score(truth,nlo,nhi).mean()),saved)
    def test_tolerance_does_not_accept_changed_evidence(self):
        lo=np.array([.1],np.float32).astype(np.float64);hi=np.array([.3],np.float32).astype(np.float64);truth=np.array([.2])
        tol=score_tolerance(truth,lo,hi)
        self.assertFalse(acceptable(1,1+1e-5,tol))
        with self.assertRaises(ValueError):native_bounds(np.array([.1234567890123]),hi)
        with self.assertRaises(ValueError):native_bounds(np.array([np.nan]),hi)
    def test_discrete_gates_remain_exact(self):
        compare_tree({'gates':{'pass':False},'metric':1.},{'gates':{'pass':False},'metric':1.+1e-15})
        with self.assertRaises(ValueError):compare_tree({'pass':False},{'pass':True})
        with self.assertRaises(ValueError):compare_tree({'pass':False},{'pass':0})
        with self.assertRaises(ValueError):compare_tree({'count':624},{'count':623})
    def test_recovery_imports_no_predictor_or_training(self):
        for path in Path('src/mindscape/r5_3_recovery').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    module=node.module or ''
                    self.assertFalse(any(p in module.split('.') for p in ('predict','model','legacy','baselines','training','campaign')),(path,module))

if __name__=='__main__':unittest.main()
