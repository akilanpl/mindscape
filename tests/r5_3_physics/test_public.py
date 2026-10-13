import ast,sys,unittest
from pathlib import Path
import numpy as np
from mindscape.r5_3_physics.public import request,validate
from mindscape.r5_3_physics.uncertainty import propagate,kinematic_matrices,terminal_risk,interval_score
from mindscape.r5_3_physics.baselines import forecast
from mindscape.v3.physics_world.schema import ObjectSpec,Scene,observe,Action
from mindscape.v3.physics_world.environment import Environment

class PhysicsTests(unittest.TestCase):
    def test_covariance_analytical_and_monte_carlo(self):
        f,b=kinematic_matrices(.1);p=np.zeros((4,4));q=np.eye(2)*.2
        for _ in range(16):p=propagate(p,f,b,q)
        self.assertAlmostEqual(p[2,2],16*.2)
        self.assertAlmostEqual(p[0,2],.1*.2*sum(range(1,17)))
        self.assertAlmostEqual(p[0,0],.01*.2*sum(i*i for i in range(1,17)))
        rng=np.random.default_rng(13);x=np.zeros((100000,4))
        for _ in range(16):x=x@f.T+rng.normal(0,np.sqrt(.2),(len(x),2))@b.T
        sampling_se=np.sqrt((p*p+np.outer(np.diag(p),np.diag(p)))/len(x))
        self.assertTrue(np.all(abs(np.cov(x,rowvar=False)-p)<5*sampling_se))
        risk=terminal_risk([0,0,0,0],p,np.array([1,1]))
        self.assertAlmostEqual(risk,np.mean(np.sum((x[:,:2]-[1,1])**2,axis=1)),delta=.06)
    def test_interval_score_penalizes_width_and_misses(self):
        self.assertGreater(interval_score(np.array([0]),np.array([-10]),np.array([10]))[0],interval_score(np.array([0]),np.array([-1]),np.array([1]))[0])
        self.assertGreater(interval_score(np.array([3]),np.array([-1]),np.array([1]))[0],2)
    def fixture(self):
        h=np.zeros((4,3,10),np.float32);h[:,0]=[0,5,1,0,1,.25,.2,.5,0,1];m=np.ones_like(h);m[:,1:]=0
        return request(h,m,np.zeros((8,3,4),np.float32),np.tile([9,1/30,0,0],(8,1)).astype(np.float32))
    def test_private_rejection_and_mask(self):
        r=self.fixture();validate(r);r['future']=1
        with self.assertRaises(ValueError):validate(r)
        del r['future'];r['masks'][-1,0,4]=0
        with self.assertRaises(ValueError):validate(r)
    def test_import_boundary_transitively(self):
        queue=['mindscape.r5_3_physics.'+n for n in ('predict','model','public','uncertainty','baselines','legacy')];seen=set()
        while queue:
            module=queue.pop()
            if module in seen:continue
            seen.add(module);path=Path('src')/Path(*module.split('.')).with_suffix('.py')
            if not path.exists():continue
            for n in ast.walk(ast.parse(path.read_text())):
                dependencies=[]
                if isinstance(n,ast.Import):dependencies=[a.name for a in n.names]
                elif isinstance(n,ast.ImportFrom):
                    base=module.split('.')[:-n.level] if n.level else []
                    dependencies=['.'.join(base+([n.module] if n.module else []))]
                for dependency in dependencies:
                    self.assertFalse(any(x in dependency.split('.') for x in ('training','environment','campaign','data','verification','audit','science')),(module,dependency))
                    if dependency.startswith('mindscape.'):queue.append(dependency)
    def test_freeflight_against_pymunk(self):
        r=self.fixture();prediction=forecast(r,'analytical_available')
        scene=Scene('motion','fixture',0,55,'projectile',(ObjectSpec((0,5),(1,0),1,.25,.2,.5),),9,0,False,False,0,8)
        env=Environment(scene);actual=np.stack([env.step(Action.from_vector(np.zeros((3,4)))) for _ in range(8)])
        np.testing.assert_allclose(prediction['states'][:,0,2:],actual[:,0,2:4],atol=1e-5)
        # Pymunk explicit substeps differ from exact continuous position by O(dt^2).
        np.testing.assert_allclose(prediction['states'][:,0,:2],actual[:,0,:2],atol=.012)
        self.assertTrue(np.all(np.linalg.eigvalsh(prediction['covariance'])>=-1e-10))

if __name__=='__main__':unittest.main()

class ScientificTests(unittest.TestCase):
    def test_serialization_preserves_metrics(self):
        import json
        from mindscape.r5_3_physics.science import metrics,selection
        h=PhysicsTests().fixture();p=forecast(h,'analytical_available')
        truth=np.zeros((8,3,10),np.float32);truth[:,:,:4]=p['states'];truth[:,:,9]=h['history'][-1,:,9]
        a=metrics(p,truth,h['history'][-1,:,9],h['history'][-1,:,5])
        stored=json.loads(json.dumps({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()}))
        self.assertEqual(a,metrics(stored,truth,h['history'][-1,:,9],h['history'][-1,:,5]))
        self.assertEqual(selection([p],np.array([1.5,1.])),selection([stored],np.array([1.5,1.])))
        stored['covariance'][0][0][0]=-1
        with self.assertRaises(ValueError):metrics(stored,truth,h['history'][-1,:,9],h['history'][-1,:,5])
    def test_finite_outputs_and_wide_intervals_are_not_go(self):
        import json
        from mindscape.r5_3_physics.science import analyze
        protocol=json.loads(Path('configs/r5_3_physics/protocol.json').read_text());rows=[]
        for family in ('motion','interaction'):
            for split in ('iid','ood'):
                for i in range(2):
                    for method in protocol['baseline_methods']+['structured']:
                        for seed in (11,23,47) if method in ('structured','matched') else (0,):
                            for h in protocol['horizons']:
                                rows.append(dict(identity=f'{family}-{split}-{i}',family=family,split=split,stratum='configuration',method=method,seed=seed,horizon=h,mse=1.,velocity_mse=1.,coverage=1.,interval_score=100.,contact_accuracy=1.,floor_penetration=0.,success=0.))
        result=analyze(rows,protocol)
        self.assertEqual(result['decision'],'NO_GO')
        self.assertFalse(result['gates']['motion']['coverage'])
        self.assertFalse(result['gates']['motion']['planning'])
        self.assertEqual(result['metrics']['motion']['cells']['ood/32/structured']['independent_scenes'],2)
