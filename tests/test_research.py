import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Optional learning dependencies')
class ResearchTests(unittest.TestCase):
    def setUp(self):
        from mindscape.data.generation import generate
        self.cfg={'environment':'integer_multiplication_claims','seed':4242,'budgets':[5],
            'splits':{'train':{'count':10,'structures':['2x1']},
                      'validation':{'count':3,'structures':['2x1']},
                      'test':{'count':3,'structures':['2x1']},
                      'ood_test':{'count':3,'structures':['3x2']}}}
        self.splits=generate(self.cfg)

    def test_claim_arithmetic_oracle_cases(self):
        from mindscape.environments.multiplication.claims import ClaimsEnvironment,verify_claims
        rng=np.random.default_rng(42)
        for a,b in [(19,3),(99,99),(0,42),(-12,13),(-19,-3)]+[tuple(map(int,rng.integers(-9999,9999,2))) for _ in range(100)]:
            env=ClaimsEnvironment();env.reset((a,b));t=env.reference()
            self.assertEqual(env.state.answer,a*b)
            self.assertEqual(verify_claims(t,a*b),(True,True))

    def test_wrong_claim_is_not_corrected(self):
        from mindscape.environments.multiplication.claims import ClaimsEnvironment,ClaimAction,verify_claims
        env=ClaimsEnvironment();env.reset((19,3))
        transition=env.step(ClaimAction(0,0,0))
        self.assertEqual(transition.result.value,0)
        self.assertEqual(env.state.accumulated,0)
        self.assertEqual(verify_claims(env.trajectory,None),(False,False))
        self.assertNotIn('expected',asdict(transition.result))

    def test_candidates_are_not_singletons(self):
        from mindscape.environments.multiplication.claims import ClaimsEnvironment
        env=ClaimsEnvironment();env.reset((19,3))
        self.assertEqual(len(env.valid_actions()),100)
        self.assertEqual({a.value for a in env.valid_actions()},set(range(100)))

    def test_future_target_is_not_a_feature(self):
        from mindscape.models.study_encoding import answer_features,policy_features
        from mindscape.core.schema import State
        from mindscape.core.serialization import decode
        e=self.splits['train'][0]
        x=policy_features(decode(State,e.initial_state))
        poisoned=replace(e,target_answer=-999999,target_trajectory=[])
        np.testing.assert_array_equal(answer_features(e.view('structured')),answer_features(poisoned.view('structured')))
        np.testing.assert_array_equal(x,policy_features(decode(State,poisoned.initial_state)))

    def test_identical_parameter_capacity(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.study import StudyModel
        sizes=[StudyModel(NumpyMLP([10]*8+[2]),c).parameter_count for c in ['answer_only','structured','trajectory','experiential']]
        self.assertEqual(set(sizes),{8722})

    def test_policy_record_three_evidence_kinds(self):
        from mindscape.models.study import StudyModel
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.evaluation.evaluator import evaluate
        from mindscape.data.backends import get_backend
        m=StudyModel(NumpyMLP([10]*8+[2]),'trajectory',constrained=True,dream=False)
        p=m.predict(self.splits['test'][0].view('trajectory_supervised'))
        evaluate(self.splits['test'][0],p,get_backend(self.cfg['environment']))
        d=p.diagnostics['decisions'][0]
        self.assertEqual(d['candidate_action_count'],100)
        self.assertIn('model_prediction',d);self.assertIn('actual_result',d);self.assertIn('verifier_judgment',d)
        self.assertFalse(p.diagnostics['oracle_feedback_visible'])

    def test_c_has_no_target_trajectory_access(self):
        from mindscape.models.study import StudyModel
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.training.claims import collect_experience
        from mindscape.memory.episodic import EpisodicMemory
        poisoned=[replace(e,target_answer='forbidden',target_trajectory=[{'forbidden':True}]) for e in self.splits['train'][:2]]
        with tempfile.TemporaryDirectory() as tmp:
            model=StudyModel(NumpyMLP([10]*8+[2]),'experiential',dream=False)
            x,y,report=collect_experience(poisoned,model,EpisodicMemory(Path(tmp)/'memory.sqlite'),0,20)
            self.assertEqual(report['full_target_trajectories_received'],0)
            self.assertEqual(report['attempts'],40)
            self.assertEqual(len(x),len(y))

    def test_dream_does_not_change_real_state(self):
        from mindscape.reasoning.simulator import DreamEngine,FunctionalTransitionModel
        from mindscape.environments.multiplication.claims import ClaimsEnvironment,ClaimAction,apply_claim
        env=ClaimsEnvironment();original=env.reset((19,3))
        engine=DreamEngine(FunctionalTransitionModel(apply_claim),lambda s:[ClaimAction(i,s.position,s.row_position) for i in [0,1]],lambda s,a,n:-a.value,2,2)
        paths=engine.rollouts(original)
        self.assertEqual(len(paths),4)
        self.assertTrue(all(s.hypothetical for s,_,_ in paths))
        self.assertEqual(env.state,original)
        self.assertEqual(env.trajectory.transitions,())

    def test_generic_simulator_and_learned_adapter(self):
        from mindscape.reasoning.simulator import DreamEngine,LearnedTransitionAdapter
        class Learned:
            def predict_state(self,state,action):return state+action
        engine=DreamEngine(LearnedTransitionAdapter(Learned()),lambda s:[1,2],lambda s,a,n:n,2,2)
        self.assertEqual(engine.select(0),2)
        self.assertTrue(all(s.hypothetical for s,_,_ in engine.rollouts(0)))

    def test_constrained_and_unconstrained_counts(self):
        from mindscape.models.study import StudyModel
        from mindscape.models.numpy_backend import NumpyMLP
        view=self.splits['test'][0].view('trajectory_supervised')
        for mode,count in [(True,100),(False,101)]:
            p=StudyModel(NumpyMLP([10]*8+[2]),'trajectory',constrained=mode,dream=False).predict(view)
            self.assertEqual(p.diagnostics['decisions'][0]['candidate_action_count'],count)

    def test_no_interaction_has_no_evidence(self):
        from mindscape.models.study import StudyModel
        from mindscape.models.numpy_backend import NumpyMLP
        p=StudyModel(NumpyMLP([10]*8+[2]),'trajectory',constrained=True,no_interaction=True,dream=False).predict(self.splits['test'][0].view('trajectory_supervised'))
        self.assertIsNone(p.trajectory)
        self.assertIn('hypothetical_result',p.diagnostics['decisions'][0])

    def test_perturbation_invariance(self):
        from mindscape.models.study import StudyModel
        from mindscape.models.numpy_backend import NumpyMLP
        for condition in ['answer_only','structured','trajectory']:
            m=StudyModel(NumpyMLP([10]*8+[2]),condition,constrained=True,dream=False)
            view=self.splits['test'][0].view('structured')
            a=m.predict(view);m.perturb=True;b=m.predict(view)
            self.assertEqual(a.answer,b.answer)
            self.assertEqual(a.trajectory,b.trajectory)

    def test_train_and_reload_all_conditions(self):
        from mindscape.data.generation import save_dataset,load_dataset
        from mindscape.training.claims import train_claims
        from mindscape.models.study import StudyModel
        from mindscape.evaluation.runner import run
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';save_dataset(data,self.splits,self.cfg)
            splits,manifest=load_dataset(data)
            for c in ['answer_only','structured','trajectory','experiential']:
                config=dict(condition=c,seed=0,budget=5,hidden=8,steps=5,batch_size=8,learning_rate=.01,validation_every=5,attempts_per_problem=10,dream=False)
                m,meta=train_claims(splits,manifest,config,Path(tmp)/c)
                restored=StudyModel.load(Path(tmp)/c/'checkpoint')
                self.assertEqual(m.parameter_count,restored.parameter_count)
                folder,result=run(restored,data,Path(tmp)/'eval',regime=meta['regime'])
                self.assertTrue((folder/'config.yaml').exists())
                self.assertTrue((folder/'predictions.jsonl').exists())
                if c=='experiential':self.assertEqual(meta['experience']['full_target_trajectories_received'],0)

    def test_pretrained_backend_offline_contract(self):
        from unittest.mock import patch
        from mindscape.models.pretrained import FrozenPretrainedBackend
        class FakeEncoder:
            parameter_count=100
            def __init__(self,path,allow_download=False):
                if allow_download:raise AssertionError('Unexpected download')
            def encode(self,texts):return np.ones((len(texts),16))
        with patch('mindscape.models.pretrained.LocalHFEncoder',FakeEncoder):
            backend=FrozenPretrainedBackend('local-cache',[2],8,0)
            self.assertEqual(backend.logits(['test']).shape,(1,2))
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'checkpoint';backend.save(path)
                restored=FrozenPretrainedBackend.load(path)
                np.testing.assert_array_equal(backend.logits(['test']),restored.logits(['test']))

    def test_completion_at_final_attempt_is_counted(self):
        from unittest.mock import patch
        from mindscape.data.backends import get_backend
        from mindscape.models.study import StudyModel
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.memory.episodic import EpisodicMemory
        from mindscape.training.claims import collect_experience
        example=get_backend('integer_multiplication_claims').example({'operands':[0,1],'source':'procedural_generator','kind':'observation'},42,'train')
        class ExploitOnly:
            def random(self):return 1.0
        model=StudyModel(NumpyMLP([10]*8+[2]),'experiential')
        scores=np.zeros(101);scores[0]=10
        with tempfile.TemporaryDirectory() as tmp:
            with patch('mindscape.training.claims.np.random.default_rng',return_value=ExploitOnly()),patch('mindscape.training.claims.candidate_scores',return_value=scores):
                x,y,report=collect_experience([example],model,EpisodicMemory(Path(tmp)/'memory.sqlite'),0,2)
        self.assertEqual(report['completed_episodes'],1)

    def test_statistical_utilities(self):
        import importlib.util
        if importlib.util.find_spec('reportlab') is None:self.skipTest('Optional plotting dependency')
        from scripts.analyze_research_study import wilson,mcnemar
        low,high=wilson(0,200)
        self.assertAlmostEqual(low,0)
        self.assertGreater(high,0)
        self.assertIsNone(wilson(0,0))
        self.assertEqual(mcnemar([True,False],[False,True])['p_value'],1)
        self.assertEqual(mcnemar([False,False],[True,True])['p_value'],.5)

    def test_invalid_study_config_fails_before_writing(self):
        from mindscape.data.generation import save_dataset,load_dataset
        from mindscape.training.claims import train_claims
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';save_dataset(data,self.splits,self.cfg)
            splits,manifest=load_dataset(data)
            cfg=dict(condition='trajectory',seed=0,budget=5,hidden=8,steps=0,batch_size=8,learning_rate=.01,validation_every=5,attempts_per_problem=10,dream=False)
            output=Path(tmp)/'bad'
            with self.assertRaises(ValueError):train_claims(splits,manifest,cfg,output)
            self.assertFalse(output.exists())
