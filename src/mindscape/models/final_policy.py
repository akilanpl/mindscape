"""Procedural claims with separate real/hypothetical evidence and posthoc judgment."""
from dataclasses import asdict
import numpy as np
from mindscape.models.study import StudyModel
from mindscape.models.base import Prediction
from mindscape.models.study_encoding import policy_features,answer_features
from mindscape.models.encoding import decode_answer
from mindscape.core.serialization import decode
from mindscape.core.schema import Observation
from mindscape.environments.multiplication.claims import ClaimsEnvironment,ClaimAction,apply_claim
from mindscape.reasoning.simulator import DreamEngine,FunctionalTransitionModel
from mindscape.memory.concept import ConceptMemory

class FinalPolicy(StudyModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.use_memory=True
        self.identifier='final_smollm2_'+self.condition
        self.concepts=ConceptMemory();self.concepts.store('decimal_claims','A claim writes units and carry at the active cursor. Numerical claims are learned.')
    def predict(self,view):
        if self.condition in ('answer_only','structured'):
            labels=self.backend.generate(answer_features(view,self.condition=='structured')[None,:])[0]
            return Prediction(decode_answer(labels),model_calls=1)
        env=ClaimsEnvironment();obs=decode(Observation,view['observation']);env.reset(obs)
        if self.use_memory:self.memory.reset(obs,env.state,env.get_goal())
        decisions=[];calls=0
        while env.state.phase!='done':
            if len(decisions)>=256:return Prediction(error_type='timeout',model_calls=calls)
            cache={}
            def scores(state):
                nonlocal calls
                key=str(state)
                if key not in cache:
                    logits=self.backend.logits(policy_features(state,self.no_state,self.no_relation,self.no_goal)[None,:])[0]
                    cache[key]=np.array([logits[v%10]+logits[10+v//10] for v in range(100)])
                    calls+=1
                return cache[key]
            value=int(scores(env.state).argmax());rollout=[]
            if self.dream:
                def candidates(s):
                    return [] if s.phase=='done' else [ClaimAction(int(v),s.position,s.row_position) for v in np.argsort(-scores(s))[:3]]
                def score(s,a,n):
                    z=scores(s);z=z-z.max();return float(z[a.value]-np.log(np.exp(z).sum()))
                paths=DreamEngine(FunctionalTransitionModel(apply_claim),candidates,score,2,3).rollouts(env.state)
                best=max(paths,key=lambda p:p[2]);value=best[1][0].value
                rollout=[{'hypothetical':True,'state':asdict(best[0].value),'actions':[asdict(a) for a in best[1]],'score':best[2]}]
            action=ClaimAction(value,env.state.position,env.state.row_position)
            d={'total_candidate_actions':100,'candidate_action_count':100,'valid_actions':list(range(100)),
               'chosen_action':asdict(action),'selected_value':value,'model_prediction':value,'action_valid':True,
               'hypothetical':False,'state_before':asdict(env.state),'hypothetical_rollouts':rollout}
            if self.no_interaction:
                env.state=apply_claim(env.state,action);d['hypothetical_result']={'value':value,'hypothetical':True}
                d['hypothetical']=True
            else:
                t=env.step(action)
                if self.use_memory:self.memory.update(t)
                d['actual_result']={**asdict(t.result),'hypothetical':False}
                d['event']=asdict(t.event)
            d['state_after']=asdict(env.state);decisions.append(d)
        return Prediction(env.state.answer,None if self.no_interaction else asdict(env.trajectory),model_calls=calls,
            diagnostics={'decisions':decisions,'oracle_feedback_visible':False,'execution_mode':'100_open_numerical_candidates','goal':asdict(env.get_goal()),'real_state_hypothetical':False})
