"""Supplemental identifiability probes. No primary-grid changes or fitted claims."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import random
import torch
from torch import nn

RULES = ((1,0),(1,3),(2,0),(2,3))  # independently supplied relation-class semantics
STATUS = 'SUPPLEMENTAL PROTOCOL ONLY; NO TRAINING OR QUALIFICATION'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def supplemental_protocol():
    protocol={'version':'r56-identifiability-supplement-v1','status':STATUS,
              'primary_grid_modified':False,'primary_fits':72,'primary_optimizer_steps':144000,
              'concept_controls':['embedding','context-vector','omitted','independent-corruption'],
              'memory_controls':['retain','reset','shuffled-past'],
              'delayed_cue_heldout_delays':[0,1,4,8],
              'delayed_cue_split':'disjoint cue/query combinations; longer delays4/8 held out from training',
              'implementation_scope':'standalone original small probes; not original-R56 graph qualification',
              'claim_scope':'supplied relational context and delayed-cue recoverability; no discovered concepts',
              'metrics':['exact-class-count','paired-counterfactual-count','delayed-cue-query-count'],
              'selection':'fixed future budget; no held-out checkpoint selection',
              'authorization':'UNISSUED','qualification':'UNISSUED'}
    return {**protocol,'protocol_sha256':_digest(protocol)}


def concept_curriculum(seed=17, split='train'):
    """All relation labels occur with every state/action; action/domain never identify rule."""
    if split not in ('train','heldout'): raise ValueError('Unknown split')
    pairs=[(a,b) for a in range(10) for b in range(10) if ((a+3*b)%5==0)==(split=='heldout')]
    rows=[]
    rng=random.Random(seed)
    # Corruptions are independent categorical draws, not a stable ID permutation.
    for left,right in pairs:
        for concept,(scale,offset) in enumerate(RULES):
            observations={'state':[left,right],'action':[1.,0.,0.,0.],'domain':0,'concept_id':concept}
            rows.append({'observations':observations,'labels':{'class':(scale*(left+right)+offset)%10},
                         'corrupted_context_id':rng.randrange(len(RULES)),
                         'split':split,'identity':f'{split}:{left}:{right}:{concept}'})
    return rows


@dataclass
class ConceptInput:
    state: torch.Tensor
    action: torch.Tensor
    domain: torch.Tensor
    context: torch.Tensor


def pack_concept(observations, *, device='cpu', context_id=None, omitted=False):
    """Allowlisted observed fields only; labels, corruption records and outcomes rejected."""
    if set(observations)!={'state','action','domain','concept_id'}: raise ValueError('Only allowlisted observations accepted')
    state=observations['state']; action=observations['action']; concept=observations['concept_id'] if context_id is None else context_id
    if len(state)!=2 or any(type(v)is not int or not 0<=v<=9 for v in state): raise ValueError('Two observed decimal digits required')
    if action!=[1.,0.,0.,0.] or observations['domain']!=0: raise ValueError('Supplement uses one matched action/domain')
    if type(concept)is not int or not 0<=concept<4: raise ValueError('Supplied context class 0..3 required')
    context=torch.zeros(4,device=device)
    if not omitted: context[concept]=1
    return ConceptInput(torch.tensor(state,dtype=torch.float32,device=device)/9,
                        torch.tensor(action,dtype=torch.float32,device=device),
                        torch.tensor([1.],device=device),context)


class ConceptProbe(nn.Module):
    """Same parameters for embedding and one-hot controls; identical supplied information.

    One-hot multiplication and embedding indexing are mathematically equivalent.
    Their equality is an implementation control, not an expected performance gain.
    """
    def __init__(self,width=24):
        super().__init__()
        self.context=nn.Embedding(4,width)
        self.observed=nn.Linear(7,width)
        self.core=nn.Sequential(nn.Linear(width*2,width),nn.Tanh(),nn.Linear(width,10))

    def forward(self,inputs:ConceptInput,*,representation='embedding'):
        if representation not in ('embedding','context-vector'): raise ValueError('Unknown matched representation')
        if inputs.context.shape!=(4,): raise ValueError('Four-dimensional supplied context required')
        # Soft/omitted contexts retain exactly the same matrix path for both controls.
        if representation=='embedding' and bool(inputs.context.sum()==1) and int((inputs.context!=0).sum())==1:
            context=self.context(inputs.context.argmax())
        else: context=inputs.context @ self.context.weight
        observed=torch.tanh(self.observed(torch.cat((inputs.state,inputs.action,inputs.domain))))
        return self.core(torch.cat((observed,context)))


def delayed_cue_curriculum():
    """Paired episodes: identical query observations, different required cue-dependent classes."""
    episodes=[]
    for cue in range(4):
        for query in range(4):
            observations=[{'phase':'cue','value':cue},{'phase':'delay','value':0},
                          {'phase':'query','value':query}]
            episodes.append({'observations':observations,'labels':{'query_class':(cue+query)%4},
                             'identity':f'cue{cue}:query{query}'})
    return episodes


def pack_memory(observation,device='cpu'):
    if set(observation)!={'phase','value'}: raise ValueError('Only phase and observed value accepted')
    phase=observation['phase']; value=observation['value']
    if phase not in ('cue','delay','query') or type(value)is not int or not 0<=value<4: raise ValueError('Invalid observation')
    vector=torch.zeros(7,device=device); vector[('cue','delay','query').index(phase)]=1; vector[3+value]=1
    return vector


class DelayedCueProbe(nn.Module):
    """Every memory control executes the exact same encoder/GRU/fusion/head parameters."""
    def __init__(self,width=24):
        super().__init__(); self.width=width
        self.encoder=nn.Linear(7,width); self.memory=nn.GRUCell(width,width)
        self.fusion=nn.Sequential(nn.Linear(width*2,width),nn.Tanh()); self.head=nn.Linear(width,4)

    def step(self,observation,memory=None,*,control='retain',shuffled_past=None):
        if control not in ('retain','reset','shuffled-past'): raise ValueError('Unknown memory control')
        encoded=torch.tanh(self.encoder(observation))
        prior=torch.zeros_like(encoded) if memory is None else memory
        if control=='reset': prior=torch.zeros_like(encoded)
        if control=='shuffled-past':
            if shuffled_past is None: raise ValueError('Independent past state required for shuffled control')
            if shuffled_past.shape!=prior.shape: raise ValueError('Shuffled state shape differs')
            prior=shuffled_past.detach()
        if prior.shape!=(self.width,) or not torch.isfinite(prior).all(): raise ValueError('Invalid bounded memory state')
        updated=torch.tanh(self.memory(encoded.unsqueeze(0),prior.unsqueeze(0)).squeeze(0))
        return {'logits':self.head(self.fusion(torch.cat((encoded,updated)))), 'memory':updated}


def run_memory_episode(model,observations,*,control='retain',shuffled_past=None):
    """New episode always starts empty; supplied replay past is detached, never a target.

    Returns original predictions and state snapshots. No correction or answer feed
    occurs; the original prior hidden vector, rather than a verifier, is fed forward.
    """
    memory=None; records=[]
    device=next(model.parameters()).device
    if not isinstance(observations,list) or not 1<=len(observations)<=32: raise ValueError('Require 1..32 bounded episode observations')
    for i,observation in enumerate(observations):
        past=None if shuffled_past is None else shuffled_past[i]
        result=model.step(pack_memory(observation,device),memory,control=control,shuffled_past=past)
        memory=result['memory']
        records.append({'prediction':result['logits'].argmax().item(),'logits':result['logits'],
                        'memory':memory,'source':'original uncorrected model output',
                        'trace':{'episode_start':i==0,'incoming_state_reset':i==0 or control=='reset',
                                 'donor_detached':control=='shuffled-past',
                                 'within_episode_graph_retained':control=='retain' and i>0,
                                 'hidden_feedback':'zero at episode boundary' if i==0 and control!='shuffled-past' else (
                                     'zero; same GRU/fusion path' if control=='reset' else (
                                     'independent donor hidden; detached across episodes' if control=='shuffled-past' else
                                     'original prior hidden; same episode graph; no target correction'))}})
    return records


def conditional_information_audit(rows):
    """Empirical optimal-count audit, independent of model logits or fitting.

    Majority counts upper-bound deterministic classification on this finite
    empirical table, not on an unknown population. Conflicting labels prove
    nonidentifiability from the grouped observed fields for those counterexamples.
    """
    import math
    from collections import Counter,defaultdict
    groups_without=defaultdict(Counter); groups_with=defaultdict(Counter)
    for row in rows:
        observations=row['observations']
        pack_concept(observations)  # same strict input boundary, no label features
        label=row['labels']['class']
        if type(label)is not int or not 0<=label<10: raise ValueError('Audit requires protected class labels 0..9')
        base=(tuple(observations['state']),tuple(observations['action']),observations['domain'])
        groups_without[base][label]+=1
        groups_with[(base,observations['concept_id'])][label]+=1
    if not rows: raise ValueError('Audit requires nonempty observations')
    count=len(rows)
    def summarize(groups):
        correct=sum(max(histogram.values()) for histogram in groups.values())
        entropy=sum(-frequency/count*math.log2(frequency/sum(histogram.values()))
                    for histogram in groups.values() for frequency in histogram.values())
        conflicts=sum(len(histogram)>1 for histogram in groups.values())
        return {'groups':len(groups),'empirical_bayes_correct_count':correct,
                'empirical_bayes_accuracy':correct/count,'conditional_label_entropy_bits':entropy,
                'conflicting_observation_groups':conflicts,'identifiable_on_empirical_support':conflicts==0}
    without=summarize(groups_without); with_context=summarize(groups_with)
    return {'examples':count,'without_context':without,'with_context':with_context,
            'empirical_conditional_information_bits':without['conditional_label_entropy_bits']-with_context['conditional_label_entropy_bits'],
            'scope':'finite supplied dataset; empirical counts only, no learned or population generalization claim'}


def delayed_cue_variants(split='train',delay=None):
    """Supplement-only held-out combinations and delays; legacy default unchanged."""
    if split not in ('train','heldout'): raise ValueError('Unknown delayed-cue split')
    delays=(0,1) if split=='train' else (0,1,4,8)
    if delay is not None:
        if type(delay)is not int or delay not in (0,1,4,8): raise ValueError('Allowed delays: 0,1,4,8')
        delays=(delay,)
    rows=[]
    for cue in range(4):
        for query in range(4):
            if ((cue+2*query)%4==0)!=(split=='heldout'): continue
            for length in delays:
                observations=[{'phase':'cue','value':cue}]+[{'phase':'delay','value':0} for _ in range(length)]+[{'phase':'query','value':query}]
                rows.append({'observations':observations,'labels':{'query_class':(cue+query)%4},
                             'split':split,'cue_query_identity':[cue,query],'delay':length,
                             'identity':f'{split}:cue{cue}:query{query}:delay{length}'})
    return rows
