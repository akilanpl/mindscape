"""Exploratory learned residual heads; explicit observation priors, no answer features."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import torch
from torch import nn
from mindscape.r56.neural import R56Model
from mindscape.r56.native_domains import pack_problem
from mindscape.r56.benchmarks import observed_tokens


@dataclass
class Prepared:
    graph: object
    domain: str
    numeric: torch.Tensor
    names: list[str]
    requested: str


class GatedConcept(nn.Embedding):
    def __init__(self,num_embeddings,embedding_dim,off=False):
        super().__init__(num_embeddings,embedding_dim)
        self.gate=nn.Parameter(torch.tensor(0.))
        self.off=off
    def forward(self,ids):
        raw=super().forward(ids)
        normalized=raw / raw.square().mean(-1,keepdim=True).add(1e-6).sqrt()
        return normalized * (self.gate*0 if self.off else torch.tanh(self.gate))


class SequenceCore(nn.Module):
    """Original recurrent observed-token encoder, mean+last access, no rule outputs."""
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(77,64,batch_first=True)
        self.project=nn.Sequential(nn.Linear(128,48),nn.Tanh())
    def forward(self,graph,**kwargs):
        sequence,_=self.gru(observed_tokens(graph)[None])
        values=sequence[0]
        state=self.project(torch.cat((values.mean(0),values[-1])))
        return {'pooled':state,'memory':state}


class NameEncoder(nn.Module):
    """Learned character encoding; no equality flags, hashes, indices or targets."""
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(129,12,padding_idx=0)
        self.gru=nn.GRU(12,16,batch_first=True)
    def forward(self,name,device):
        ids=torch.tensor([[min(ord(c),127)+1 for c in name[:64]]],dtype=torch.long,device=device)
        if not name: ids=torch.zeros((1,1),dtype=torch.long,device=device)
        _,hidden=self.gru(self.embedding(ids));return hidden[0,0]


class R57Candidate(nn.Module):
    def __init__(self,condition):
        super().__init__()
        if condition not in ('graph-concept-on','graph-concept-off','matched-gru'):raise ValueError('Unknown condition')
        self.condition=condition
        if condition=='matched-gru':self.core=SequenceCore()
        else:
            self.core=R56Model()
            old=self.core.concepts
            gated=GatedConcept(32,48,off=condition=='graph-concept-off')
            with torch.no_grad():gated.weight.copy_(old.weight)
            self.core.concepts=gated
        self.math_head=nn.Sequential(nn.Linear(48+24,48),nn.Tanh(),nn.Linear(48,12))
        self.physics_head=nn.Sequential(nn.Linear(48+9,48),nn.Tanh(),nn.Linear(48,2))
        self.name_encoder=NameEncoder()
        self.pointer=nn.Sequential(nn.Linear(48+16*3,48),nn.Tanh(),nn.Linear(48,1))

    def prepare(self,problem):
        allowed={'domain','operation','left','right','incoming','carry_in','borrow_in','concept_ids',
                 'source','old_name','candidates','requested_name','instruction',
                 'mass','force','velocity','impulse','dt'}
        if set(problem)-allowed:raise ValueError('Unrecognized observed fields; labels/targets forbidden')
        p=copy.deepcopy(problem);domain=p['domain'];device=next(self.parameters()).device
        names=list(p.get('candidates',[]));requested=p.get('requested_name','')
        if domain=='coding':
            p['candidates']=sorted(names) # core sees a canonical set; pointer retains original order
        graph=pack_problem(p,device)
        if domain=='math':
            left,right=p['left'],p['right'];incoming=p.get('incoming',p.get('carry_in',p.get('borrow_in',0)))
            # One-hot raw operands, observed incoming and operation only; no arithmetic features.
            numeric=torch.cat((torch.nn.functional.one_hot(torch.tensor(left,device=device),10),
                               torch.nn.functional.one_hot(torch.tensor(right,device=device),10),
                               torch.nn.functional.one_hot(torch.tensor(incoming,device=device),2),
                               torch.tensor([float(p.get('operation')=='add_digit'),float(p.get('operation')=='borrow_digit')],device=device))).float()
        elif domain=='physics':
            values=list(p.get('force',[0,0]))+[p['mass'],p.get('dt',1)]+list(p.get('velocity',[0,0]))+list(p.get('impulse',[0,0]))
            raw=torch.tensor(values,dtype=torch.float32,device=device)
            # Signed log scaling of separate observations; never multiply force*time or divide by mass.
            numeric=torch.cat((raw.sign()*torch.log1p(raw.abs()),torch.ones(1,device=device)))
        else:numeric=torch.empty(0,device=device)
        return Prepared(graph,domain,numeric,names,requested)

    def forward(self,prepared):
        out=self.core(prepared.graph,memory=None,stateless=False)
        state=out['pooled']
        # Original core heads remain measurable; new heads use core state plus direct observations.
        if prepared.domain=='math':
            logits=self.math_head(torch.cat((state,prepared.numeric)))
            out={**out,'math_digit_logits':logits[:10],'math_carry_logits':logits[10:]}
        elif prepared.domain=='physics':
            out={**out,'physics_delta':self.physics_head(torch.cat((state,prepared.numeric)))}
        else:
            device=state.device;query=self.name_encoder(prepared.requested,device)
            scores=[]
            for name in prepared.names:
                candidate=self.name_encoder(name,device)
                scores.append(self.pointer(torch.cat((state,query,candidate,query*candidate))).squeeze(-1))
            out={**out,'coding_logits':torch.stack(scores)}
        return out

    def predict(self,problem):
        return self(self.prepare(problem))


def make_model(condition,seed=17):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return R57Candidate(condition)
