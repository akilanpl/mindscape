"""Explicit ordinal-decimal OUTPUT prior; learned scalar, no input arithmetic solver."""
import torch
from torch import nn
from mindscape.r57.model import R57Candidate


class OrdinalCandidate(R57Candidate):
    def __init__(self,condition):
        super().__init__(condition)
        self.ordinal_weights=nn.Parameter(torch.empty(2,4))
        nn.init.normal_(self.ordinal_weights,mean=0.,std=.1)
        self.register_buffer('ordinal_centers',torch.stack((torch.arange(20.),torch.arange(-10.,10.))))
        # Fixed label geometry, not input-derived answers; rows correspond to existing digit/carry labels.
        self.register_buffer('ordinal_digits',torch.stack((torch.arange(20)%10,torch.arange(-10,10)%10)))
        self.register_buffer('ordinal_carries',torch.stack((torch.arange(20)//10,(torch.arange(-10,10)<0).long())))
    def prepare(self,problem):
        prepared=super().prepare(problem)
        if problem['domain']=='math':
            prepared.ordinal_operation=int(problem['operation']=='borrow_digit')
            prepared.ordinal_raw=torch.tensor([problem['left']/9,problem['right']/9,
                problem.get('incoming',problem.get('carry_in',problem.get('borrow_in',0))),1.],device=next(self.parameters()).device,dtype=torch.float32)
        return prepared
    def forward(self,prepared):
        if prepared.domain!='math':return super().forward(prepared)
        operation=prepared.ordinal_operation
        scalar=(self.ordinal_weights[operation]*prepared.ordinal_raw).sum()
        joint=-(scalar*20-self.ordinal_centers[operation]).square()
        digit=joint.reshape(2,10).logsumexp(0)
        carry=joint.reshape(2,10).logsumexp(1)
        if operation:carry=carry.flip(0)
        return {'math_digit_logits':digit,'math_carry_logits':carry,'math_ordinal_scalar':scalar}


def make_model(condition,seed=17):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return OrdinalCandidate(condition)


def domain_loss(output,row,torch,device):
    if row['problem']['domain']!='math':
        from mindscape.r56.compact_gpu import domain_loss as original
        return original(output,row,torch,device)
    labels=row['labels'];borrow=row['problem']['operation']=='borrow_digit'
    # Bijective recoding of existing supervised labels; no additional ground-truth oracle.
    target=(labels['math_digit']+(-10 if borrow else 10)*labels['math_carry'])/20
    return torch.nn.functional.mse_loss(output['math_ordinal_scalar'],torch.tensor(target,device=device,dtype=torch.float32))
