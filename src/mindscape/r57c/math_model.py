"""Exploratory learned scalar arithmetic with explicitly declared decoder prior."""
import torch
from torch import nn
from mindscape.r57.model import R57Candidate


class ScalarArithmetic(nn.Module):
    """Learned operation-specific affine latent; fixed periodic digit lattice prior.

    No modulo/floor/reference solver occurs. Raw operands feed trainable weights.
    Digit lattice explicitly encodes decimal periodicity and is not prior-free.
    Carry is a separately learned binary linear head of the latent.
    """
    def __init__(self,prior='lattice'):
        super().__init__();self.prior=prior
        self.weights=nn.Parameter(torch.empty(2,3));self.bias=nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.weights,mean=.5,std=.1)
        self.carry=nn.Linear(1,2)
        if prior=='mlp':self.decoder=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,10))
        elif prior!='lattice':raise ValueError('Unknown prior')
        self.log_precision=nn.Parameter(torch.tensor(-1.))
        self.register_buffer('centers',torch.arange(10.).unsqueeze(1)+10*torch.arange(-3.,4.).unsqueeze(0))
    def forward(self,raw):
        operation=raw[3].long();value=(self.weights[operation]*raw[:3]).sum()+self.bias[operation]
        if self.prior=='mlp':digit=self.decoder(value.reshape(1))
        else:digit=torch.logsumexp(-self.log_precision.exp()*(value-self.centers).square(),dim=1)
        carry=self.carry(value.reshape(1))
        return digit,carry


class MathCorrected(R57Candidate):
    def __init__(self,condition,prior='lattice'):
        super().__init__(condition)
        self.scalar_arithmetic=ScalarArithmetic(prior)
    def prepare(self,problem):
        prepared=super().prepare(problem)
        if problem['domain']=='math':
            prepared.raw_math=torch.tensor([problem['left'],problem['right'],problem.get('incoming',problem.get('carry_in',problem.get('borrow_in',0))),float(problem['operation']=='borrow_digit')],device=next(self.parameters()).device,dtype=torch.float32)
        return prepared
    def forward(self,prepared):
        if prepared.domain!='math':return super().forward(prepared)
        digit,carry=self.scalar_arithmetic(prepared.raw_math)
        return {'math_digit_logits':digit,'math_carry_logits':carry}


def make_model(condition,seed=17,prior='lattice'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return MathCorrected(condition,prior)
