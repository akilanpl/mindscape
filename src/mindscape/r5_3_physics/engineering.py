"""Native gradient and conservation checks; no optimizer steps."""
import torch
from .model import World
from mindscape.v3.physics_world.accelerator import verify_placement

def check():
    for kind in ('structured','matched'):
        model=World(7,kind)
        state=torch.tensor([[[-1.,5.,1.,0.],[1.,5.,-1.,0.],[0.,0.,0.,0.]]],device='mps')
        attrs=torch.tensor([[[1.,.25,.2,.5,1.],[1.3,.25,.2,.5,1.],[1.,0.,0.,0.,0.]]],device='mps');action=torch.zeros((1,3,4),device='mps');context=torch.tensor([[0.,1/30,0.,0.]],device='mps')
        initial=state.clone()
        for _ in range(4):state,std,contact=model(state,attrs,action,context)
        loss=state.square().mean()+std.square().mean();loss.backward();verify_placement(model,gradients=True)
        if not all(torch.isfinite(v.grad).all().item() for v in model.parameters()):raise RuntimeError('Gradient nonfinite')
        if kind=='structured':
            before=(initial[...,2:]*attrs[...,0,None]*attrs[...,4,None]).sum(1)
            after=(state[...,2:]*attrs[...,0,None]*attrs[...,4,None]).sum(1)
            if not torch.allclose(before,after,atol=1e-4):raise RuntimeError('Pair momentum constraint')
            order=torch.tensor([1,0,2],device='mps');x1,_,_=model(initial,attrs,action,context);x2,_,_=model(initial[:,order],attrs[:,order],action[:,order],context)
            if not torch.allclose(x1[:,order],x2,atol=1e-4):raise RuntimeError('Object permutation constraint')
    return dict(status='PASS',device='mps',multistep_gradients=True,internal_pair_momentum=True,permutation_without_indexed_spring=True,optimizer_steps=0)
