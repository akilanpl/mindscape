"""Object-relative paired impulse residuals; four public integration substeps."""
import torch
from torch import nn
from mindscape.v3.physics_world.accelerator import require_mps,verify_placement

class World(nn.Module):
    def __init__(self,seed,kind):
        device=require_mps(); torch.manual_seed(seed)
        with torch.device(device):
            super().__init__()
            self.edge=nn.Sequential(nn.Linear(10,24),nn.SiLU(),nn.Linear(24,3))
            self.node=nn.Sequential(nn.Linear(10,24),nn.SiLU(),nn.Linear(24,4))
        self.kind=kind
    def forward(self,state,attrs,action,context):
        verify_placement(self,state,attrs,action,context)
        mass=attrs[...,0].clamp(.05,5); radius=attrs[...,1]; friction=attrs[...,2]; restitution=attrs[...,3]; active=attrs[...,4]
        x=state[...,:2]; v=state[...,2:]+action[...,2:]/mass[...,None]
        force=action[...,:2].clone(); n=3; dt=1/30
        nodes=torch.cat((state/6,attrs,context[:,None,:1].expand(-1,n,-1)/15),-1)
        raw=self.node(nodes)
        # Shared scalar radial/dissipative forces are antisymmetric; external floor/gravity need not conserve momentum.
        for i in range(n):
            for j in range(i+1,n):
                delta=x[:,j]-x[:,i]; distance=delta.norm(dim=-1).clamp(min=1e-5); normal=delta/distance[:,None]
                relative=v[:,j]-v[:,i]; gap=distance-radius[:,i]-radius[:,j]
                features=torch.cat((delta/3,relative/6,gap[:,None],mass[:,i,None]/5,mass[:,j,None]/5,radius[:,i,None],radius[:,j,None],context[:,2,None]/10),-1)
                # Symmetrize learned radial coefficients under swapping object identities.
                reverse=torch.cat((-delta/3,-relative/6,gap[:,None],mass[:,j,None]/5,mass[:,i,None]/5,radius[:,j,None],radius[:,i,None],context[:,2,None]/10),-1)
                coeff=(self.edge(features)+self.edge(reverse))/2
                spring=context[:,2]*(distance-1.5)+.3*(relative*normal).sum(-1) if i==0 and j==1 else torch.zeros_like(distance)
                radial=(10*torch.tanh(coeff[:,0])+torch.nn.functional.softplus(coeff[:,1])*(relative*normal).sum(-1)).clamp(-15,15)
                correction=radial if self.kind=='structured' else 0*radial
                pair=(spring+correction)[:,None]*normal*(active[:,i]*active[:,j])[:,None]
                force[:,i]=force[:,i]+pair; force[:,j]=force[:,j]-pair
        if self.kind=='matched': force=force+3*torch.tanh(raw[...,:2])*mass[...,None]
        else: force=force+0*raw[...,:2] # all parameter gradients remain auditable
        acceleration=force/mass[...,None]; acceleration=acceleration+torch.stack((torch.zeros_like(context[:,0]),-context[:,0]),-1)[:,None]
        for _ in range(4):
            x=x+v*(dt/4); v=v+acceleration*(dt/4)
        # Explicit public contact prior, not authoritative engine queries or answer repair.
        touch=x[...,1]<radius+.02
        v=torch.stack((torch.where(touch,v[...,0]*torch.exp(-friction*dt*8),v[...,0]),torch.where(touch&(v[...,1]<0),-restitution*v[...,1],v[...,1])),-1)
        x=torch.stack((x[...,0],torch.maximum(x[...,1],radius+.02)),-1)
        contact=touch.to(state.dtype)
        for i in range(n):
            for j in range(i+1,n):
                d=x[:,j]-x[:,i]; dist=d.norm(dim=-1).clamp(min=1e-5); normal=d/dist[:,None]
                closing=((v[:,j]-v[:,i])*normal).sum(-1)
                hit=(dist<radius[:,i]+radius[:,j])&(closing<0)&(active[:,i]>0)&(active[:,j]>0)
                impulse=-(1+restitution[:,i]*restitution[:,j])*closing/(1/mass[:,i]+1/mass[:,j])*hit
                vi=v[:,i]-impulse[:,None]*normal/mass[:,i,None]; vj=v[:,j]+impulse[:,None]*normal/mass[:,j,None]
                v=v.clone(); v[:,i]=vi; v[:,j]=vj
                contact=contact.clone(); contact[:,i]=torch.maximum(contact[:,i],hit.to(state.dtype));contact[:,j]=torch.maximum(contact[:,j],hit.to(state.dtype))
        std=.001+.3*torch.sigmoid(raw[...,2:])
        return torch.cat((x,v),-1)*active[...,None],std,contact*active
