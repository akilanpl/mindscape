"""Inference only: no simulator, dataset, verifier, or training imports."""
import numpy as np
import torch
from .public import validate,VERSION

def initial(h,m):
    state=h[-1,:,:4].copy()
    state[:,2:]=np.where(m[-1,:,2:4]>0,state[:,2:],(h[-1,:,:2]-h[-2,:,:2])*30)
    attrs=h[-1,:,4:].copy()[:,[0,1,2,3,5]]
    attrs[:,0]=np.where(m[-1,:,4]>0,attrs[:,0],1)
    attrs[:,2]=np.where(m[-1,:,6]>0,attrs[:,2],.2)
    attrs[:,3]=np.where(m[-1,:,7]>0,attrs[:,3],.5)
    return state,attrs

def forecast(model,request,seed=0,particles=32):
    h,m,a,c=validate(request); state,attrs=initial(h,m); rng=np.random.default_rng(seed)
    states=np.repeat(state[None],particles,0); at=np.repeat(attrs[None],particles,0)
    # Priors fixed a priori, persistent across a trajectory. Hidden mass/materials never read.
    for i in range(3):
        if not m[-1,i,4]: at[:,i,0]=rng.uniform(.8,1.8,particles)
        if not m[-1,i,6]: at[:,i,2]=rng.uniform(.1,.4,particles)
        if not m[-1,i,7]: at[:,i,3]=rng.uniform(.4,.95,particles)
    states[:,:,:2]+=rng.normal(0,.005,states[:,:,:2].shape)
    states[:,:,2:]+=rng.normal(0,np.where(m[-1,:,2:4]>0,.005,.212),states[:,:,2:].shape)
    at=torch.tensor(at,device='mps'); s=torch.tensor(states,device='mps',dtype=torch.float32)
    allstates=[]; contacts=[]
    with torch.inference_mode():
        for action,context in zip(a,c,strict=True):
            nxt,std,contact=model(s,at,torch.tensor(np.repeat(action[None],particles,0),device='mps'),torch.tensor(np.repeat(context[None],particles,0),device='mps'))
            noise=torch.tensor(rng.normal(size=(particles,3,2)).astype(np.float32),device='mps')*std*at[...,4,None]
            s=nxt+torch.cat((noise/60,noise),-1)
            allstates.append(s.cpu().numpy());contacts.append(contact.cpu().numpy().mean(0))
    samples=np.stack(allstates,1); mean=samples.mean(0)
    covariance=np.stack([np.cov(samples[:,t].reshape(particles,12),rowvar=False) for t in range(len(a))])
    require_finite=[mean,covariance,samples]
    if not all(np.isfinite(v).all() for v in require_finite): raise RuntimeError('Nonfinite forecast')
    return {'version':VERSION,'states':mean,'covariance':covariance,'lower':np.quantile(samples,.025,axis=0),'upper':np.quantile(samples,.975,axis=0),'contacts':np.stack(contacts),'samples':samples,'private_queries':0}
