"""Frozen V3.1 mean predictor; corrected kinematic covariance, no training import."""
import numpy as np
import torch
from mindscape.v3_1.physics.model import Dynamics
from .public import validate,VERSION
from .uncertainty import propagate,kinematic_matrices

def load(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=Dynamics(saved['identity']['variant'],saved['identity']['seed'])
    model.load_state_dict(saved['state'])
    for key in ('memory_keys','memory_residuals'):
        value=saved.get(key)
        setattr(model,key,None if value is None else value.to('mps'))
    model.memory_active=saved.get('memory_active')
    model.eval()
    return model

def forecast(model,request):
    h,m,a,c=validate(request)
    h=torch.tensor(h[None],device='mps');m=torch.tensor(m[None],device='mps')
    states=[];contacts=[];cov=[];p=np.zeros((12,12));f,b=kinematic_matrices(1/30)
    F=np.kron(np.eye(3),f);B=np.kron(np.eye(3),b)
    with torch.inference_mode():
        for action,context in zip(a,c,strict=True):
            state,var,contact,_=model(h,m,torch.tensor(action[None],device='mps'),torch.tensor(context[None],device='mps'))
            q=np.diag(var[0,:,2:].cpu().numpy().ravel())
            p=propagate(p,F,B,q);cov.append(p);states.append(state[0].cpu().numpy());contacts.append(contact[0].cpu().numpy())
            current=torch.cat((state,h[:,-1,:,4:8],contact[...,None],h[:,-1,:,9:10]),-1)
            h=torch.cat((h[:,1:],current[:,None]),1);status=m[:,-1].clone();status[...,:4]=2*current[...,9:10];m=torch.cat((m[:,1:],status[:,None]),1)
    means=np.stack(states);cov=np.stack(cov);width=1.96*np.sqrt(np.diagonal(cov,axis1=1,axis2=2).reshape(-1,3,4))
    return dict(version=VERSION,states=means,covariance=cov,lower=means-width,upper=means+width,contacts=np.stack(contacts),private_queries=0)
