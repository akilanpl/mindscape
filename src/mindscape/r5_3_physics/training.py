"""Native MPS training only; no inference module imports this file."""
import time
import numpy as np
import torch
from .model import World
from .predict import initial
from mindscape.v3.physics_world.accelerator import verify_placement

def fit(rows,kind,seed,protocol,path):
    inputs=[]
    for row in rows:
        for t in range(3,len(row['actions'])-4,4):
            state,attrs=initial(row['observations'][t-3:t+1],row['masks'][t-3:t+1])
            inputs.append((state,attrs,row['actions'][t:t+4],row['contexts'][t:t+4],row['states'][t+1:t+5,:,:4]))
    arrays=[torch.tensor(np.stack([v[k] for v in inputs]),device='mps',dtype=torch.float32) for k in range(5)]
    model=World(seed,kind);opt=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate']);rng=np.random.default_rng(seed);curve=[];start=time.perf_counter()
    for update in range(protocol['updates_per_fit']):
        idx=torch.tensor(rng.integers(len(inputs),size=protocol['batch_size']),device='mps');state,attrs,actions,contexts,targets=[v[idx] for v in arrays];active=attrs[...,4,None];loss=0
        opt.zero_grad(set_to_none=True)
        for t in range(4):
            state,std,_=model(state,attrs,actions[:,t],contexts[:,t]);error=state-targets[:,t]
            scale=torch.tensor([1,1,3,3],device='mps');mse=((error/scale).square()*active).sum()/active.sum().clamp(min=1)/4
            nll=((error[...,2:].square()/std.square()+2*std.log())*active).sum()/active.sum().clamp(min=1)/2
            loss=loss+mse+.02*nll
        require=torch.isfinite(loss).item()
        if not require: raise RuntimeError('Nonfinite training loss')
        loss.backward();verify_placement(model,gradients=True)
        if not all(torch.isfinite(v.grad).all().item() for v in model.parameters()): raise RuntimeError('Nonfinite gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();curve.append(float(loss.item()))
    torch.mps.synchronize()
    torch.save({'state':model.state_dict(),'optimizer':opt.state_dict(),'kind':kind,'seed':seed,'updates':160,'curve':curve},path)
    return model,dict(kind=kind,seed=seed,updates=160,curve=curve,seconds=time.perf_counter()-start,training_windows=len(inputs),parameters=sum(v.numel() for v in model.parameters()))
