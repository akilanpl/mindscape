"""Executable numerical candidate; training targets cannot enter inference.

CPU loss/backward is a synthetic-only diagnostic. Native fitting requires an
external validated authorization and MPS; no command-line launch entry exists.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..benchmark import Task, digest
from ..models import LocalTransition
from mindscape.r5_2_math.arithmetic import run, sign_features
from mindscape.r5_2_math.model import features

SPEC = {
    'version': 'alpha-ood-numerical-v2', 'status': 'PROPOSED_REQUIRES_REVIEW',
    'indices': ['operation[3]', 'left[10]', 'right[10]', 'incoming[10]'],
    'relations': 'R5.2 features: seven observable features only',
    'hidden': [48], 'heads': {'digit': 10, 'carry': 10, 'sign': 2},
    'loss': 'mean digit CE + mean carry CE over valid teacher transitions + mean sign CE over signs in the same sampled tasks',
    'weights': [1, 1, 1], 'feedback_train': 'full teacher trace; no scheduled sampling',
    'feedback_evaluation': 'predicted sign/digit/carry and predicted complete operation state',
    'reset': 'task start and every scheduler row/addition index=0; full BPTT within row',
    'sampling': 'NumPy PCG64 seed; 800 with-replacement batches of 32 training task indices',
    'optimizer': {'name': 'Adam', 'lr': .003, 'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False},
    'schedule': 'constant; no clipping; no validation tuning; final checkpoint only',
    'initialization': 'LocalTransition seeded CPU initialization, common embeddings/heads exact parity before native placement',
    'padding': 'zero input; valid mask excludes loss and hidden update; reset before step',
    'checkpoint': 'all model and optimizer state, 800 losses, exact batch plan, seed, bindings; exclusive creation',
}

def native_task(task: Task) -> dict:
    return {'operands': [task.start, *[n for _, n in task.steps]],
            'operations': [op for op, _ in task.steps], 'split': task.split}


def batch_plan(seed: int, count: int = 768) -> np.ndarray:
    if seed not in (11,23,47) or count != 768:
        raise ValueError('frozen seed/train count mismatch')
    return np.random.Generator(np.random.PCG64(seed)).integers(count, size=(800,32))


def package_training(tasks: tuple[Task,...], traces: list[dict]) -> list[dict]:
    """Only custodian-released train demonstrations; never accepts OOD/validation."""
    if len(tasks) != len(traces) or not tasks or any(t.split != 'train' for t in tasks):
        raise ValueError('training-only trace coverage required')
    output=[]
    for task,trace in zip(tasks,traces,strict=True):
        if len(trace['stages']) != len(task.steps) or not trace['transitions']:
            raise ValueError('incomplete demonstration')
        sequence=[]
        for event in trace['transitions']:
            x,r=features(event['state']); y=(event['written_digit'],event['written_carry'])
            if any(type(v) is not int or not 0 <= v < 10 for v in y):
                raise ValueError('invalid teacher vocabulary')
            sequence.append((x,r,y,event['state']['index']==0))
        signs=[]; current=task.start
        for (op,right),stage in zip(task.steps,trace['stages'],strict=True):
            if type(stage['sign']) is not int or stage['sign'] not in (0,1):
                raise ValueError('invalid sign')
            signs.append((sign_features(current,right,op),stage['sign']))
            current=int(stage['answer'])
        output.append({'identity':task.identity,'sequence':sequence,'signs':signs})
    return output


def loss(model: LocalTransition, examples: list[dict], device: str) -> torch.Tensor:
    """Real differentiable loss, no optimizer step. Variable-length task masking."""
    if not examples:
        raise ValueError('empty batch')
    b=len(examples); length=max(len(e['sequence']) for e in examples)
    hidden=torch.zeros((b,48),device=device)
    total=torch.zeros((),device=device); denominator=sum(len(e['sequence']) for e in examples)
    for j in range(length):
        rows=[e['sequence'][j] if j<len(e['sequence']) else ([0,0,0,0],[0.]*7,[0,0],True) for e in examples]
        x=torch.tensor([r[0] for r in rows],dtype=torch.long,device=device)
        relations=torch.tensor([r[1] for r in rows],dtype=torch.float32,device=device)
        y=torch.tensor([r[2] for r in rows],dtype=torch.long,device=device)
        valid=torch.tensor([j<len(e['sequence']) for e in examples],device=device)
        reset=torch.tensor([r[3] for r in rows],device=device)
        old=hidden; hidden=hidden * (~reset[:,None])
        d,c,next_hidden=model(x,relations,hidden)
        hidden=torch.where(valid[:,None],next_hidden,old)
        total=total+((F.cross_entropy(d,y[:,0],reduction='none')+F.cross_entropy(c,y[:,1],reduction='none'))*valid).sum()/denominator
    signs=[s for e in examples for s in e['signs']]
    sx=torch.tensor([s[0] for s in signs],dtype=torch.float32,device=device)
    sy=torch.tensor([s[1] for s in signs],dtype=torch.long,device=device)
    total=total+F.cross_entropy(model.sign_head(sx),sy)
    if not torch.isfinite(total):
        raise ValueError('nonfinite loss')
    return total


class Predictor:
    def __init__(self,model,device):
        self.model=model.eval(); self.device=device; self.observed_events=[]; self.observed_signs=[]; self.reset()
    def reset(self):
        self.hidden=torch.zeros((1,48),device=self.device)
    def local(self,state):
        x,r=features(state)
        with torch.inference_mode():
            d,c,self.hidden=self.model(torch.tensor([x],device=self.device),torch.tensor([r],dtype=torch.float32,device=self.device),self.hidden)
        digit,carry=int(d.argmax(-1)),int(c.argmax(-1))
        diagnostics={'kind':self.model.kind,'digit_probabilities':d.softmax(-1)[0].cpu().tolist(),
                     'carry_probabilities':c.softmax(-1)[0].cpu().tolist()}
        self.observed_events.append({'state':dict(state),'written_digit':digit,'written_carry':carry,'prediction':diagnostics})
        return digit,carry,diagnostics
    def sign(self,a,b,op):
        with torch.inference_mode():
            z=self.model.sign_head(torch.tensor([sign_features(a,b,op)],dtype=torch.float32,device=self.device))
        sign=int(z.argmax(-1)); diagnostics={'kind':self.model.kind,'sign_probabilities':z.softmax(-1)[0].cpu().tolist()}
        self.observed_signs.append({'left':a,'right':b,'operation':op,'sign':sign,'prediction':diagnostics})
        return sign,diagnostics


def predict_task(task, model, device, *, access=None, bindings=None):
    """No label/reference imports. Retain invalid results as explicit failed rows."""
    if task.split=='ood':
        from .custody import require_access
        require_access(access,'evaluate',bindings)
        from mindscape.v3.physics_world.accelerator import require_mps
        require_mps()
        if device!='mps':
            raise ValueError('scientific OOD predictions require native placement')
    predictor=Predictor(model,device)
    try:
        raw=run(native_task(task),predictor)
        return {'identity':task.identity,'result':raw,'failure':None}
    except (ValueError,RuntimeError) as exc:
        return {'identity':task.identity,'result':None,'failure':{'type':type(exc).__name__,'message':str(exc),
            'partial_transitions':predictor.observed_events,'partial_signs':predictor.observed_signs}}


def fit_native(*, tasks, traces, kind, seed, bindings, authorization, checkpoint: Path):
    """Future complete optimizer path; external gate validates before allocation.

    authorization is an instance of custody.VerifiedAccess created only after
    independent signature checks, with immutable content and action binding.
    """
    from .custody import require_access
    from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
    require_access(authorization,'fit',bindings)
    require_mps()
    if len(tasks)!=768 or len(set(t.identity for t in tasks))!=768:
        raise ValueError('frozen train count')
    examples=package_training(tasks,traces)
    if bindings['train_demonstration_hash'] != digest(traces) or bindings['train_input_hash'] != digest([t.public() for t in tasks]):
        raise ValueError('training content substitution')
    if checkpoint.exists():
        raise FileExistsError('no overwrite or retry')
    plan=batch_plan(seed)
    model=LocalTransition(kind,seed).to('mps'); verify_placement(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    losses=[]
    for indices in plan:
        optimizer.zero_grad(set_to_none=True)
        value=loss(model,[examples[int(i)] for i in indices],'mps')
        value.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('disconnected/nonfinite gradient; abort without retry')
        optimizer.step()
        losses.append(float(value.detach().cpu()))
    torch.mps.synchronize()
    payload={'version':SPEC['version'],'model':model.state_dict(),'optimizer':optimizer.state_dict(),
             'kind':kind,'seed':seed,'updates':800,'losses':losses,'batch_plan':plan.tolist(),
             'bindings':dict(bindings),'spec_hash':digest(SPEC)}
    with checkpoint.open('xb') as stream:
        torch.save(payload,stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return model,{'checkpoint_hash':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'updates':800,'seed':seed,'kind':kind}


def load_final_checkpoint(path: Path, receipt: dict, bindings: dict, device='mps'):
    """Reload the single final checkpoint; exact bytes and complete plan required."""
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['checkpoint_hash']:
        raise ValueError('checkpoint byte substitution')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if type(payload['updates']) is not int or payload['updates']!=800 or len(payload['losses'])!=800 or payload['bindings']!=bindings or payload['spec_hash']!=digest(SPEC):
        raise ValueError('checkpoint metadata mismatch')
    if type(payload['seed']) is not int or payload['seed'] not in (11,23,47) or payload['seed']!=receipt['seed'] or payload['kind']!=receipt['kind'] or payload['batch_plan']!=batch_plan(payload['seed']).tolist():
        raise ValueError('checkpoint schedule mismatch')
    if any(type(v) not in (int,float) or not np.isfinite(v) for v in payload['losses']):
        raise ValueError('checkpoint nonfinite losses')
    groups=payload['optimizer']['param_groups']
    states=payload['optimizer']['state']
    if len(groups)!=1 or groups[0]['lr']!=.003 or tuple(groups[0]['betas'])!=(.9,.999) or groups[0]['eps']!=1e-8 or groups[0]['weight_decay']!=0 or groups[0]['amsgrad'] is not False:
        raise ValueError('checkpoint Adam configuration')
    model=LocalTransition(payload['kind'],payload['seed'])
    parameters=list(model.parameters())
    if len(states)!=len(parameters) or len(groups[0]['params'])!=len(parameters):
        raise ValueError('checkpoint Adam parameter coverage')
    for identifier,param in zip(groups[0]['params'],parameters,strict=True):
        state=states[identifier]
        step=state['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or step.item()!=800:
            raise ValueError('checkpoint optimizer step count')
        for name in ('exp_avg','exp_avg_sq'):
            if state[name].shape!=param.shape or not torch.isfinite(state[name]).all():
                raise ValueError('checkpoint Adam moment corruption')
        if (state['exp_avg_sq']<0).any():
            raise ValueError('checkpoint negative second moment')
    model.load_state_dict(payload['model'],strict=True)
    if device!='mps':
        raise ValueError('scientific checkpoint placement must be native MPS')
    from mindscape.v3.physics_world.accelerator import require_mps,verify_placement
    require_mps(); model=model.to('mps'); verify_placement(model)
    return model.eval()
