"""Candidate native adapter. Every numerical choice requires external freeze/review.

No protected data generation occurs here. Construction, backward and receipt inspection
are separable from updates; authorization is checked before native initialization.
"""
import hashlib
import io
import json
from pathlib import Path
import random
import os

import torch
from torch.nn import functional as F
from mindscape.v5.model import GraphTransition, tensors
from mindscape.v5.schema import MAX_NODES, OUTPUTS
from .readiness import DOMAINS, SEEDS, route_core, digest, validate_manifest, validate_slot

STATUS = 'PROPOSED_REQUIRES_FREEZE_AND_REVIEW'
CANDIDATE = dict(status=STATUS, loss='equal changed/unchanged entity MSE strata + 0.1 BCE logits',
                 adam=dict(lr=.001, betas=[.9,.999], eps=1e-8, weight_decay=0., amsgrad=False),
                 permutation_seed=2026101305, bootstrap_seed=2026101306,
                 bootstrap_replicates=10000, checkpoint='only slot240', validity_threshold=.5)


def schedules(manifest):
    validate_manifest(manifest)
    registry = {d: sorted(r['identity'] for r in manifest if r['domain']==d and r['split']=='train') for d in DOMAINS}
    result = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        domains = ['mathematics', 'physics'] * 60 + ['coding'] * 120
        result[str(seed)] = [dict(index=i, domain=d, batch_ids=[rng.choice(registry[d]) for _ in range(16)]) for i,d in enumerate(domains)]
    return result


def permutation(manifest):
    validate_manifest(manifest)
    rng = random.Random(CANDIDATE['permutation_seed'])
    result = {}
    for d in ('mathematics','physics'):
        ids = sorted(r['identity'] for r in manifest if r['domain']==d and r['split']=='train')
        shuffled = ids.copy(); rng.shuffle(shuffled)
        result.update(zip(ids, shuffled))
    return result


def target_tensors(rows, device):
    domain = rows[0]['domain']
    y = torch.zeros(len(rows),MAX_NODES,OUTPUTS[domain],device=device)
    valid = []
    for i,row in enumerate(rows):
        values = row['target']['delta']
        value = row['target']['valid']
        if type(value) is not int or value not in (0,1):
            raise ValueError('Binary validity label')
        value_tensor = torch.as_tensor(values,device=device,dtype=torch.float32)
        if value_tensor.ndim != 2 or value_tensor.shape[1] != OUTPUTS[domain] or not 1<=len(values)<=MAX_NODES or not torch.isfinite(value_tensor).all():
            raise ValueError('Effect shape/finiteness')
        y[i,:len(values)] = value_tensor
        valid.append(value)
    return y, torch.tensor(valid,device=device,dtype=torch.float32)


def balanced_loss(effect, logits, target, valid, active):
    if effect.shape != target.shape or active.shape != effect.shape[:2] or logits.shape != valid.shape:
        raise ValueError('Loss dimensions')
    if any(not torch.isfinite(t).all() for t in (effect,logits,target,valid,active)) or not ((valid==0)|(valid==1)).all() or not ((active==0)|(active==1)).all():
        raise ValueError('Nonfinite loss input')
    error=(effect-target).square().mean(-1)
    changed=target.abs().sum(-1)>0
    terms=[error[mask].mean() for mask in (changed & active.bool(),~changed & active.bool()) if mask.any()]
    if not terms: raise ValueError('Empty active graph')
    return torch.stack(terms).mean()+.1*F.binary_cross_entropy_with_logits(logits,valid)


def tensor_hash(values):
    stream=io.BytesIO()
    # Stable raw content instead of serialization metadata.
    for name,value in sorted(values.items()):
        stream.write(name.encode()); stream.write(str((tuple(value.shape),str(value.dtype))).encode())
        stream.write(value.detach().cpu().contiguous().numpy().tobytes())
    return hashlib.sha256(stream.getvalue()).hexdigest()


def groups(model):
    return {'core':list(model.core.named_parameters()), **{f'{kind}.{d}':list(getattr(model,attr)[d].named_parameters()) for kind,attr in [('encoder','encoders'),('head','heads')] for d in DOMAINS}}


def snapshot(model,opt):
    result={}
    for group,params in groups(model).items():
        states={}
        for name,p in params:
            state=opt.state.get(p,{})
            if state and set(state)!={'step','exp_avg','exp_avg_sq'}: raise ValueError('Unexpected Adam state')
            for key,value in state.items():
                if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all(): raise ValueError('Corrupt Adam state')
                if key=='step' and (value.shape!=torch.Size([]) or value.dtype!=torch.float32 or float(value)!=int(float(value)) or float(value)<0): raise ValueError('Invalid Adam step')
                if key!='step' and (value.shape!=p.shape or value.dtype!=p.dtype or (key=='exp_avg_sq' and (value<0).any())): raise ValueError('Adam shape/dtype/nonnegative variance')
                states[name+'.'+key]=value
        grads=[p.grad for _,p in params]
        if any(g is not None and not torch.isfinite(g).all() for g in grads): raise ValueError('Nonfinite gradient')
        result[group]=dict(parameters=tensor_hash(dict(params)), optimizer=tensor_hash(states),
                           gradient=None if all(g is None for g in grads) else float(sum(g.square().sum() for g in grads if g is not None).sqrt()), counters={name:int(float(opt.state[p]['step'])) if opt.state.get(p) else 0 for name,p in params})
    return result


class NativeCallbacks:
    """External authority/custodian must verify signatures, locks and stage scope.

    authority.verify is deliberately external: a local approved flag is insufficient.
    Data provider may return train targets, public composition graphs only; composition
    outcomes are accessible solely through custodian.evaluate after all nine seals.
    """
    def __init__(self,authority,provider,custodian):
        self.authority=authority; self.provider=provider; self.custodian=custodian
        self.verified=None; self.pending_evidence={}

    def verify_bindings(self,spec,manifest):
        if spec.get('candidate_sha256')!=digest(CANDIDATE) or spec.get('permutation')!=permutation(manifest) or spec.get('schedules')!=schedules(manifest): return False
        if self.authority.verify(spec,manifest,CANDIDATE) is not True: return False
        targets=spec.get('training_target_sha256',{})
        if set(targets)!={r['identity'] for r in manifest if r['split']=='train'} or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in targets.values()): return False
        if self.custodian.verify_source_runtime_locks(spec) is not True: return False
        self.registry={r['identity']:r for r in manifest}
        self.verified=digest(spec)
        return True

    def _check(self,spec):
        if self.verified!=digest(spec): raise ValueError('No independently authenticated frozen authorization')

    def initialize(self,seed,condition,spec):
        self._check(spec)
        from mindscape.v3.physics_world.accelerator import require_mps
        require_mps()
        with torch.random.fork_rng(devices=[]): model=GraphTransition(seed)
        model=model.to('mps')
        if sum(p.numel() for p in model.parameters())!=6967: raise ValueError('Architecture changed')
        opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False)
        initial=tensor_hash(model.state_dict())
        return dict(model=model,opt=opt,next=0,seed=seed,condition=condition,initial=initial),initial

    def step(self,context,condition,slot,spec):
        self._check(spec)
        if condition!=context['condition'] or type(slot['index']) is not int or slot['index']!=context['next'] or slot!=spec['schedules'][str(context['seed'])][context['next']]: raise ValueError('Unexpected slot/order')
        rows=self.provider.training_rows(slot['batch_ids'])
        if len(rows)!=16 or [r['identity'] for r in rows]!=slot['batch_ids'] or any(r['domain']!=slot['domain'] or r['split']!='train' for r in rows): raise ValueError('Training exposure mismatch')
        for row in rows:
            bound=self.registry.get(row['identity'])
            if bound is None or bound['split']!='train' or bound['domain']!=row['domain'] or digest(row['graph'])!=bound['public_sha256']: raise ValueError('Training public binding')
        labels=rows
        if condition=='randomized_auxiliary' and slot['domain']!='coding':
            labels=self.provider.training_rows([spec['permutation'][i] for i in slot['batch_ids']])
            if len(labels)!=16 or [r['identity'] for r in labels]!=[spec['permutation'][i] for i in slot['batch_ids']] or any(r['domain']!=slot['domain'] or r['split']!='train' for r in labels): raise ValueError('Permutation escaped auxiliary training')
        for row in labels:
            if digest(row['target'])!=spec['training_target_sha256'].get(row['identity']): raise ValueError('Training target commitment')
        model,opt=context['model'],context['opt']
        opt.zero_grad(set_to_none=True)
        before=snapshot(model,opt)
        inputs=tensors([r['graph'] for r in rows]); target,valid=target_tensors(labels,inputs[1].device)
        with route_core(model,condition,slot['domain']):
            effect,logits=model(*inputs)
            loss=balanced_loss(effect,logits,target,valid,inputs[4]); loss.backward()
            gradients=snapshot(model,opt)
            opt.step()  # Reachable only after independent authorization and native MPS guard.
        after=snapshot(model,opt)
        for g in before:
            enabled=(g=='core' and (condition!='domain_only' or slot['domain']=='coding')) or g.endswith('.'+slot['domain'])
            if any(after[g]['counters'][n]!=count+int(enabled) for n,count in before[g]['counters'].items()): raise ValueError('Adam actual counter differs')
        receipts={g:dict(gradient=gradients[g]['gradient'],parameters_changed=before[g]['parameters']!=after[g]['parameters'],optimizer_changed=before[g]['optimizer']!=after[g]['optimizer']) for g in before}
        validate_slot(receipts,condition,slot['domain'])
        context['next']+=1
        self.pending_evidence[context['seed'],condition,slot['index']]=dict(before=before,after=after,gradients=gradients,loss=float(loss.detach()))
        return receipts

    def persist_slot(self,seed,condition,slot,groups,output):
        # Custody must retain raw tensor hashes; receipt booleans are never science evidence.
        evidence=self.pending_evidence.pop((seed,condition,slot['index']))
        self.custodian.persist_slot(seed,condition,slot,groups,evidence,output)

    def seal_final(self,context,seed,condition,update,spec,output):
        self._check(spec)
        if type(update) is not int or update!=240 or context['next']!=240: raise ValueError('Truncated fit')
        path=Path(output)/f'{seed}-{condition}.pt'
        if path.exists(): raise ValueError('No overwrite/retry')
        with path.open('xb') as stream:
            torch.save(dict(model=context['model'].state_dict(),optimizer=context['opt'].state_dict(),update=240,spec=digest(spec),seed=seed,condition=condition,initial=context['initial'],schedule_sha256=digest(spec['schedules'][str(seed)]),bindings=spec['bindings']),stream)
            stream.flush(); os.fsync(stream.fileno())
        public=self.provider.composition_public()
        if len(public)!=72 or any(set(r)!={'identity','domain','graph'} for r in public): raise ValueError('Composition public contract')
        expected={i for i,r in self.registry.items() if r['split']=='composition'}
        if {r['identity'] for r in public}!=expected: raise ValueError('Composition identities differ')
        for row in public:
            bound=self.registry[row['identity']]
            if row['domain']!=bound['domain'] or row['graph']['domain']!=row['domain'] or digest(row['graph'])!=bound['public_sha256']: raise ValueError('Composition public binding')
        predictions=[]
        with torch.inference_mode():
            for row in public:
                effect,valid=context['model'](*tensors([row['graph']]))
                predictions.append(dict(identity=row['identity'],domain=row['domain'],delta=effect[0,:len(row['graph']['entities'])].cpu().tolist(),valid_probability=float(valid.sigmoid()[0])))
        prediction_path=path.with_suffix('.predictions.json')
        with prediction_path.open('x') as stream:
            stream.write(json.dumps(predictions,allow_nan=False)); stream.flush(); os.fsync(stream.fileno())
        return self.custodian.seal(seed,condition,update,path,prediction_path,spec)

    def verify_all_seals(self,seals,spec):
        self._check(spec)
        return self.custodian.verify_nine_signed_seals(seals,spec)

    def independent_evaluate(self,seals,spec,manifest,output):
        self._check(spec)
        if self.authority.verify_stage('final_evaluation',spec,seals) is not True: raise ValueError('Separate final evaluation authorization missing')
        if self.verify_all_seals(seals,spec) is not True: raise ValueError('Custody missing')
        return self.custodian.evaluate_with_locked_v5_executors(seals,spec,manifest,output)
