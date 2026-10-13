"""Independent CPU Pymunk replay and checkpoint inspection; no neural forwards."""
import json,hashlib
from pathlib import Path
import numpy as np
from mindscape.v3.physics_world.schema import Scene,Action
from mindscape.v3.physics_world.environment import Environment,trajectory
from .science import metrics,analyze,selection

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(root):
    root=Path(root);manifest=load(root/'manifest.json');p=manifest['protocol'];evaluation={r['identity']:Scene.from_record(r['scene']) for r in manifest['records'] if r['scene']['split'] in ('iid','ood')}
    rows={i:trajectory(s) for i,s in evaluation.items()};index=load(root/'predictions-index.json');plans=load(root/'planning-index.json');records=load(root/'metrics.json')
    expected={(i,k,s) for i in evaluation for k in p['kinds'] for s in p['optimizer_seeds']}|{(i,k,0) for i in evaluation for k in p['baseline_methods'] if k not in ('matched','v3_1_frozen')}|{(i,'v3_1_frozen',s) for i in evaluation for s in (11,23)}
    if len(index)!=len(expected) or {(r['identity'],r['method'],r['seed']) for r in index}!=expected or len(plans)!=len(expected) or len(records)!=len(expected)*4:raise ValueError('Audit cell coverage')
    savedmetrics={(r['identity'],r['method'],r['seed'],r['horizon']):r for r in records}
    if len(savedmetrics)!=len(records):raise ValueError('Duplicate metric cell')
    for entry,plan in zip(index,plans,strict=True):
        i=entry['identity'];row=rows[i];pred=load(root/entry['file']);saved=load(root/plan['file'])
        if sha(root/entry['file'])!=entry['sha256'] or sha(root/plan['file'])!=plan['sha256'] or plan['identity']!=i:raise ValueError('Prediction hash/identity')
        scores=selection(saved['predictions'],np.array([1.5,1.]))
        if scores!=saved['scores'] or int(np.argmin(scores))!=saved['selected'] or saved['environment_queries_during_selection']!=0:raise ValueError('Planning selection consistency')
        choice=[(x,y) for x in (-2,0,2) for y in (-2,0,2)][saved['selected']];a=np.zeros((32,3,4),np.float32);a[0,0,2:]=choice
        if not np.array_equal(a,np.asarray(saved['action'])):raise ValueError('Chosen action mismatch')
        env=Environment(evaluation[i])
        for act in row['actions'][:3]:env.step(Action.from_vector(act))
        actual=np.stack([env.step(Action.from_vector(act)) for act in a])
        if not np.array_equal(actual,np.asarray(saved['actual'])):raise ValueError('Authoritative planning replay')
        success=bool(np.linalg.norm(actual[-1,0,:2]-[1.5,1.])<=.75)
        if success!=saved['success']:raise ValueError('Success mismatch')
        for horizon in p['horizons']:
            sliced={k:np.asarray(pred[k])[:horizon] for k in ('states','covariance','lower','upper','contacts')};mm=metrics(sliced,row['states'][4:4+horizon],row['observations'][3,:,9],row['observations'][3,:,5]);original=savedmetrics[i,entry['method'],entry['seed'],horizon]
            if original['success']!=success or any(v!=original[k] for k,v in mm.items()):raise ValueError('Recomputed forecast metric')
            # Independent direct MSE implementation checks the shared reporting helper.
            active=np.flatnonzero(row['observations'][3,:,9]);error=sliced['states'][:,active]-row['states'][4:4+horizon,active,:4]
            independent=float(np.mean(error**2/np.array([1,1,9,9])))
            if not np.isclose(independent,original['mse'],rtol=1e-12):raise ValueError('Independent MSE mismatch')
    import torch
    fits=load(root/'fits.json')
    if len(fits)!=6 or {(f['kind'],f['seed']) for f in fits}!={(k,s) for k in p['kinds'] for s in p['optimizer_seeds']}:raise ValueError('Fit count')
    for f in fits:
        file=root/f"{f['kind']}-s{f['seed']}.pt"
        if sha(file)!=f['sha256']:raise ValueError('Checkpoint hash')
        saved=torch.load(file,map_location='cpu',weights_only=True)
        if saved['updates']!=160 or saved['kind']!=f['kind'] or saved['seed']!=f['seed'] or saved['curve']!=f['curve'] or len(saved['curve'])!=160:raise ValueError('Optimizer receipt')
        if not all(torch.isfinite(v).all() for v in saved['state'].values()) or not saved['optimizer']['state'] or any(v['step'].item()!=160 or not torch.isfinite(v['exp_avg']).all() or not torch.isfinite(v['exp_avg_sq']).all() for v in saved['optimizer']['state'].values()):raise ValueError('Checkpoint finite/step')
    if analyze(records,p)!=load(root/'summary.json'):raise ValueError('Recomputed gates')
    return dict(status='PASS',forecasts=len(index),planning_executions=len(plans),independent_scenes=len(evaluation),fits=6,updates=960,neural_forwards=0)
