"""Proposed task-paired scorer and real executors; never called on this sprint data."""
import numpy as np
from .readiness import DOMAINS, SEEDS, CONDITIONS, validate_manifest, digest
from mindscape.v5.execute import execute


def independent_outcomes(tasks, runtime_lock, executor=None):
    """Invoke actual register/WASI/Pymunk executors after caller verifies nine seals."""
    import importlib.metadata
    import hashlib
    from pathlib import Path
    required=('src/mindscape/v5/execute.py','src/mindscape/v3/coding_learning/execution.py','src/mindscape/v3/physics_world/environment.py')
    if any(p not in runtime_lock['source_files'] for p in required) or not runtime_lock['wasi_assets']: raise ValueError('Executor source/WASI locks missing')
    for relative,expected in runtime_lock['source_files'].items():
        from .numerical_locks import checked_path
        if hashlib.sha256(checked_path(runtime_lock['repository'],relative).read_bytes()).hexdigest()!=expected: raise ValueError('Executor source hash differs')
    for asset,expected in runtime_lock['wasi_assets'].items():
        if Path(asset).is_symlink() or '..' in Path(asset).parts or not Path(asset).is_absolute(): raise ValueError('Unsafe WASI asset path')
        if any(p.is_symlink() for p in Path(asset).parents): raise ValueError('Symlink WASI asset parent')
        if hashlib.sha256(Path(asset).read_bytes()).hexdigest()!=expected: raise ValueError('WASI asset hash differs')
    if runtime_lock['pymunk_version'] != importlib.metadata.version('pymunk'):
        raise ValueError('Pinned Pymunk differs')
    if executor is None: raise ValueError('Locked WASI Executor required')
    result={}
    for task in tasks:
        if task['identity'] in result: raise ValueError('Duplicate independent identity')
        result[task['identity']]=execute(task,executor)
    return result


def task_metrics(prediction,target):
    if type(target['valid']) is not int or target['valid'] not in (0,1): raise ValueError('Binary executed validity')
    actual=np.asarray(target['delta'],dtype=float); estimate=np.asarray(prediction['delta'],dtype=float)
    probability=prediction['valid_probability']
    if actual.shape!=estimate.shape or actual.ndim!=2 or 0 in actual.shape or actual.shape[0]>7 or not np.isfinite(actual).all() or not np.isfinite(estimate).all() or type(probability) not in (int,float) or not np.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Invalid raw predictions')
    squared=((estimate-actual)**2).mean(axis=1); changed=np.abs(actual).sum(axis=1)>0
    return dict(changed=float(squared[changed].mean()) if changed.any() else None,
                unchanged=float(squared[~changed].mean()) if (~changed).any() else None,
                whole=float(squared.mean()),invalid=float((estimate**2).mean()) if target['valid']==0 else None,
                valid=int((probability>=.5)==bool(target['valid'])),valid_class=target['valid'])


def score_campaign(predictions,truth,manifest,public_graphs,replicates=10000):
    """Candidate: task macro MSE; paired task bootstrap, seeds never count as tasks."""
    validate_manifest(manifest)
    if replicates!=10000: raise ValueError('Bootstrap freeze differs')
    expected={(s,c) for s in SEEDS for c in CONDITIONS}
    if set(predictions)!=expected: raise ValueError('Nine raw predictions required')
    ids={d:sorted(r['identity'] for r in manifest if r['domain']==d and r['split']=='composition') for d in DOMAINS}
    all_ids=set(sum(ids.values(),[]))
    if set(public_graphs)!=all_ids: raise ValueError('Bound public composition graphs required')
    for row in manifest:
        if row['split']=='composition' and digest(public_graphs[row['identity']])!=row['public_sha256']: raise ValueError('Scorer graph commitment differs')
    if any(len(v)!=24 for v in ids.values()) or set(truth)!=all_ids: raise ValueError('Composition denominator')
    metrics={}
    for fit,rows in predictions.items():
        if len(rows)!=72 or len({r['identity'] for r in rows})!=72 or {r['identity'] for r in rows}!=all_ids: raise ValueError('Missing prediction identity')
        from mindscape.v5.schema import OUTPUTS
        for row in rows:
            domain=next(d for d in DOMAINS if row['identity'] in ids[d])
            if len(row['delta'])!=len(public_graphs[row['identity']]['entities']) or row.get('domain')!=domain or any(len(a)!=OUTPUTS[domain] for a in row['delta']): raise ValueError('Prediction domain/output binding')
        metrics[fit]={r['identity']:task_metrics(r,truth[r['identity']]) for r in rows}
    gates={}; gains={}; intervals={}; regressions={}
    def values(seed,condition,domain,key):
        return np.array([metrics[seed,condition][i][key] for i in ids[domain] if metrics[seed,condition][i][key] is not None])
    for control in ('domain_only','randomized_auxiliary'):
        per_seed=[]; paired=[]
        for seed in SEEDS:
            a=values(seed,control,'coding','changed'); b=values(seed,'shared','coding','changed')
            if len(a)==0 or len(a)!=len(b) or a.mean()<=0: raise ValueError('Undefined changed gain')
            per_seed.append(float(1-b.mean()/a.mean())); paired.append((a,b))
        gains[control]=per_seed
        rng=np.random.default_rng(2026101306)
        draws=[]
        for _ in range(replicates):
            indices=rng.integers(len(paired[0][0]),size=len(paired[0][0]))
            draws.append(np.mean([1-b[indices].mean()/a[indices].mean() if a[indices].mean()>0 else -np.inf for a,b in paired]))
        lower=float(np.quantile(draws,.025)) if np.isfinite(draws).all() else None; intervals[control]=lower
        gates['gain_'+control]=np.mean(per_seed)>=.10
        gates['seeds_'+control]=all(v>0 for v in per_seed)
        gates['bootstrap_'+control]=lower is not None and lower>0
        for domain in DOMAINS:
            for key in ('whole','changed'):
                if key=='changed' and domain=='coding': continue
                fractions=[]
                for seed in SEEDS:
                    a=values(seed,control,domain,key).mean(); b=values(seed,'shared',domain,key).mean()
                    fractions.append(float((b-a)/a) if a>0 else (0. if b==0 else None))
                name=f'{domain}_{key}_{control}'; regressions[name]=fractions
                gates[name]=all(v is not None and v<=.05 for v in fractions)
    for seed in SEEDS:
        for domain in DOMAINS:
            selected=[metrics[seed,'shared'][i] for i in ids[domain]]
            classes=[[r['valid'] for r in selected if r['valid_class']==label] for label in (0,1)]
            if not all(classes): raise ValueError('Both validity classes required')
            gates[f'validity_{seed}_{domain}']=np.mean([np.mean(c) for c in classes])>=.90
    return dict(status='GATES_PASS_PENDING_INDEPENDENT_AUDIT' if all(gates.values()) else 'NO_GO',gates={k:bool(v) for k,v in gates.items()},gains=gains,bootstrap_lower=intervals,regressions=regressions,metrics={f'{s}-{c}':v for (s,c),v in metrics.items()})
