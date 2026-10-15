"""Separate CPU evaluator CLI. Requires candidate freeze; never fits/selects models."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time
from .evaluation import evaluate,paired_cluster_interval,analytic,fit_physics_baselines,physics_monotonicity,decode

CONDITIONS=('graph-concept-on','graph-concept-off','matched-gru')
SEEDS=(17,29,43)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def verify_freeze(output):
    freeze=json.loads((output/'candidate-freeze.json').read_text())
    if freeze.get('status')!='CANDIDATE_FROZEN':raise PermissionError('Candidate must be frozen before any final evaluation')
    root=Path(freeze['source_root']).resolve();locks=freeze.get('source_hashes')
    if not isinstance(locks,dict) or not locks:raise PermissionError('Nonempty frozen candidate source inventory required')
    for name,digest in locks.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise PermissionError('Unsafe source lock')
        if sha(root/path)!=digest:raise PermissionError('Candidate source changed: '+name)
    if sha(output/'data-v1/manifest.json')!=freeze.get('data_manifest_sha256'):raise PermissionError('Data manifest changed')
    data=json.loads((output/'data-v1/manifest.json').read_text())
    for name,digest in data['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(output/'data-v1'/path)!=digest:raise PermissionError('Frozen data file changed: '+name)
    return freeze


def legacy_path(legacy,seed,condition):
    return legacy/('seed17-2000-updates-v1' if seed==17 and condition=='graph-concept-on' else f'remaining-eight-v1/{seed}-{condition}')


def original_predictor(source_root,condition,seed,checkpoint=None,expected_hash=None):
    import torch
    from mindscape.r56.neural import R56Model
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import matched_baseline,controlled_graph
    from mindscape.r56.learning import load_checkpoint
    import mindscape.r56.neural as module
    if Path(module.__file__).resolve()!=source_root/'src/mindscape/r56/neural.py':raise PermissionError('Original source import root differs')
    with torch.random.fork_rng():
        torch.manual_seed(seed);model=matched_baseline('gru',seed=seed) if condition=='matched-gru' else R56Model()
    if checkpoint:load_checkpoint(model,checkpoint,expected_hash)
    model.cpu().eval()
    def predict(problem):
        with torch.no_grad():return model(controlled_graph(pack_problem(problem),condition),memory=None,stateless=False)
    return predict,{'parameters':sum(p.numel() for p in model.parameters()),'device':'cpu','checkpoint_sha256':expected_hash,'phase':'final' if checkpoint else 'initial-reconstructed-assigned-seed'}


def new_predictor(condition,seed,path,expected_hash):
    import torch
    from mindscape.r57.model import make_model
    from mindscape.r56.learning import load_checkpoint
    model=make_model(condition,seed);load_checkpoint(model,path,expected_hash);model.cpu().eval()
    def predict(problem):
        with torch.no_grad():return model(model.prepare(problem))
    return predict,{'parameters':sum(p.numel() for p in model.parameters()),'device':'cpu','checkpoint_sha256':expected_hash}


def invariance(predict,rows):
    """Read-only derived probes; raw changes retained, no repair to final scores."""
    records=[]
    for row in rows:
        p=row['problem']
        if p['domain']!='coding':continue
        first=decode(predict(deepcopy(p)),p)['coding_index'];permuted=deepcopy(p);permuted['candidates']=list(reversed(p['candidates']))
        second=decode(predict(permuted),permuted)['coding_index'];oldwinner=p['candidates'][first] if 0<=first<len(p['candidates']) else None;newwinner=permuted['candidates'][second] if 0<=second<len(permuted['candidates']) else None
        renamed=deepcopy(p);mapping={name:f'identifier_{i}_q' for i,name in enumerate(sorted(set(p['candidates']+[p['requested_name']])))}
        renamed['candidates']=[mapping[x] for x in p['candidates']];renamed['requested_name']=mapping[p['requested_name']]
        third=decode(predict(renamed),renamed)['coding_index']
        records.append({'identity':row['identity'],'original_index':first,'reversed_index':second,'original_winner':oldwinner,'reversed_winner':newwinner,'permutation_equivariant':oldwinner==newwinner,'alpha_index':third,'candidate_alpha_index_invariant':third==first,'scope':'candidate-name alpha renaming only; source-binding alpha invariance separately required'})
    return {'records':records,'permutation_equivariant_fraction':sum(r['permutation_equivariant'] for r in records)/len(records) if records else None,'candidate_alpha_invariant_fraction':sum(r['candidate_alpha_index_invariant'] for r in records)/len(records) if records else None}


def majority_predictor(training):
    math=defaultdict(list);coding=[]
    for row in training:
        p=row['problem'];l=row['labels']
        if p['domain']=='math':math[p['operation']].append((l['math_digit'],l['math_carry']))
        elif p['domain']=='coding':coding.append(l['coding_index'])
    values={op:Counter(v).most_common(1)[0][0] for op,v in math.items()};position=Counter(coding).most_common(1)[0][0]
    def predict(p):
        if p['domain']=='math':return dict(zip(('math_digit','math_carry'),values[p['operation']]))
        if p['domain']=='coding':return {'coding_index':position}
        return {'physics_delta':[0.,0.]}
    return predict


def compare(left,right):
    result={}
    for domain,metric in [('math','correct'),('coding','correct'),('physics','mse')]:
        a=[r for r in left['records'] if r['domain']==domain];b=[r for r in right['records'] if r['domain']==domain]
        change=paired_cluster_interval(a,b,metric)
        delta=change['mean_right_minus_left'];change['status']='REGRESSION' if (delta>0 if domain=='physics' else delta<0) else ('IMPROVED' if delta else 'UNCHANGED')
        result[domain]=change
    return result


def run(output,legacy,original_source,baseline_only=False):
    import torch
    output=Path(output).resolve();legacy=Path(legacy).resolve();original_source=Path(original_source).resolve()
    freeze=verify_freeze(output)
    # Original code must be exact frozen R5.6, not a modified counterpart.
    snapshot=json.loads((output/'baseline-snapshot.json').read_text())
    for name in ('neural.py','native_domains.py','benchmarks.py','learning.py'):
        path=original_source/'src/mindscape/r56'/name
        expected=snapshot['sha256'].get(str(path))
        if not expected or sha(path)!=expected:raise PermissionError('Original source differs from frozen snapshot: '+name)
    torch.set_num_threads(1)
    # Read final only after all candidate/source/data freeze checks above.
    destination=output/('scoreboard-baseline' if baseline_only else 'scoreboard-final');destination.mkdir(exist_ok=False)
    summary=[];paired={};started=time.monotonic()
    for seed in SEEDS:
        rows=json.loads((output/f'data-v1/evaluator-only/final-{seed}.json').read_text());training=json.loads((output/f'data-v1/train/{seed}.json').read_text())
        # Frozen data manifest must explicitly bind each final/train file.
        baselines={'analytic':evaluate(analytic,rows),'training-majority':evaluate(majority_predictor(training),rows)}
        physics,fitmeta=fit_physics_baselines(training)
        prows=[r for r in rows if r['problem']['domain']=='physics']
        for kind in ('zero','constant','linear','analytic'):baselines['physics-'+kind]=evaluate(lambda p,k=kind:physics(k,p),prows)
        write(destination/f'baselines-{seed}.json',{'results':baselines,'fit_metadata':fitmeta,'source':'deterministic_rule_or_training_fitted_baseline; never neural'})
        for condition in CONDITIONS:
            runs={};old=legacy_path(legacy,seed,condition);checkpoint=old/'checkpoint.json'
            for phase in ('initial','final'):
                predictor,meta=original_predictor(original_source,condition,seed,checkpoint if phase=='final' else None,sha(checkpoint) if phase=='final' else None)
                result=evaluate(predictor,rows);result.update(metadata=meta,seed=seed,condition=condition,version='R56',phase=phase)
                result['invariance']=invariance(predictor,rows);result['physics_monotonicity']=physics_monotonicity(predictor,[r['problem'] for r in prows]);runs['R56-'+phase]=result
                write(destination/f'predictions/R56-{seed}-{condition}-{phase}.json',result)
            if not baseline_only:
                fit=output/f'checkpoints/{seed}-{condition}';receipt=json.loads((fit/'result.json').read_text())
                if receipt.get('optimizer_updates')!=2000:raise ValueError('Incomplete R57 fit')
                for phase,key in [('initial','initial_sha256'),('final','checkpoint_sha256')]:
                    path=fit/f'{phase}.json';expected=receipt[key]
                    if sha(path)!=expected:raise ValueError('R57 checkpoint changed')
                    predictor,meta=new_predictor(condition,seed,path,expected);result=evaluate(predictor,rows);result.update(metadata=meta,seed=seed,condition=condition,version='R57',phase=phase)
                    result['invariance']=invariance(predictor,rows);result['physics_monotonicity']=physics_monotonicity(predictor,[r['problem'] for r in prows]);runs['R57-'+phase]=result
                    write(destination/f'predictions/R57-{seed}-{condition}-{phase}.json',result)
                paired[f'{seed}-{condition}']=compare(runs['R56-final'],runs['R57-final'])
            for name,result in runs.items():summary.append({'seed':seed,'condition':condition,'run':name,'metrics':result['metrics']})
    means={}
    for condition in CONDITIONS:
        for runname in sorted({r['run'] for r in summary}):
            selected=[r for r in summary if r['condition']==condition and r['run']==runname];entry={}
            for domain,key in [('math','joint_accuracy'),('coding','accuracy'),('physics','mse')]:
                values=[r['metrics'][domain][key] for r in selected];entry[domain]={'mean':statistics.mean(values),'seed_sd':statistics.stdev(values) if len(values)>1 else None,'seeds':values}
            means[condition+'/'+runname]=entry
    result={'candidate_freeze':freeze,'summary':summary,'means':means,'paired':paired,'R57_status':'PENDING_NOT_SCORED' if baseline_only else 'EVALUATED_NO_SELECTION','evaluation_device':'cpu','seconds':time.monotonic()-started,'optimizer_updates_executed':0,'scientific_significance_claim':False}
    write(destination/'scoreboard.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--legacy',required=True,type=Path);parser.add_argument('--original-source',required=True,type=Path);parser.add_argument('--baseline-only',action='store_true');args=parser.parse_args()
    result=run(args.output,args.legacy,args.original_source,args.baseline_only)
    print(json.dumps({'status':result['R57_status'],'optimizer_updates':0,'evaluation_device':'cpu'}))

if __name__=='__main__':main()
