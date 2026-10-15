"""Independent exploratory evaluation. Never trains, repairs or selects checkpoints."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
import random


def canonical_id(problem):
    observable={k:v for k,v in problem.items() if k not in {'target_name','labels','instruction','concept_ids'}}
    if observable.get('domain')=='math':
        observable['incoming']=observable.pop('carry_in',observable.pop('borrow_in',observable.get('incoming',0)))
    return hashlib.sha256(json.dumps(observable,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def decode(output,problem):
    domain=problem['domain']
    def number(v):
        if hasattr(v,'detach'):v=v.detach().cpu().tolist()
        return v
    if domain=='math':
        return {k:int(number(output[k])) if k in output else int(output[head].argmax().item()) for k,head in [('math_digit','math_digit_logits'),('math_carry','math_carry_logits')]}
    if domain=='coding':
        return {'coding_index':int(number(output['coding_index'])) if 'coding_index' in output else int(output['coding_logits'][:len(problem['candidates'])].argmax().item())}
    values=number(output.get('physics_delta'))
    if not isinstance(values,(list,tuple)) or len(values)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Finite two-axis delta prediction required')
    return {'physics_delta':[float(v) for v in values]}


def evaluate(predictor,rows):
    """Predictor receives problem only, references consumed after original decoding."""
    seen=set();records=[]
    for row in rows:
        problem=row['problem'];identity=canonical_id(problem)
        if identity in seen:raise ValueError('Duplicate normalized evaluation problem')
        seen.add(identity)
        # Deep copy prevents a predictor from mutating references or future cases.
        p=json.loads(json.dumps(problem));p.pop('target_name',None);p.pop('labels',None)
        predicted=decode(predictor(p),p);labels=row['labels'];domain=p['domain']
        r={'identity':row.get('identity',identity),'normalized_identity':identity,'family':row.get('family',identity),'domain':domain,'prediction':predicted,'reference':labels,'repaired':False}
        if domain=='math':r.update(digit_correct=predicted['math_digit']==labels['math_digit'],carry_correct=predicted['math_carry']==labels['math_carry']);r['correct']=r['digit_correct'] and r['carry_correct']
        elif domain=='coding':r.update(correct=predicted['coding_index']==labels['coding_index'],target_position=labels['coding_index'],scope=row.get('scope',row.get('family','unspecified')))
        else:
            delta=[a-b for a,b in zip(predicted['physics_delta'],labels['physics_delta'])];r.update(mse=sum(v*v for v in delta)/2,mae=sum(abs(v) for v in delta)/2,sign_correct=sum((a>0)-(a<0)==(b>0)-(b<0) for a,b in zip(predicted['physics_delta'],labels['physics_delta']))/2)
        records.append(r)
    return {'format':'r57-independent-evaluation-v1','records':records,'metrics':summarize(records),'qualification':False,'selection_performed':False}


def summarize(records):
    result={}
    for domain in ('math','coding','physics'):
        rows=[r for r in records if r['domain']==domain]
        if not rows:continue
        mean=lambda key:sum(float(r[key]) for r in rows)/len(rows)
        result[domain]={'count':len(rows)}
        if domain=='math':result[domain].update(joint_accuracy=mean('correct'),digit_accuracy=mean('digit_correct'),carry_accuracy=mean('carry_correct'))
        elif domain=='coding':
            groups=defaultdict(list)
            for r in rows:groups[str(r['target_position'])].append(r)
            result[domain].update(accuracy=mean('correct'),by_target_position={k:{'count':len(v),'accuracy':sum(x['correct'] for x in v)/len(v)} for k,v in groups.items()})
            scopes=defaultdict(list)
            for r in rows:scopes[r['scope']].append(r)
            result[domain]['by_scope']={k:{'count':len(v),'accuracy':sum(x['correct'] for x in v)/len(v)} for k,v in scopes.items()}
        else:result[domain].update(mse=mean('mse'),mae=mean('mae'),axis_sign_accuracy=mean('sign_correct'))
    return result


def paired_cluster_interval(left,right,metric,replicates=2000,seed=2027):
    """Paired family bootstrap, descriptive only; no significance declaration."""
    a={r['normalized_identity']:r for r in left};b={r['normalized_identity']:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):raise ValueError('Same unique cases required for paired comparison')
    groups=defaultdict(list)
    for identity,row in a.items():
        if row['family']!=b[identity]['family']:raise ValueError('Cluster identity mismatch')
        groups[row['family']].append(float(b[identity][metric])-float(row[metric]))
    if not groups:raise ValueError('No paired cases')
    families=sorted(groups);rng=random.Random(seed);means=[]
    for _ in range(replicates):
        values=[v for family in rng.choices(families,k=len(families)) for v in groups[family]]
        means.append(sum(values)/len(values))
    means.sort();all_values=[v for vals in groups.values() for v in vals]
    return {'mean_right_minus_left':sum(all_values)/len(all_values),'interval95':[means[int(.025*replicates)],means[min(len(means)-1,int(.975*replicates))]],'clusters':len(families),'interpretation':'descriptive paired-family bootstrap; no significance claim','raw_differences':{k:v for k,v in groups.items()}}


def analytic(problem):
    """Transparent deterministic reference baseline, never neural output."""
    d=problem['domain']
    if d=='math':
        a,b=problem['left'],problem['right'];incoming=problem.get('incoming',problem.get('carry_in',problem.get('borrow_in',0)))
        value=a-b-incoming if problem['operation']=='borrow_digit' else a+b+incoming
        return {'math_digit':value%10,'math_carry':int(value<0) if problem['operation']=='borrow_digit' else value//10}
    if d=='coding':return {'coding_index':problem['candidates'].index(problem['requested_name'])}
    return {'physics_delta':[(f*problem.get('dt',1)+j)/problem['mass'] for f,j in zip(problem.get('force',[0,0]),problem.get('impulse',[0,0]))]}


def fit_physics_baselines(training_rows):
    """Training-only fitted constant/linear controls; no target-derived input features."""
    import torch
    rows=[r for r in training_rows if r['problem']['domain']=='physics']
    if not rows:raise ValueError('Physics training examples required')
    features=lambda p:[1.,*p['velocity'],p['mass'],*p.get('force',[0,0]),p.get('dt',1),*p.get('impulse',[0,0])]
    x=torch.tensor([features(r['problem']) for r in rows],dtype=torch.float64);y=torch.tensor([r['labels']['physics_delta'] for r in rows],dtype=torch.float64)
    weights=torch.linalg.lstsq(x,y,driver='gelsd').solution;constant=y.mean(0).tolist()
    def predict(kind,p):
        if kind=='zero':return {'physics_delta':[0.,0.]}
        if kind=='constant':return {'physics_delta':list(constant)}
        if kind=='linear':return {'physics_delta':(torch.tensor(features(p),dtype=torch.float64)@weights).tolist()}
        if kind=='analytic':return analytic(p)
        raise ValueError('Unknown baseline')
    return predict,{'fit_source':'training labels only','features':'raw velocity,mass,force,dt,impulse +intercept; no Fdt/m prior','linear_method':'CPU float64 least-squares SVD','constant':constant}


def physics_monotonicity(predictor,problems):
    records=[]
    for p in problems:
        for axis in (0,1):
            base=json.loads(json.dumps(p));base.setdefault('force',[0.,0.]);changed=json.loads(json.dumps(base));changed['force'][axis]+=1.
            first=decode(predictor(base),base)['physics_delta'];second=decode(predictor(changed),changed)['physics_delta'];difference=second[axis]-first[axis]
            records.append({'identity':canonical_id(base),'axis':axis,'delta_for_force_plus_one':difference,'nondecreasing':difference>=0,'strict_increase':difference>0,'reference_delta':base.get('dt',1)/base['mass']})
    return {'records':records,'nondecreasing_fraction':sum(r['nondecreasing'] for r in records)/len(records) if records else None,'scope':'paired force perturbation only; zero predictor passes nondecrease, inspect strict increase and reference magnitude'}


def evaluate_r56_checkpoint(checkpoint,expected_hash,rows,condition,*,source_root):
    """Independent evaluator process uses preserved original source and safe JSON only."""
    import importlib.util
    import torch
    from pathlib import Path
    root=Path(source_root)
    # Caller must launch a separate evaluator process with original source first on
    # PYTHONPATH; refuse a different neural module rather than import-cache mixing.
    import mindscape.r56.neural as neural
    if Path(neural.__file__).resolve()!=root.resolve()/'src/mindscape/r56/neural.py':raise ValueError('Original model import root mismatch')
    from mindscape.r56.learning import load_checkpoint
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import matched_baseline,controlled_graph
    model=matched_baseline('gru') if condition=='matched-gru' else neural.R56Model()
    metadata=load_checkpoint(model,checkpoint,expected_hash);model.cpu().eval();torch.set_num_threads(1)
    def predictor(problem):
        with torch.no_grad():return model(controlled_graph(pack_problem(problem),condition),memory=None,stateless=False)
    result=evaluate(predictor,rows)
    result.update(checkpoint_sha256=expected_hash,checkpoint_metadata=metadata,source_root=str(root.resolve()),condition=condition,source='original-R56-neural-checkpoint',device='cpu')
    return result
