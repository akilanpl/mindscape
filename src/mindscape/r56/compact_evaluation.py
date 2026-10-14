"""Read-only compact checkpoint evaluator; never fits, repairs or qualifies weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .curriculum import digest

STATUS='INDEPENDENT EVALUATION RECORD ONLY; SCIENTIFIC QUALIFICATION NOT ISSUED'
FORWARD_POLICY='stateless-false-reset-memory-per-independent-record'


def build_protected_evaluation(training_rows,dev,test):
    """Evaluator-side construction; protected references never enter training handoff."""
    from .qualification import heldout_suite
    # Historical suite enumerator reads carry_in. Normalize original observations,
    # never alter the already frozen training examples or answer labels.
    canonical=[]
    for row in training_rows:
        copied={**row,'problem':dict(row['problem'])}
        if copied['problem']['domain']=='math':
            copied['problem']['carry_in']=copied['problem'].get('incoming',copied['problem'].get('carry_in',0))
        canonical.append(copied)
    suite,references=heldout_suite(canonical)
    repeats=[]
    for row in suite['single_transitions']:
        if row['problem']['domain']=='physics':
            problem={**row['problem'],'repeat_steps':4}
            identity=digest(problem)
            repeats.append({'axis':'repeated-physics-force','identity':identity,'problem':problem})
            p=problem
            expected=[v+4*f*p['dt']/p['mass'] for v,f in zip(p['velocity'],p['force'])]
            references[identity]={'velocity':expected}
    suite['repeated_physics']=repeats
    return {'dev':dev,'test':test,'heldout_suite':[suite,references]}


def _local_prediction(model,problem,condition,device='cpu'):
    from .native_domains import pack_problem
    from .benchmarks import controlled_graph
    import torch
    with torch.no_grad():
        out=model(controlled_graph(pack_problem(problem,device),condition),memory=None,stateless=False)
    domain=problem['domain']
    if domain=='math':
        return {'digit':int(out['math_digit_logits'].argmax()),'outgoing':int(out['math_carry_logits'].argmax())}
    if domain=='coding':
        count=len(problem['candidates']); choice=int(out['coding_logits'][:count].argmax())
        return {'candidate_index':choice,'name':problem['candidates'][choice]}
    delta=out['physics_delta'].detach().cpu().tolist()
    return {'delta':delta,'velocity':[v+d for v,d in zip(problem['velocity'],delta)]}


def compose_addition(predictor,problem):
    """Supplied LSD schedule; only original predicted carry feeds the next query."""
    left,right=problem['left'],problem['right']
    if not all(isinstance(v,str) and 1<=len(v)<=128 and v.isascii() and v.isdecimal() for v in (left,right)):
        raise ValueError('Bounded unsigned decimal strings required')
    carry=0; digits=[]; transitions=[]
    for index in range(max(len(left),len(right))):
        local={'domain':'math','operation':'add_digit','left':int(left[-index-1]) if index<len(left) else 0,
               'right':int(right[-index-1]) if index<len(right) else 0,'incoming':carry,'concept_ids':problem.get('concept_ids',[0])}
        prediction=predictor(local)
        if type(prediction.get('digit'))is not int or not 0<=prediction['digit']<=9 or prediction.get('outgoing') not in (0,1):
            raise ValueError('Invalid original predicted digit/carry')
        transitions.append({'observation':local,'original_prediction':dict(prediction)})
        digits.append(str(prediction['digit'])); carry=prediction['outgoing']
    if carry:digits.append(str(carry))
    return {'answer':''.join(reversed(digits)).lstrip('0') or '0','transitions':transitions,
            'controller':'supplied LSD traversal; memory reset per local query; own predicted carries; no correction'}


def repeat_physics(predictor,problem):
    steps=problem['repeat_steps']
    if type(steps)is not int or not 1<=steps<=32:raise ValueError('Repeated physics steps must be 1..32')
    current={k:v for k,v in problem.items() if k!='repeat_steps'}; transitions=[]
    for _ in range(steps):
        prediction=predictor(current)
        transitions.append({'observation':dict(current),'original_prediction':prediction})
        current={**current,'velocity':list(prediction['velocity'])}
    return {'velocity':current['velocity'],'transitions':transitions,'controller':'original predicted velocity feedback; no correction'}


def evaluate_records(predictor,protected):
    """Pure read-only bookkeeping; predictor can be a labelled engineering fixture."""
    records=[]
    for split in ('dev','test'):
        for row in protected[split]:
            prediction=predictor(row['problem']); labels=row['labels']; domain=row['problem']['domain']
            expected=({'digit':labels['math_digit'],'outgoing':labels['math_carry']} if domain=='math' else
                      {'candidate_index':labels['coding_index']} if domain=='coding' else {'delta':labels['physics_delta']})
            records.append(_score(row.get('identity',digest(row['problem'])),split,domain,prediction,expected))
    suite,references=protected['heldout_suite']
    for row in suite['single_transitions']:
        records.append(_score(row['identity'],row['axis'],row['problem']['domain'],predictor(row['problem']),references[row['identity']]))
    for row in suite['compositions']:
        records.append(_score(row['identity'],row['axis'],'math',compose_addition(predictor,row['problem']),references[row['identity']]))
    for row in suite.get('repeated_physics',[]):
        records.append(_score(row['identity'],row['axis'],'physics',repeat_physics(predictor,row['problem']),references[row['identity']]))
    summaries={}
    for record in records:
        key=record['axis']+':'+record['metric_name']; group=summaries.setdefault(key,{'count':0,'metric_sum':0.})
        group['count']+=1;group['metric_sum']+=record['metric']
    return {'records':records,'summaries':summaries,'status':STATUS,'scientifically_qualified':False,
            'selection':'final immutable checkpoint only; no feedback to trainer',
            'composition_policy':'reset hidden state every digit, preserve model carry feedback; differs from production infer_problem recurrent composition; future integration needs reviewed state-policy matching'}


def _score(identity,axis,domain,prediction,expected):
    if domain=='physics':
        key='velocity' if 'velocity' in expected else 'delta'
        metric=sum((float(a)-float(b))**2 for a,b in zip(prediction[key],expected[key]))/len(expected[key]); name='mean-squared-error'
    else:metric=float(all(prediction.get(key)==value for key,value in expected.items()));name='exact-accuracy'
    return {'identity':identity,'axis':axis,'domain':domain,'original_prediction':prediction,'protected_reference':expected,
            'metric':metric,'metric_name':name,'repaired':False}


def evaluate_checkpoint(root,manifest_path,protected_path,checkpoint_path,expected_checkpoint_hash,seed,condition):
    """Actual evaluator entrypoint: immutable checkpoint/hash/source/condition checks."""
    from .compact_gpu import CONDITIONS,SCOPE,read_json,file_hash
    from .learning import source_locks,load_checkpoint
    manifest=read_json(manifest_path); unsigned=dict(manifest);identity=unsigned.pop('package_sha256',None)
    if digest(unsigned)!=identity or manifest.get('scope')!=SCOPE:raise ValueError('Compact manifest hash/scope differs')
    if source_locks(root)!=manifest.get('source_locks'):raise ValueError('Frozen scientific source differs')
    if seed not in manifest['seeds'] or condition not in CONDITIONS:raise ValueError('Unknown fit condition/seed')
    if file_hash(protected_path)!=manifest['evaluation_references_sha256'][str(seed)]:raise ValueError('Protected evaluation hash differs')
    path=Path(checkpoint_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>64_000_000:raise ValueError('Unsafe or oversized checkpoint')
    if file_hash(path)!=expected_checkpoint_hash:raise ValueError('Immutable checkpoint hash differs')
    import torch
    from .neural import R56Model
    from .benchmarks import matched_baseline
    model=matched_baseline('gru',44962,seed) if condition=='matched-gru' else R56Model()
    metadata=load_checkpoint(model,path,expected_checkpoint_hash)
    from dataclasses import asdict
    from .neural import ModelConfig
    expected_config={'kind':'matched-gru','target_parameters':44962} if condition=='matched-gru' else asdict(ModelConfig())
    if metadata.get('model_config')!=expected_config or metadata.get('source_locks')!=manifest['source_locks']:
        raise ValueError('Checkpoint architecture/source provenance differs')
    if metadata.get('package_sha256')!=identity or metadata.get('seed')!=seed or metadata.get('condition')!=condition:
        raise ValueError('Checkpoint attribution differs')
    if metadata.get('optimizer_updates')!=2000 or metadata.get('forward_policy')!=FORWARD_POLICY:
        raise ValueError('Checkpoint is not frozen final fit or forward policy differs')
    model.cpu().eval()
    protected=read_json(protected_path)
    with torch.no_grad():report=evaluate_records(lambda p:_local_prediction(model,p,condition),protected)
    return {**report,'package_sha256':identity,'checkpoint_sha256':expected_checkpoint_hash,
            'protected_evaluation_sha256':file_hash(protected_path),'source_locks':manifest['source_locks'],
            'seed':seed,'condition':condition,'device':'cpu','forward_policy':FORWARD_POLICY,
            'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_metadata':metadata}


def main():
    parser=argparse.ArgumentParser(description='Read-only compact immutable-checkpoint evaluator')
    for name in ('root','manifest','protected','checkpoint','checkpoint-sha256','condition','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    report=evaluate_checkpoint(args.root,args.manifest,args.protected,args.checkpoint,args.checkpoint_sha256,args.seed,args.condition)
    with Path(args.output).open('x') as stream:json.dump(report,stream,sort_keys=True,allow_nan=False)


if __name__=='__main__':main()
