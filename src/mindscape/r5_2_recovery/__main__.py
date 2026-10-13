"""Recover labels from authenticated frozen loop order, never from accuracy."""
import ast
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path
from collections import Counter
from mindscape.v3.schema import digest
from mindscape.r5_2_math.arithmetic import run
from mindscape.r5_2_math.verification import verify, context_key, Teacher
from mindscape.r5_2_math.analysis import analyze
from mindscape.r5_2_math.controls import Memory

CONDITIONS = ('recurrent', 'reset_hidden', 'memory_only', 'zero_weight', 'zero_weight_memory')
ROOT = Path('results/r5_2_math/native-one-campaign')
OUT = Path('experiments/r5_2_math/recovery_v1')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}

def reconstruct(records, tasks, seed):
    require(len(records) == len(CONDITIONS)*len(tasks), 'Incomplete prediction coverage')
    recovered = []
    for i, r in enumerate(records):
        task = tasks[i % len(tasks)]
        require(r['seed'] == seed and r['condition'] == 'carry', 'Unexpected original label/seed')
        require(all(r[k] == task[k] for k in ('identity','split','family')), 'Frozen iteration order mismatch')
        recovered.append({**r, 'condition': CONDITIONS[i // len(tasks)]})
    return recovered

def diagnostics(result, score, memory, held):
    rows=[]
    for e, expected in zip(result['transitions'],score['posthoc_transitions'],strict=True):
        key=context_key(e['state'])
        row={'key':key,'memory_seen':key in memory,'correct':expected['correct'],
             'withheld_supervision':tuple([key.split('|')[0],*map(int,key.split('|')[1:])]) in held}
        for name in ('digit','carry'):
            probabilities=e['prediction'].get(name+'_probabilities')
            if probabilities is not None:
                require(len(probabilities)==10 and all(math.isfinite(p) and 0<=p<=1 for p in probabilities)
                        and abs(sum(probabilities)-1)<1e-5,'Invalid probabilities')
                order=sorted(range(10),key=lambda v:(-probabilities[v],v))
                require(e['written_'+name]==order[0], 'Probability/write mismatch')
                row[name+'_rank']=order.index(expected['expected_'+name])+1
                row[name+'_top2_contains_reference']=expected['expected_'+name] in order[:2]
        rows.append(row)
    return rows

def replay_record(task,record,memory,held):
    original=record['result']; events=iter(original['transitions']); stages=iter(original['stages'])
    class Saved:
        def reset(self): pass
        def local(self,state):
            e=next(events); require(e['state']==state,'Local state mismatch')
            return e['written_digit'],e['written_carry'],e['prediction']
        def sign(self,a,b,op):
            s=next(stages)
            p=s['sign_prediction'].get('probabilities')
            if p is not None:
                require(len(p)==2 and all(math.isfinite(v) and 0<=v<=1 for v in p)
                        and abs(sum(p)-1)<1e-5 and s['sign']==max(range(2),key=lambda i:p[i]),'Sign probability mismatch')
            return s['sign'],s['sign_prediction']
    result=run(task,Saved()); require(result==original,'Written replay mismatch')
    require(next(events,None) is None and next(stages,None) is None,'Extra events')
    base=verify(task,result)
    saved=record['verification']
    require(set(saved)==set(base)|{'context_diagnostics'},'Verifier field set mismatch')
    require({k:saved[k] for k in base}==base,'Independent verifier mismatch')
    require(saved['context_diagnostics']==diagnostics(result,base,memory,held),'Diagnostic mismatch')
    return base

class Zero:
    def reset(self): pass
    def local(self,state): return 0,0,{'kind':'neural_unverified','digit_probabilities':[0.10000000149011612]*10,'carry_probabilities':[0.10000000149011612]*10}
    def sign(self,a,b,op): return 0,{'kind':'neural_unverified','probabilities':[0.5,0.5]}

def execute():
    before=inventory(ROOT); preservation=ROOT.with_suffix('.preservation.json'); archive=ROOT.with_suffix('.tar.gz')
    external={str(p):sha(p) for p in (preservation,archive)}
    receipt=load(preservation)
    require(before==receipt['files'] and sha(archive)==receipt['archive_sha256'],'Preservation hash mismatch')
    with tarfile.open(archive) as tar:
        restored={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
    require(restored==before,'Archive restore mismatch')
    freeze=load(ROOT/'freeze.json')
    for section in ('sources','historical_pins'):
        for name,h in freeze[section].items(): require(sha(Path(name))==h,'Frozen source/pin changed: '+name)
    manifest=load(ROOT/'manifest.json'); protocol=manifest['protocol']; tasks=manifest['tasks']
    require(digest(tasks)==manifest['task_hash'] and digest(protocol)==manifest['protocol_hash'],'Manifest digest')
    require(load(Path('configs/r5_2_math/tasks.json'))==manifest,'Committed manifest mismatch')
    source=ast.parse(Path('src/mindscape/r5_2_math/__main__.py').read_text())
    dictionaries=[n for n in ast.walk(source) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='conditions' for t in n.targets)]
    require(len(dictionaries)==1 and tuple(ast.literal_eval(k) for k in dictionaries[0].value.keys)==CONDITIONS,'Frozen conditions order')
    train=[t for t in tasks if t['split']=='train']; evaluation=[t for t in tasks if t['split']!='train']
    require(len(train)==768 and len(evaluation)==204 and len({t['identity'] for t in tasks})==len(tasks),'Task coverage')
    require(Counter(t['family'] for t in evaluation if t['split']=='ood')=={f:24 for f in protocol['families']},'OOD denominators')
    held={tuple(v) for v in protocol['held_local_contexts']}; memory={}; signs={}; labels=Counter(); signcounts=Counter(); exposure=0
    traces=load(ROOT/'training-demonstrations.json'); require(len(traces)==len(train),'Training trace count')
    for task,trace in zip(train,traces,strict=True):
        require(run(task,Teacher())==trace and verify(task,trace)['strict'],'Training reference mismatch')
        for e in trace['transitions']:
            s=e['state']; key=context_key(s); allowed=tuple([s['operation'],s['left'],s['right'],s['incoming']]) not in held
            exposure+=not allowed
            if allowed: labels[s['operation']]+=1; memory[key]=[e['written_digit'],e['written_carry']]
        current=task['operands'][0]
        for op,b,stage in zip(task['operations'],task['operands'][1:],trace['stages'],strict=True):
            signs[f'{current}|{b}|{op}']=stage['sign']; signcounts[str(stage['sign'])]+=1; current=int(stage['answer'])
    require(load(ROOT/'memory.json')=={'local':memory,'sign':signs},'Independent memory reconstruction')
    refs=load(ROOT/'reference.json'); require(len(refs)==204,'Reference coverage'); misses=total=0
    for task,r in zip(evaluation,refs,strict=True):
        require(task['identity']==r['identity'] and run(task,Teacher())==r['result'] and verify(task,r['result'])==r['verification'] and r['verification']['strict'],'Reference mismatch')
        if task['split']=='ood':
            total+=len(r['result']['transitions']); misses+=sum(context_key(e['state']) not in memory for e in r['result']['transitions'])
    q={'decision':'DATA_READY','training_local_labels':dict(labels),'sign_labels':dict(signcounts),'training_held_context_input_exposures':exposure,'ood_reference_contexts':total,'ood_memory_misses':misses,'ood_memory_miss_fraction':misses/total,'reference_exact_tasks':204}
    require(q==load(ROOT/'qualification.json'),'Qualification receipt mismatch')
    gates=protocol['qualification']; require(all(v>=gates['minimum_training_local_labels_per_operation'] for v in labels.values()) and signcounts['1']>=gates['minimum_sign_positive'] and signcounts['0']>=gates['minimum_sign_negative'] and misses/total>=gates['minimum_memory_local_miss_fraction_ood'],'Qualification gates')
    fits=load(ROOT/'fits.json')
    require({(f['kind'],f['seed']) for f in fits}=={(k,s) for k in protocol['architectures'] for s in protocol['optimizer_seeds']},'Exact six-fit coverage')
    from mindscape.r5_2_math.checkpoints import audit
    checkpoint=audit(ROOT/'checkpoints',fits,protocol)
    for f in fits:
        lines=[json.loads(v) for v in (ROOT/'checkpoints'/f"{f['kind']}-s{f['seed']}-progress.jsonl").read_text().splitlines()]
        require([v['update'] for v in lines]==list(range(1,801)) and [v['loss'] for v in lines]==f['losses'],'Progress/update mismatch')
    records=[]; mapping=[]
    for seed in protocol['optimizer_seeds']:
        raw=load(ROOT/f'predictions-s{seed}.json')
        try: analyze(raw,protocol)
        except ZeroDivisionError: pass
        else: raise ValueError('Original failure not reproduced')
        recovered=reconstruct(raw,evaluation,seed)
        for i,r in enumerate(recovered):
            task=evaluation[i%204]; replay_record(task,r,memory,held)
            c=r['condition']; result=r['result']
            if c in ('memory_only','zero_weight','zero_weight_memory'):
                predictor=Memory(memory,signs) if c=='memory_only' else Zero() if c=='zero_weight' else Memory(memory,signs,Zero())
                require(run(task,predictor)==result,'Independent deterministic control mismatch')
            else:
                require(all(e['prediction']['kind']=='neural_unverified' for e in result['transitions']),'Neural condition diagnostic mismatch')
            mapping.append({'file':f'predictions-s{seed}.json','index':i,'identity':r['identity'],'original':'carry','recovered':c})
        records.extend(recovered)
    summary=analyze(records,protocol)
    # Independently count answers and local transitions, avoiding analyzer verifier counters.
    for c in CONDITIONS:
        for seed in protocol['optimizer_seeds']:
            rr=[r for r in records if r['condition']==c and r['seed']==seed and r['split']=='ood']
            independently=sum(int(r['result']['answer'])==int(r['verification']['expected']) for r in rr)
            require(summary['metrics'][c][str(seed)]['ood']['correct']==independently and len(rr)==144,'Independent aggregate mismatch')
            local=sum(e['written_digit']==v['expected_digit'] and e['written_carry']==v['expected_carry'] for r in rr for e,v in zip(r['result']['transitions'],r['verification']['posthoc_transitions'],strict=True))
            n=sum(len(r['result']['transitions']) for r in rr)
            require(summary['metrics'][c][str(seed)]['ood']['local_accuracy']==local/n,'Independent local aggregate mismatch')
    for condition in CONDITIONS:
        for seed in protocol['optimizer_seeds']:
            for family, metric in summary['metrics'][condition][str(seed)].items():
                group=[r for r in records if r['condition']==condition and r['seed']==seed and
                       (r['split']=='ood' if family=='ood' else r['split']=='development' if family=='development' else r['family']==family)]
                correct=sum(int(r['result']['answer'])==int(r['verification']['expected']) for r in group)
                require(metric['tasks']==len(group) and metric['correct']==correct and metric['exact_accuracy']==correct/len(group),'Independent family metric mismatch')
    mean=lambda c,f: statistics.mean(summary['metrics'][c][str(s)][f]['exact_accuracy'] for s in protocol['optimizer_seeds'])
    g=protocol['gate']
    independent={
        'overall':mean('recurrent','ood')>=g['minimum_mean_ood_exact'],
        'length':mean('recurrent','length')>=g['minimum_length_exact'],
        'carry_borrow':mean('recurrent','carry_borrow')>=g['minimum_carry_borrow_exact'],
        'sign':mean('recurrent','sign')>=g['minimum_sign_exact'],
        'unseen_context':mean('recurrent','unseen_local_context')>=g['minimum_unseen_context_exact'],
        'intermediate':statistics.mean(summary['metrics']['recurrent'][str(s)]['ood']['local_accuracy'] for s in protocol['optimizer_seeds'])>=g['minimum_intermediate_next_state_accuracy'],
        'reset_gain':mean('recurrent','ood')-mean('reset_hidden','ood')>=g['minimum_ood_gain_over_reset'],
        'memory_controls':all(mean('recurrent','ood')-mean(c,'ood')>=g['minimum_ood_gain_over_memory_and_zero_weight_memory'] for c in ('memory_only','zero_weight_memory')),
        'seed_reset_gains':all(summary['metrics']['recurrent'][str(s)]['ood']['correct']>summary['metrics']['reset_hidden'][str(s)]['ood']['correct'] for s in protocol['optimizer_seeds'])}
    require(independent==summary['gates'],'Independent frozen gate mismatch')
    require(inventory(ROOT)==before and all(sha(Path(p))==h for p,h in external.items()),'Original evidence mutated')
    summary.update(provenance='POST-HOC RECOVERY FROM FROZEN PREDICTIONS',historical_decision='FAILED_CAMPAIGN',neural_re_forward=False,additional_optimizer_steps=0)
    evidence={'status':'PASS','original_files':before,'external_files':external,'checkpoints':checkpoint,'fits':6,'optimizer_updates':4800,'prediction_records':len(records),'independent_tasks':204,'independent_ood_tasks':144,'qualification':q,'original_bytes_unchanged':True,'reference_correct':204,'limitations':['No neural re-forward: checkpoint-to-prediction relationship is archived provenance, not regenerated output.','Neural condition attribution follows authenticated frozen insertion order; records lack intrinsic neural condition identifiers.','24 independent tasks per OOD family; three seeds do not triple task denominator.']}
    for name,value in [('summary.json',summary),('audit.json',evidence),('label-map.json',mapping)]:
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'decision':summary['decision'],'gates':summary['gates'],'records':len(records)}))

if __name__=='__main__':
    try: execute()
    except Exception as e:
        (OUT/'inconclusive.json').write_text(json.dumps({'decision':'INCONCLUSIVE','error':str(e),'historical_decision':'FAILED_CAMPAIGN'})+'\n')
        raise
