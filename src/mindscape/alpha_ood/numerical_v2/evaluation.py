"""Full raw trajectory seal, exact replay, and independently released scoring."""
from pathlib import Path
import json
from ..benchmark import digest, FAMILIES, memorization, transition_lookup
from ..scoring import replay
from .custody import require_access, verify_signed
from .adapter import native_task

CONDITIONS=('recurrent','feedforward','memorization','transition_lookup','procedural')
SEALED_CONDITIONS=CONDITIONS[:-1]
SEEDS=(11,23,47)


def training_baselines(tasks, traces):
    """Training-only operation-state controls; no OOD target access."""
    if len(tasks)!=len(traces) or any(t.split!='train' for t in tasks):
        raise ValueError('train-only baselines')
    memory={}; table={}
    for t,trace in zip(tasks,traces,strict=True):
        states=tuple(int(s['answer']) for s in trace['stages'])
        if len(states)!=len(t.steps):
            raise ValueError('trace stage coverage')
        memory[t.identity]=states; current=t.start
        for (op,right),state in zip(t.steps,states,strict=True):
            table[current,op,right]=state; current=state
    return memory,table


def baseline_row(task,condition,seed,memory,table):
    if condition not in ('memorization','transition_lookup') or seed not in SEEDS:
        raise ValueError('baseline specification')
    states=memorization(task,memory) if condition=='memorization' else transition_lookup(task,table)
    return {'identity':task.identity,'condition':condition,'seed':seed,
            'result':{'answer':str(states[-1]),'stages':[{'answer':str(s)} for s in states],
                      'transitions':[],'baseline_mechanism':condition},'failure':None}


def expected_order(tasks):
    return [(c,s,t.identity) for c in SEALED_CONDITIONS for s in SEEDS for t in tasks]


def validate_records(tasks,rows):
    expected=expected_order(tasks)
    if len(rows)!=len(expected):
        raise ValueError('all conditions/seeds/tasks required; failed records retained')
    for row,key in zip(rows,expected,strict=True):
        if set(row)!= {'identity','condition','seed','result','failure'} or (row['condition'],row['seed'],row['identity'])!=key:
            raise ValueError('reordered/missing/substituted prediction')
        if (row['result'] is None)==(row['failure'] is None):
            raise ValueError('exactly one result/failure required')
        if row['result'] is not None:
            result=row['result']
            if not isinstance(result,dict) or not isinstance(result.get('transitions'),list) or not isinstance(result.get('stages'),list):
                raise ValueError('full raw trajectory required')


def seal_predictions(tasks,rows,binding,path:Path):
    validate_records(tasks,rows)
    seal={'version':'alpha-ood-numerical-v2','binding':binding,'prediction_hash':digest(rows),
          'ordered_identity_hash':digest(expected_order(tasks)), 'labels_released':False,
          'rows':rows,'scientific_status':'FROZEN_PREDICTIONS_SEALED_UNVERIFIED'}
    with path.open('x') as stream:
        json.dump(seal,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return seal


def replay_neural(task,result):
    """Independent scheduler verifies raw predicted feedback; no target use."""
    from mindscape.r5_2_math.arithmetic import run
    class Saved:
        def __init__(self): self.event=0; self.stage=0
        def reset(self): pass
        def local(self,state):
            event=result['transitions'][self.event]; self.event+=1
            if event['state']!=state: raise ValueError('predicted-feedback state mismatch')
            return event['written_digit'],event['written_carry'],event['prediction']
        def sign(self,a,b,op):
            stage=result['stages'][self.stage]; self.stage+=1
            return stage['sign'],stage['sign_prediction']
    saved=Saved(); reconstructed=run(native_task(task),saved)
    if saved.event!=len(result['transitions']) or saved.stage!=len(result['stages']) or reconstructed!=result:
        raise ValueError('trajectory replay mismatch')


def _score_released(*,tasks,seal,labels,release_receipt,access,binding,trusted_keys):
    require_access(access,'evaluate',binding)
    if seal['binding']!=binding or seal['labels_released'] is not False or seal['prediction_hash']!=digest(seal['rows']):
        raise ValueError('seal content mismatch')
    if seal['ordered_identity_hash']!=digest(expected_order(tasks)) or digest(labels)!=binding['label_commitment']:
        raise ValueError('identity/label commitment mismatch')
    release_binding={**binding,'prediction_hash':seal['prediction_hash'],'ordered_identity_hash':seal['ordered_identity_hash']}
    release=verify_signed(release_receipt,'seal_before_label_release',release_binding,trusted_keys)
    eval_receipt=dict(access.receipts)['sealed_final_evaluation_authorization']['payload']
    if release['sequence']<=eval_receipt['sequence']:
        raise ValueError('independent custody event order')
    validate_records(tasks,seal['rows'])
    task_map={t.identity:t for t in tasks}; compact=[]
    for row in seal['rows']:
        task=task_map[row['identity']]; result=row['result']
        if result is None:
            # Explicit failure produces incorrect states while keeping denominator.
            values=[v+1 for v in labels[task.identity]]
        else:
            if row['condition'] in ('recurrent','feedforward'):
                replay_neural(task,result)
            if len(result['stages'])!=len(task.steps): raise ValueError('stage count')
            if any(not isinstance(stage.get('answer'),str) or str(int(stage['answer']))!=stage['answer'] for stage in result['stages']):
                raise ValueError('noncanonical operation state')
            values=[int(s['answer']) for s in result['stages']]
            if result['answer']!=str(values[-1]): raise ValueError('answer/stages mismatch')
        compact.append({'identity':task.identity,'condition':row['condition'],'seed':row['seed'],'states':values})
    # Procedural reference is constructed ONLY after independently verified label
    # release. It is not a preseal prediction and never enters learned execution.
    for seed in SEEDS:
        compact.extend({'identity':t.identity,'condition':'procedural','seed':seed,'states':labels[t.identity]} for t in tasks)
    report=replay(tuple(tasks),compact,labels,{'prediction_hash':digest(compact),
                    'label_commitment':digest(labels),'custodian_attested_precedence':True},CONDITIONS,SEEDS)
    metrics=report['metrics']; comparisons=report['comparisons']
    mean=lambda condition,family,key:sum(metrics[condition][str(s)][family][key] for s in SEEDS)/3
    gates={'mean_full_task':mean('recurrent','overall','exact')>=.80,
           'families':all(mean('recurrent',f,'exact')>=.70 for f in FAMILIES),
           'intermediate':mean('recurrent','overall','intermediate_accuracy')>=.90,
           'absolute_gains':all(comparisons[b+':'+family]['mean_gain']>=.10 for b in CONDITIONS if b not in ('recurrent','procedural') for family in ('overall',*FAMILIES)),
           'each_seed_feedforward':all(metrics['recurrent'][str(s)]['overall']['exact']>metrics['feedforward'][str(s)]['overall']['exact'] for s in SEEDS),
           'bootstrap':all(comparisons[b+':'+family]['paired_task_bootstrap95'][0]>0 for b in CONDITIONS if b not in ('recurrent','procedural') for family in ('overall',*FAMILIES)),
           'procedural_reference':all(metrics['procedural'][str(s)]['overall']['exact']==1 for s in SEEDS)}
    return {**report,'gates':gates,'scientific_status':'GATES_PASS_PENDING_INDEPENDENT_AUDIT' if all(gates.values()) else 'NO_GO_UNVERIFIED',
            'raw_prediction_hash':seal['prediction_hash'],'raw_rows':seal['rows']}


def score_released(**kwargs):
    """Scientific boundary: exactly the proposed sealed OOD task denominator."""
    from collections import Counter
    tasks=kwargs['tasks']
    if len(tasks)!=216 or any(t.split!='ood' for t in tasks) or Counter(t.family for t in tasks)!=dict.fromkeys(FAMILIES,36):
        raise ValueError('scientific scoring requires exactly216 OOD tasks, six families of36')
    return _score_released(**kwargs)


def independently_audited_verdict(report,binding,audit_receipt,trusted_keys):
    """Separate external metric audit, never self-issued by numerical scorer."""
    audit_binding={**binding,'report_hash':digest(report),'prediction_hash':report['raw_prediction_hash']}
    verify_signed(audit_receipt,'independent_metric_audit',audit_binding,trusted_keys)
    passed=all(report['gates'].values())
    return {'report_hash':digest(report),'scientific_status':'INDEPENDENT_SCIENTIFIC_PASS' if passed else 'INDEPENDENT_SCIENTIFIC_FAIL',
            'release_status':'NO_RELEASE_AUTHORIZATION'}
