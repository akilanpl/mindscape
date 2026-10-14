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
    for event in result['transitions']:
        if any(type(event.get(name)) is not int or not 0<=event[name]<=9 for name in ('written_digit','written_carry')):
            raise ValueError('strict integer digit/carry vocabulary')
        state=event['state']
        if any(type(state.get(name)) is not int for name in ('left','right','incoming','index')):
            raise ValueError('strict integer observable state')
    if any(type(stage.get('sign')) is not int or stage['sign'] not in (0,1) for stage in result['stages']):
        raise ValueError('strict integer sign vocabulary')
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
    # Execute a separate arithmetic reference ONLY after signed label release.
    # Checking against committed labels is meaningful: labels are not copied
    # into the procedural condition, and corrupt custodian traces fail closed.
    from ..benchmark import reference
    procedural_states={task.identity:list(reference(task)) for task in tasks}
    if labels!=procedural_states:
        raise ValueError('released label trace disagrees with executed procedural reference')
    for seed in SEEDS:
        compact.extend({'identity':t.identity,'condition':'procedural','seed':seed,'states':procedural_states[t.identity]} for t in tasks)
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
    require_access(kwargs['access'],'evaluate',kwargs['binding'])
    from .boundary import verify_boundary
    verify_boundary(kwargs.pop('source_manifest',None),kwargs.pop('runtime_policy',None),kwargs['binding'])
    return _score_released(**kwargs)


GATE_KEYS=frozenset({'mean_full_task','families','intermediate','absolute_gains',
    'each_seed_feedforward','bootstrap','procedural_reference'})


def validate_audit_report(report):
    """A signed malformed report is still invalid; no vacuous/truthy pass."""
    import math
    gates=report.get('gates')
    if not isinstance(gates,dict) or set(gates)!=GATE_KEYS or any(type(value) is not bool for value in gates.values()):
        raise ValueError('complete exact-boolean seven-gate report required')
    passed=all(gates.values())
    status='GATES_PASS_PENDING_INDEPENDENT_AUDIT' if passed else 'NO_GO_UNVERIFIED'
    if report.get('scientific_status')!=status or type(report.get('unique_tasks')) is not int or report['unique_tasks']!=216:
        raise ValueError('scientific status/task denominator mismatch')
    metrics=report.get('metrics')
    if not isinstance(metrics,dict) or set(metrics)!=set(CONDITIONS):
        raise ValueError('complete audited condition set required')
    def finite_tree(value):
        if isinstance(value,dict):
            for child in value.values(): finite_tree(child)
        elif isinstance(value,(list,tuple)):
            for child in value: finite_tree(child)
        elif type(value) in (int,float) and not math.isfinite(value):
            raise ValueError('nonfinite metric')
    for condition in CONDITIONS:
        seeds=metrics[condition]
        if not isinstance(seeds,dict) or set(seeds)!={str(seed) for seed in SEEDS}:
            raise ValueError('complete audited seed set required')
        for seed in SEEDS:
            families=seeds[str(seed)]
            if not isinstance(families,dict) or set(families)!={'overall',*FAMILIES}:
                raise ValueError('complete audited family set required')
            for family in ('overall',*FAMILIES):
                metric=families[family]; total=216 if family=='overall' else 36
                if not isinstance(metric,dict) or type(metric.get('independent_tasks')) is not int or metric['independent_tasks']!=total:
                    raise ValueError('audited task denominator mismatch')
                correct=metric.get('exact_correct'); exact=metric.get('exact')
                if type(correct) is not int or not 0<=correct<=total or type(exact) not in (int,float) or exact!=correct/total:
                    raise ValueError('audited exact metric mismatch')
                if metric.get('full_trace_exact')!=exact:
                    raise ValueError('full trace metric mismatch')
                for key in ('full_trace_exact','final_answer_exact','intermediate_accuracy'):
                    if type(metric.get(key)) not in (int,float) or not math.isfinite(metric[key]) or not 0<=metric[key]<=1:
                        raise ValueError('invalid audited accuracy')
                numerator=metric.get('intermediate_correct'); denominator=metric.get('intermediate_denominator')
                if type(numerator) is not int or type(denominator) is not int or denominator<total or not 0<=numerator<=denominator or metric['intermediate_accuracy']!=numerator/denominator:
                    raise ValueError('intermediate metric denominator mismatch')
                finite_tree(metric)
    comparisons=report.get('comparisons')
    required_comparisons={baseline+':'+family for baseline in ('feedforward','memorization','transition_lookup') for family in ('overall',*FAMILIES)}
    if not isinstance(comparisons,dict) or set(comparisons)!=required_comparisons:
        raise ValueError('complete audited comparisons required')
    for key,comparison in comparisons.items():
        denominator=216 if key.endswith(':overall') else 36
        if type(comparison.get('independent_tasks')) is not int or comparison['independent_tasks']!=denominator:
            raise ValueError('comparison task denominator mismatch')
        interval=comparison.get('paired_task_bootstrap95')
        if type(comparison.get('mean_gain')) not in (int,float) or not -1<=comparison['mean_gain']<=1 or not isinstance(interval,(list,tuple)) or len(interval)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in interval) or not -1<=interval[0]<=interval[1]<=1:
            raise ValueError('invalid audited comparison')
    finite_tree(comparisons)
    rows=report.get('raw_rows')
    if not isinstance(rows,list) or len(rows)!=216*3*4 or digest(rows)!=report.get('raw_prediction_hash'):
        raise ValueError('audited complete raw prediction hash mismatch')
    identities=[row.get('identity') for row in rows[:216]]
    if len(set(identities))!=216:
        raise ValueError('raw prediction identity denominator mismatch')
    expected=[(condition,seed,identity) for condition in SEALED_CONDITIONS for seed in SEEDS for identity in identities]
    if any((row.get('condition'),row.get('seed'),row.get('identity'))!=key for row,key in zip(rows,expected,strict=True)):
        raise ValueError('raw prediction coverage/order mismatch')
    return passed


def independently_audited_verdict(report,binding,audit_receipt,trusted_keys,*,source_manifest=None,runtime_policy=None):
    """Separate external metric audit, never self-issued by numerical scorer."""
    passed=validate_audit_report(report)
    audit_binding={**binding,'report_hash':digest(report),'prediction_hash':report['raw_prediction_hash']}
    verify_signed(audit_receipt,'independent_metric_audit',audit_binding,trusted_keys)
    from .boundary import verify_boundary
    verify_boundary(source_manifest,runtime_policy,binding)
    return {'report_hash':digest(report),'scientific_status':'INDEPENDENT_SCIENTIFIC_PASS' if passed else 'INDEPENDENT_SCIENTIFIC_FAIL',
            'release_status':'NO_RELEASE_AUTHORIZATION'}
