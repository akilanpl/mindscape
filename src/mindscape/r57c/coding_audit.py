"""Read-only coding shortcut diagnosis; diagnostic development data only."""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def diagnostic_rows(seed=2029):
    rng=random.Random(seed)
    families=[('ascii',['bravo','alpha','delta','charlie']),('unicode',['βήτα','άλφα','γάμμα','δέλτα']),
              ('prefix',['value','value_extra','value_extra_more','value_x']),('suffix',['foo_x','bar_x','foo_y','bar_y']),
              ('length',['q','long_identifier','medium_id','xy']),('misleading',['aa9','aa0','aa7','aa1'])]
    families += [('random'+str(i),['id_'+''.join(rng.choices('abcdefghjkmnpqrstuvxyz',k=7)) for _ in range(4)]) for i in range(4)]
    rows=[]
    for scope,names in families:
        for target in range(4):
            for shift in range(4):
                candidates=names[shift:]+names[:shift]
                p={'domain':'coding','operation':'rename','source':'def f(x):\n return x + 1','old_name':'x','candidates':candidates,'requested_name':names[target],'concept_ids':[3]}
                rows.append({'identity':digest(p),'family':scope,'scope':scope,'problem':p,'labels':{'coding_index':candidates.index(names[target])},'split':'diagnostic-dev'})
    return rows


def baselines(rows):
    result={}
    for name,choose in [('exact-request',lambda p:p['candidates'].index(p['requested_name'])),('lexmin',lambda p:min(range(4),key=lambda i:p['candidates'][i])),('lexmax',lambda p:max(range(4),key=lambda i:p['candidates'][i])),('shortest',lambda p:min(range(4),key=lambda i:len(p['candidates'][i]))),('longest',lambda p:max(range(4),key=lambda i:len(p['candidates'][i])))]:
        result[name]=sum(choose(r['problem'])==r['labels']['coding_index'] for r in rows)/len(rows)
    return result


def score(model,rows):
    import torch
    records=[]
    with torch.no_grad():
        for r in rows:
            problem=deepcopy(r['problem']);out=model.predict(problem);logits=out['coding_logits'].detach().cpu();index=int(logits.argmax())
            # Protected label poisoning changes no model input; exact replay checked.
            poisoned=deepcopy(r);poisoned['labels']={'coding_index':(r['labels']['coding_index']+1)%4}
            replay=model.predict(poisoned['problem'])['coding_logits'].detach().cpu()
            assert torch.equal(logits,replay)
            records.append({'identity':r['identity'],'family':r['family'],'requested_name':problem['requested_name'],'candidates':problem['candidates'],'target':r['labels']['coding_index'],'prediction':index,'winner':problem['candidates'][index],'correct':index==r['labels']['coding_index'],'logits':logits.tolist(),'label_poison_unchanged':True})
        for field in ('labels','target_name','coding_index'):
            bad=deepcopy(rows[0]['problem']);bad[field]=0
            try:model.prepare(bad)
            except ValueError:continue
            raise AssertionError('Protected field accepted: '+field)
    groups=defaultdict(list)
    for r in records:groups[r['family']].append(r)
    # Fixed name order, requested identity changes: count winner changes.
    counterfactual=[]
    for family in groups:
        chosen=[r for r in records if r['family']==family and r['candidates']==groups[family][0]['candidates']]
        counterfactual.append({'family':family,'target_classes':len({r['target'] for r in chosen}),'predicted_classes':len({r['prediction'] for r in chosen}),'all_requests_correct':all(r['correct'] for r in chosen)})
    return {'records':records,'accuracy':sum(r['correct'] for r in records)/len(records),'by_scope':{k:sum(r['correct'] for r in v)/len(v) for k,v in groups.items()},'counterfactual':counterfactual,'protected_fields_rejected':True}


def audit(output,old_output):
    import torch
    from mindscape.r57.model import make_model
    from mindscape.r56.learning import load_checkpoint
    output=Path(output);output.mkdir(parents=True,exist_ok=False);old_output=Path(old_output)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=diagnostic_rows();results=[]
    for seed in (17,29,43):
        for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
            root=old_output/f'checkpoints/{seed}-{condition}';receipt=json.loads((root/'result.json').read_text())
            for phase,key in [('initial','initial_sha256'),('final','checkpoint_sha256')]:
                path=root/f'{phase}.json';expected=receipt[key];assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
                model=make_model(condition,seed);load_checkpoint(model,path,expected);model.cpu().eval()
                value=score(model,rows);value.update(seed=seed,condition=condition,phase=phase,checkpoint_sha256=expected)
                (output/f'{seed}-{condition}-{phase}.json').write_text(json.dumps(value,indent=2));results.append({k:v for k,v in value.items() if k!='records'})
    # Frozen prior final is audited only for distribution shortcut, no new final created.
    distribution={}
    for seed in (17,29,43):
        prior=json.loads((old_output/f'data-v1/evaluator-only/final-{seed}.json').read_text());coding=[r for r in prior if r['problem']['domain']=='coding']
        distribution[str(seed)]={'count':len(coding),'requested_lexmin':sum(r['problem']['requested_name']==min(r['problem']['candidates']) for r in coding),'requested_positions':dict(Counter(r['labels']['coding_index'] for r in coding)),'requested_suffix_zero':sum(r['problem']['requested_name'].endswith('_0') for r in coding)}
    identity_counts={}
    for seed in (17,29,43):
        train=json.loads((old_output/f'data-v1/train/{seed}.json').read_text());prior=json.loads((old_output/f'data-v1/evaluator-only/final-{seed}.json').read_text())
        a={digest(r['problem']) for r in train if r['problem']['domain']=='coding'};b={digest(r['problem']) for r in prior if r['problem']['domain']=='coding'};c={digest(r['problem']) for r in rows}
        identity_counts[str(seed)]={'train_coding_unique':len(a),'old_final_coding_unique':len(b),'diagnostic_unique':len(c),'train_old_final_overlap':len(a&b),'train_diagnostic_overlap':len(a&c),'old_final_diagnostic_overlap':len(b&c)}
    summary={'identity_counts':identity_counts,'scope':'diagnostic development only; no new final or optimization','rows':len(rows),'rows_sha256':digest(rows),'baselines':baselines(rows),'old_final_distribution':distribution,'results':results,'optimizer_updates':0,'classification':'CONFIRMED evaluation-distribution shortcut if old requested_lexmin=count; model dependence quantified by counterfactual records; training cause unresolved'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2));(output/'diagnostic-dev.json').write_text(json.dumps(rows,indent=2))
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--old-output',required=True);a=p.parse_args();s=audit(a.output,a.old_output);print(json.dumps({'rows':s['rows'],'baselines':s['baselines'],'old_final_distribution':s['old_final_distribution'],'results':[{k:r[k] for k in ('seed','condition','phase','accuracy')} for r in s['results']]}))
if __name__=='__main__':main()
