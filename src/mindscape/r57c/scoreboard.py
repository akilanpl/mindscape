"""Independent focused-correction final evaluator; all fits verified before final reads."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
from mindscape.r57.evaluation import evaluate,analytic,fit_physics_baselines,physics_monotonicity
from mindscape.r57.scoreboard import sha,write,verify_freeze,original_predictor,legacy_path,invariance,majority_predictor,compare

SEEDS=(17,29,43)
CONDITIONS=('graph-concept-on','graph-concept-off','matched-gru')


def precheck(output,completed_check=None):
    """No final file decoded until all nine real fits/checkpoint/loss receipts pass."""
    if completed_check is None:
        from .runner import completed as completed_check
    receipts=[]
    for seed in SEEDS:
        for condition in CONDITIONS:receipts.append(completed_check(output/f'checkpoints/{seed}-{condition}'))
    frozen=verify_freeze(output)
    snapshot=json.loads((output/'preservation-snapshot.json').read_text())
    for name,digest in snapshot['sha256'].items():
        if sha(name)!=digest:raise PermissionError('Preserved earlier evidence changed: '+name)
    return frozen,receipts


def predictor(version,condition,seed,path=None,expected=None):
    import torch
    from mindscape.r56.learning import load_checkpoint
    if version=='R57':from mindscape.r57.model import make_model
    else:from .ordinal_model import make_model
    model=make_model(condition,seed)
    if path:load_checkpoint(model,path,expected)
    model.cpu().eval()
    def predict(p):
        with torch.no_grad():return model.predict(p)
    return predict


def coding_counterfactual(predict,rows):
    import ast
    from mindscape.r57.evaluation import decode
    records=[]
    for row in rows:
        p=row['problem']
        if p['domain']!='coding':continue
        original=decode(predict(deepcopy(p)),p)['coding_index'];changed=deepcopy(p);oldtarget=p['candidates'].index(p['requested_name']);nexttarget=(oldtarget+1)%len(p['candidates']);changed['requested_name']=p['candidates'][nexttarget]
        switched=decode(predict(changed),changed)['coding_index']
        renamed=deepcopy(p);old=p['old_name'];new='audit_bound_identifier'
        class Rename(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id==old:n.id=new
                return n
            def visit_arg(self,n):
                if n.arg==old:n.arg=new
                return n
        renamed['source']=ast.unparse(ast.fix_missing_locations(Rename().visit(ast.parse(p['source']))));renamed['old_name']=new
        alpha=decode(predict(renamed),renamed)['coding_index']
        records.append({'identity':row['identity'],'initial_target':oldtarget,'changed_target':nexttarget,'initial_prediction':original,'changed_prediction':switched,'both_requests_correct':original==oldtarget and switched==nexttarget,'prediction_changed':original!=switched,'source_alpha_prediction':alpha,'source_alpha_invariant':original==alpha})
    return records


def breakdown(rows,records):
    byid={r['identity']:r for r in records};groups=defaultdict(list)
    for row in rows:
        p=row['problem'];d=p['domain'];record=byid[row['identity']]
        if d=='math':key=f"math/{p['operation']}/carry{row['labels']['math_carry']}"
        elif d=='physics':key='physics/'+str(row.get('scope',row.get('family','unspecified')))
        else:key='coding/'+str(row.get('scope',row.get('family','unspecified')))
        groups[key].append(record)
    return {k:{'count':len(v),'metric':'mse' if k.startswith('physics') else 'accuracy','value':sum(r['mse'] if k.startswith('physics') else r['correct'] for r in v)/len(v)} for k,v in groups.items()}


def run(output):
    import torch
    output=Path(output).resolve();frozen,receipts=precheck(output)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    destination=output/'scoreboard-final';destination.mkdir(exist_ok=False);started=time.monotonic();summary=[];paired={}
    original_source=Path(frozen['original_source']).resolve();legacy=Path(frozen['legacy_root']);prior=Path(frozen['r57_root']);priorfreeze=json.loads((prior/'candidate-freeze.json').read_text());original_snapshot=json.loads((prior/'baseline-snapshot.json').read_text())
    for seed in SEEDS:
        rows=json.loads((output/f'data-v1/evaluator-only/final-{seed}.json').read_text());training=json.loads((output/f'data-v1/train/{seed}.json').read_text());physics=[r for r in rows if r['problem']['domain']=='physics']
        fitted,metadata=fit_physics_baselines(training);baselines={'analytic':evaluate(analytic,rows),'training-majority':evaluate(majority_predictor(training),rows)}
        for name in ('zero','constant','linear','analytic'):baselines['physics-'+name]=evaluate(lambda p,n=name:fitted(n,p),physics)
        for name,choose in [('lexmin',min),('lexmax',max),('shortest',lambda names:min(names,key=len)),('longest',lambda names:max(names,key=len))]:
            coding=[r for r in rows if r['problem']['domain']=='coding'];baselines['coding-'+name]=evaluate(lambda p,c=choose:{'coding_index':p['candidates'].index(c(p['candidates']))},coding)
        write(destination/f'baselines-{seed}.json',{'results':baselines,'fit_metadata':metadata,'source':'deterministic_rule_or_training_fitted; not neural'})
        for condition in CONDITIONS:
            runs={}
            for version in ('R56','R57','R57c'):
                for phase in ('initial','final'):
                    if version=='R56':
                        old=legacy_path(legacy,seed,condition);ckpt=old/'checkpoint.json';oldresult=json.loads((old/'result.json').read_text());expected=sha(ckpt)
                        if expected!=original_snapshot['sha256'].get(str(ckpt)) or expected!=oldresult['checkpoint_sha256']:raise PermissionError('Original R56 checkpoint changed')
                        predict,_=original_predictor(original_source,condition,seed,ckpt if phase=='final' else None,expected if phase=='final' else None)
                    else:
                        root=prior if version=='R57' else output;fit=root/f'checkpoints/{seed}-{condition}';receipt=json.loads((fit/'result.json').read_text());key='initial_sha256' if phase=='initial' else 'checkpoint_sha256';ckpt=fit/f'{phase}.json';expected=receipt[key]
                        if sha(ckpt)!=expected:raise PermissionError('Checkpoint changed')
                        predict=predictor(version,condition,seed,ckpt,expected)
                    value=evaluate(predict,rows);value.update(version=version,phase=phase,seed=seed,condition=condition,checkpoint_sha256=expected if phase=='final' or version!='R56' else None,evaluation_device='cpu',initial_source='assigned-seed reconstruction' if version=='R56' and phase=='initial' else 'safe JSON checkpoint')
                    value['invariance']=invariance(predict,rows);value['requested_name_and_source_alpha']=coding_counterfactual(predict,rows);value['physics_monotonicity']=physics_monotonicity(predict,[r['problem'] for r in physics]);value['breakdown']=breakdown(rows,value['records']);runs[version+'-'+phase]=value
                    write(destination/f'predictions/{seed}-{condition}-{version}-{phase}.json',value);summary.append({k:value[k] for k in ('version','phase','seed','condition','metrics','checkpoint_sha256')})
            paired[f'{seed}-{condition}']={'R56_to_R57c':compare(runs['R56-final'],runs['R57c-final']),'R57_to_R57c':compare(runs['R57-final'],runs['R57c-final'])}
    means={}
    for condition in CONDITIONS:
        for version in ('R56','R57','R57c'):
            for phase in ('initial','final'):
                selected=[r for r in summary if r['condition']==condition and r['version']==version and r['phase']==phase];entry={}
                for domain,key in [('math','joint_accuracy'),('coding','accuracy'),('physics','mse')]:
                    values=[r['metrics'][domain][key] for r in selected];entry[domain]={'mean':statistics.mean(values),'seed_sd':statistics.stdev(values),'values':values}
                means[f'{condition}/{version}/{phase}']=entry
    result={'source_commit':frozen['source_commit'],'candidate_freeze_sha256':sha(output/'candidate-freeze.json'),'summary':summary,'means':means,'paired':paired,'fit_receipts':receipts,'verified_fits':9,'optimizer_updates':sum(r['optimizer_updates'] for r in receipts),'evaluation_optimizer_updates':0,'evaluation_device':'cpu','seconds':time.monotonic()-started,'prior_classification':'explicit ordinal-decimal output geometry prior; original math core does not participate in ordinal prediction','scientific_qualification':False,'model_selection_performed':False}
    write(destination/'scoreboard.json',result);reports(output,destination,result);return result


def reports(output,destination,result):
    import csv
    lines=['# Focused correction independent comparison','','Explicit ordinal output-geometry prior; math-core topology does not participate. No significance or general superiority claim.','','| Seed | Condition | Version | Phase | Math joint | Coding | Physics MSE |','|---|---|---|---|---:|---:|---:|']
    with (destination/'comparison.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','condition','version','phase','math_joint','coding','physics_mse'])
        for row in result['summary']:
            m=row['metrics'];values=[row['seed'],row['condition'],row['version'],row['phase'],m['math']['joint_accuracy'],m['coding']['accuracy'],m['physics']['mse']];writer.writerow(values);lines.append('| '+' | '.join(map(str,values))+' |')
    lines+=['','## Paired regression flags','']
    regressions=[]
    for fit,comparisons in result['paired'].items():
        for comparison,domains in comparisons.items():
            for domain,value in domains.items():
                lines.append(f"- {fit} {comparison} {domain}: {value['status']}; delta {value['mean_right_minus_left']:+.6f}; descriptive interval {value['interval95']}")
                if value['status']=='REGRESSION':regressions.append(f'{fit}/{comparison}/{domain}')
    with (destination/'COMPARISON.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    status='REGRESSIONS_PRESENT_NO_SUPERIORITY_CLAIM' if regressions else 'NO_OBSERVED_REGRESSION_ON_THIS_FINAL_SET_ONLY'
    resultpath=output/'RESULTS.md'
    if resultpath.exists():resultpath.rename(output/'RESULTS.before-final-scoreboard.md')
    with resultpath.open('x') as f:f.write(f"# Focused correction exploratory results\n\nStatus: {status}. Nine verified fits / {result['optimizer_updates']} updates. Independent CPU evaluation completed; evaluation updates zero.\n\n[Comparison](scoreboard-final/COMPARISON.md) · [JSON](scoreboard-final/scoreboard.json) · [CSV](scoreboard-final/comparison.csv)\n\nExplicit ordinal-decimal output prior; no learned arithmetic topology or scientific qualification claim. Initial/final raw predictions and checkpoint hashes are in scoreboard-final/predictions. All earlier evidence remains unchanged.\n\nRegressions: {regressions}\n")


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args();r=run(args.output);print(json.dumps({'verified_fits':r['verified_fits'],'evaluation_optimizer_updates':0,'scoreboard':str(args.output/'scoreboard-final')}))
if __name__=='__main__':main()
