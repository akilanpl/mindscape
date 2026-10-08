"""Independent artifact audit; declarations are not substitutes for measured checks."""
import argparse,json
from pathlib import Path
from collections import Counter
from mindscape.data.generation import load_dataset
from mindscape.core.serialization import decode
from mindscape.core.schema import State
from mindscape.environments.multiplication.claims import expected_value

def main():
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--dataset',required=True);a=p.parse_args();root=Path(a.results);s,m=load_dataset(a.dataset);runs=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()];checks=[]
 def check(name,value):
  checks.append(dict(check=name,passed=bool(value)))
  if not value:raise AssertionError(name)
 primary=[r for r in runs if r['variant']=='full'];check('84 primary trained models /168 split evaluations',len(primary)==168)
 keys=Counter((r['condition'],r['budget'],r['seed'],r['split']) for r in primary);check('Unique primary matrix',len(keys)==168 and all(v==1 for v in keys.values()));check('Same parameter count',len({r['parameter_count'] for r in primary})==1)
 hold={e.example_id for split in ['validation','test','ood_test'] for e in s[split]};train={e.example_id for e in s['train']};check('No held-out training identities',not train&hold)
 allrows=0;decision_count=0
 for r in runs:
  folder=Path(r['directory']);cfg=json.loads((folder/'config.json').read_text());meta=cfg['training_metadata'];check('Checkpoint exact nested problem subset '+folder.name,set(meta['training_ids'])==set(m['nested_subsets'][str(r['budget'])]))
  rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()];allrows+=len(rows);valid=True
  for row in rows:
   for d in (row.get('diagnostics') or {}).get('decisions',[]):
    decision_count+=1;valid &= d['total_candidate_actions']==100 and len(d['valid_actions'])==100
    valid &= not (row['diagnostics'].get('oracle_feedback_visible',True))
    if 'actual_result' in d:
     valid &= not d['hypothetical'] and not d['actual_result']['hypothetical'] and d['actual_result']['value']==d['model_prediction']
     expected=expected_value(decode(State,d['state_before']));valid &= d['correct_action']==expected and d['verifier_judgment']==(expected==d['selected_value'])
    else:valid &= d['hypothetical'] and row['predicted_trajectory'] is None
    valid &= all(x['hypothetical'] for x in d['hypothetical_rollouts'])
  check('Candidate/evidence/posthoc separation '+folder.name,valid)
 check('Every evaluation has required artifacts',all(all((Path(r['directory'])/f).exists() for f in ['config.yaml','metrics.json','predictions.jsonl','summary.md','plots/outcomes.svg']) for r in runs))
 bykey={(r['variant'],r['condition'],r['seed'],r['budget'],r['split']):r for r in runs}
 for seed in [0,1,2]:
  for split in ['test','ood_test']:
   def outcomes(variant,c):
    r=bykey[(variant,c,seed,50,split)];return [(x['predicted_answer'],x['correct']) for x in map(json.loads,(Path(r['directory'])/'predictions.jsonl').read_text().splitlines())]
   check(f'Matched information policy exact predictions seed{seed} {split}',outcomes('matched_information_control','trajectory')==outcomes('full','trajectory'))
   check(f'Memory matched-order/update predictions seed{seed} {split}',outcomes('no_episodic_memory','experiential')==outcomes('full','experiential'))
 source=Path('src/mindscape/models/final_policy.py').read_text();check('No inference numerical oracle call',all(x not in source for x in ['expected_value','verify_claims','reference()','is_goal_reached()']))
 (root/'fairness_audit.json').write_text(json.dumps(dict(checks=checks,check_count=len(checks),all_passed=True,prediction_count=allrows,decision_count=decision_count,limits='Input/supervision/compute differences remain documented; syntactic audit is not proof of universal fairness.'),indent=2));print(len(checks),'checks passed')
if __name__=='__main__':main()
