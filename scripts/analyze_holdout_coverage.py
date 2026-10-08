"""Posthoc carry/context novelty strata; never visible to training or prediction."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from mindscape.data.generation import load_dataset
from mindscape.core.schema import State
from mindscape.core.serialization import decode
from mindscape.environments.multiplication.claims import local_inputs

def contexts(e):return {(*local_inputs(decode(State,t['state_before'])),t['state_before']['phase']) for t in e.target_trajectory}
def carries(e):return tuple((t['state_before']['carry'],t['result']['carry']) for t in e.target_trajectory)
def main():
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--dataset',required=True);a=p.parse_args();root=Path(a.results);s,m=load_dataset(a.dataset);byid={e.example_id:e for es in s.values() for e in es};runs=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()];output=[];counts=[]
 for budget in [10,25,50,100,250,500,1000]:
  train=[byid[i] for i in m['nested_subsets'][str(budget)]];seen=set().union(*(contexts(e) for e in train));seen_carries={carries(e) for e in train}
  for split in ['test','ood_test']:
   novelty={e.example_id:dict(unseen_local_contexts=len(contexts(e)-seen),total_distinct_contexts=len(contexts(e)),unseen_carry_signature=carries(e) not in seen_carries,structure=e.metadata['structural_category']) for e in s[split]}
   counts.append(dict(budget=budget,split=split,examples=len(novelty),examples_with_unseen_local_context=sum(v['unseen_local_contexts']>0 for v in novelty.values()),examples_with_unseen_carry_signature=sum(v['unseen_carry_signature'] for v in novelty.values()),interpretation='Posthoc descriptive strata; structural holdouts were specified before training.'))
   for r in runs:
    if r['variant']!='full' or r['budget']!=budget or r['split']!=split:continue
    groups=defaultdict(list)
    for row in map(json.loads,(Path(r['directory'])/'predictions.jsonl').read_text().splitlines()):
     n=novelty[row['example_id']];groups[(n['unseen_local_contexts']>0,n['unseen_carry_signature'],n['structure'])].append(row)
    for key,rows in groups.items():output.append(dict(condition=r['condition'],seed=r['seed'],budget=budget,split=split,unseen_local_context=key[0],unseen_carry_signature=key[1],structure=key[2],n=len(rows),accuracy=sum(x['correct'] for x in rows)/len(rows),groundedness=sum(x['grounded'] for x in rows)/len(rows)))
 out=root/'holdout_coverage.json'
 if out.exists():raise ValueError('Refuse overwrite')
 out.write_text(json.dumps(dict(dataset_coverage=counts,prediction_strata=output),indent=2));print(counts[-2:])
if __name__=='__main__':main()
