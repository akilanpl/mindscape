"""Same-set overfit check on final primary checkpoints; never a held-out score."""
import argparse,json
from pathlib import Path
from mindscape.models.final_backend import FrozenFeatures,FinalBackend
from mindscape.models.final_policy import FinalPolicy
from mindscape.data.generation import load_dataset

def main():
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--cache',default='work/final_features');a=p.parse_args();root=Path(a.results);s,manifest=load_dataset(a.dataset);encoder=FrozenFeatures(a.model,a.cache);report=[];byid={e.example_id:e for e in s['train']}
 for condition in ['answer_only','structured','trajectory','experiential']:
  path=root/f'full_{condition}_n10_s0';meta=json.loads((path/'metadata.json').read_text());b=FinalBackend.load(path/'checkpoint',encoder);m=FinalPolicy(b,condition,dream=False);es=[byid[i] for i in meta['training_ids']];correct=sum(m.predict(e.view('experiential')).answer==e.target_answer for e in es)
  report.append(dict(condition=condition,diagnostic='training_subset_same_set',n=len(es),seed=0,accuracy=correct/len(es),received_complete_trajectories=condition=='trajectory',interpretation='Memorization check only; C may have incomplete positive experiences.'))
 out=root/'sanity.json'
 if out.exists():raise ValueError('Refuse overwrite')
 out.write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
