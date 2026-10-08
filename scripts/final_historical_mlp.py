"""Historical MLP on the fresh final dataset, explicitly a reference study."""
import argparse,json
from pathlib import Path
from mindscape.data.generation import load_dataset
from mindscape.training.claims import CONDITIONS,train_claims
from mindscape.models.study import StudyModel
from mindscape.evaluation.runner import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);s,m=load_dataset(a.dataset);records=[]
 for seed in [0,1,2]:
  for condition in CONDITIONS:
   cfg=dict(condition=condition,budget=1000,seed=seed,hidden=64,steps=600,batch_size=64,validation_every=100,learning_rate=.003,attempts_per_problem=64,dream=False)
   model,meta=train_claims(s,m,cfg,out/f'{condition}_s{seed}');model=StudyModel.load(out/f'{condition}_s{seed}'/'checkpoint')
   for split in ['test','ood_test']:
    folder,metrics=run(model,a.dataset,out/'evaluations',split,meta['regime'],1000,notes='Historical 8722-parameter MLP fresh-dataset reference; differs in representation/loss/update budget from main final study.');(folder/'plots').mkdir()
    records.append(dict(condition=condition,seed=seed,split=split,directory=str(folder),metrics=metrics,training_time=meta['training_time']))
   print(condition,seed,flush=True)
 (out/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));print(out)
if __name__=='__main__':main()
