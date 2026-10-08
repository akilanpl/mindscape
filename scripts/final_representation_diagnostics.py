"""Final representation diagnostics on train/validation, never test-driven tuning."""
import argparse,json
from pathlib import Path
import numpy as np
from mindscape.data.generation import load_dataset
from mindscape.models.final_backend import FrozenFeatures,FinalBackend
from mindscape.models.final_policy import FinalPolicy
from mindscape.training.claims import supervised_arrays

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--cache',default='work/final_features');p.add_argument('--output',required=True);a=p.parse_args();s,_=load_dataset(a.dataset);encoder=FrozenFeatures(a.model,a.cache);rows=[]
 for name,es in [('single_digit',[e for e in s['train'] if e.metadata['structural_category']=='1x1']),('mixed100',s['train'][:100])]:
  for condition in ['answer_only','trajectory']:
   b=FinalBackend(encoder,0);x,y=supervised_arrays(es,condition);stats=b.fit(x,y,1200,0,2 if condition=='trajectory' else 9);m=FinalPolicy(b,condition,dream=False)
   for split,examples in [('same_training',es)]+([('validation',s['validation'])] if name=='mixed100' else []):
    xx,yy=supervised_arrays(examples,condition);pred=b.generate(xx);active=2 if condition=='trajectory' else 9
    rows.append(dict(diagnostic=name,condition=condition,split=split,seed=0,n=len(examples),exact_answer_accuracy=sum(m.predict(e.view('experiential')).answer==e.target_answer for e in examples)/len(examples),local_label_accuracy=float(np.mean(np.all(pred[:,:active]==yy[:,:active],axis=1))),carry_label_accuracy=float(np.mean(pred[:,1]==yy[:,1])) if condition=='trajectory' else None,optimizer=stats))
 out=Path(a.output)
 if out.exists():raise ValueError('Refuse overwrite')
 out.write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
