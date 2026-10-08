"""Ceiling, memorization and random references; no learned neural claims."""
import argparse,json
from pathlib import Path
import numpy as np
from mindscape.data.generation import load_dataset
from mindscape.models.reference import ReferenceModel
from mindscape.models.base import Prediction
from mindscape.evaluation.runner import run
class Memorization:
 identifier='training_operand_memorization';parameter_count=0
 def __init__(self,es):self.table={tuple(sorted(e.observation['operands'])):e.target_answer for e in es}
 def predict(self,view):return Prediction(self.table.get(tuple(sorted(view['observation']['operands']))),model_calls=0)
class Random:
 identifier='uniform_integer_random_reference';parameter_count=0
 def __init__(self,seed=0):self.rng=np.random.default_rng(seed)
 def predict(self,view):return Prediction(int(self.rng.integers(0,100000000)),model_calls=0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);s,_=load_dataset(a.dataset);records=[]
 for model in [ReferenceModel(),Memorization(s['train']),Random()]:
  for split in ['test','ood_test']:
   folder,metrics=run(model,a.dataset,out/'evaluations',split,notes='Non-neural interpretive reference, not a learned efficiency comparison.');(folder/'plots').mkdir();from analyze_research_study import bars;bars(folder/'plots'/'outcomes.svg','Non-neural reference rates (%)',['accuracy','grounded','goal'],[100*metrics[k] for k in ['accuracy','grounded_rate','goal_success_rate']]);records.append(dict(model=model.identifier,split=split,directory=str(folder),metrics=metrics))
 (out/'references.json').write_text(json.dumps(records,indent=2));print(records)
if __name__=='__main__':main()
