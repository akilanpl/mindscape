"""Historical pipeline diagnostics; does not change primary training or checkpoints."""
import json, time
from pathlib import Path
import numpy as np
from mindscape.data.generation import load_dataset
from mindscape.models.numpy_backend import NumpyMLP
from mindscape.training.claims import supervised_arrays
from mindscape.training.optimizer import fit_fixed
from mindscape.models.study import StudyModel

def main():
    splits,_=load_dataset('datasets/generated/research_v1')
    out=Path('experiments/final_audit');out.mkdir(exist_ok=False)
    report=[]
    subsets={'tiny10':splits['train'][:10], 'single_digit':[e for e in splits['train'] if e.metadata['structural_category']=='1x1']}
    for name,examples in subsets.items():
        if not examples: continue
        for condition in ['answer_only','trajectory']:
            x,y=supervised_arrays(examples,condition)
            backend=NumpyMLP([10]*8+[2],64,0)
            t=time.perf_counter()
            _,metrics=fit_fixed(backend,x,y,x,y,steps=6000,batch_size=64,learning_rate=.003,seed=0,validation_every=1000)
            model=StudyModel(backend,condition,dream=False)
            correct=sum(model.predict(e.view('experiential')).answer==e.target_answer for e in examples)
            report.append({'diagnostic':name,'condition':condition,'n':len(examples),'rows':len(x),'same_set_accuracy':correct/len(examples),'seconds':time.perf_counter()-t,'metrics':metrics})
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
