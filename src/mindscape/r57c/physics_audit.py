"""Read-only independent-grid physics diagnosis of immutable R57 checkpoints."""
from __future__ import annotations
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import torch
from mindscape.r57.model import make_model
from mindscape.r57.mechanism import permute_graph,perturb_graph
from mindscape.r56.learning import load_checkpoint


def diagnostic_grid():
 return [{'domain':'physics','mass':m,'dt':t,'force':f,'velocity':[.3,-.7],'concept_ids':[2]}
         for m in (.7,2.3,6.1) for t in (.2,.8) for f in ([1.7,-2.1],[-1.7,2.1])]


def exact(p):return [f*p['dt']/p['mass'] for f in p['force']]
def mse(a,b):return sum((x-y)**2 for x,y in zip(a,b))/len(a)


def predict(model,p,perturb=None):
 prepared=model.prepare(p)
 if perturb:prepared.graph=perturb_graph(prepared.graph,perturb)
 with torch.no_grad():return model(prepared)['physics_delta'].tolist()


def baselines(rows,grid):
 physics=[r for r in rows if r['problem']['domain']=='physics']
 targets=torch.tensor([r['labels']['physics_delta'] for r in physics],dtype=torch.float64)
 def features(p):return list(p['force'])+[p['mass'],p['dt']]+list(p['velocity'])+[1.]
 design=torch.tensor([features(r['problem']) for r in physics],dtype=torch.float64)
 weights=torch.linalg.lstsq(design,targets).solution
 mean=targets.mean(0).tolist()
 return {'training_only_constant':mean,'records':[{'problem':p,'reference':exact(p),'zero':[0,0],'constant':mean,
          'linear_observed_only':(torch.tensor(features(p),dtype=torch.float64)@weights).tolist(),
          'analytic':exact(p)} for p in grid],
         'linear_features':'rawforce,mass,dt,velocity,intercept; no force*dt/m feature','optimizer_updates':0}


def audit_model(model,grid):
 records=[]
 for p in grid:
  prediction=predict(model,p);reference=exact(p)
  record={'problem':p,'prediction':prediction,'reference':reference,'mse':mse(prediction,reference),'probes':{},'derivatives':{}}
  for kind in ('permuted','disconnected','reversed','relation-shuffled'):record['probes'][kind]=predict(model,p,kind)
  for var in ('force_x','force_y','mass','dt'):
   changed=copy.deepcopy(p);eps=.01
   if var.startswith('force'):changed['force'][0 if var.endswith('x') else 1]+=eps
   else:changed[var]+=eps
   measured=[(a-b)/eps for a,b in zip(predict(model,changed),prediction)]
   reference_derivative=[(a-b)/eps for a,b in zip(exact(changed),reference)]
   record['derivatives'][var]={'measured':measured,'reference':reference_derivative}
  records.append(record)
 return records


def gate_audit(model,p):
 if model.condition=='matched-gru':return None
 results={};gate=model.core.concepts.gate
 original=gate.detach().clone()
 for mode in ('trained','gate-zero','id-permuted','unit-id0'):
  with torch.no_grad():gate.copy_(torch.zeros_like(original) if mode=='gate-zero' else original)
  local={**p,'concept_ids':[3] if mode=='id-permuted' else [0] if mode=='unit-id0' else [2]}
  model.zero_grad(set_to_none=True);out=model.predict(local)['physics_delta'];loss=(out-torch.tensor(exact(local))).square().mean();loss.backward()
  results[mode]={'prediction':out.detach().tolist(),'mse':float(loss.detach()),'gate_raw':float(gate.detach()),
                 'gate_tanh':float(gate.detach().tanh()),'gate_gradient':None if gate.grad is None else float(gate.grad),
                 'embedding_gradient_norm':None if model.core.concepts.weight.grad is None else float(model.core.concepts.weight.grad.norm()),
                 'feature_gradient_norm':float(model.core.features.weight.grad.norm()),
                 'physics_head_gradient_norm':float(model.physics_head[0].weight.grad.norm())}
 with torch.no_grad():gate.copy_(original)
 return results


def run(root,output):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 grid=diagnostic_grid();records=[];baseline={};hashes={}
 for seed in (17,29,43):
  rows=json.loads((root/f'data-v1/train/{seed}.json').read_text());baseline[str(seed)]=baselines(rows,grid)
  for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
   folder=root/f'checkpoints/{seed}-{condition}';result=json.loads((folder/'result.json').read_text())
   stages={};gates={}
   for stage,key in (('initial','initial_sha256'),('final','checkpoint_sha256')):
    path=folder/f'{stage}.json';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=result[key]:raise ValueError('Immutable checkpoint hash differs')
    model=make_model(condition,seed);load_checkpoint(model,path,sha);model.eval()
    stages[stage]=audit_model(model,grid);gates[stage]=gate_audit(model,grid[0]);hashes[str(path)]=sha
   records.append({'seed':seed,'condition':condition,'stages':stages,'gate_probes':gates})
 report={'status':'EXPLORATORY_DIAGNOSIS_NOT_CAUSAL_TRAINING_PROOF','optimizer_updates':0,'grid':grid,
         'records':records,'training_only_baselines':baseline,'immutable_checkpoint_hashes':hashes,
         'scope':'new independent diagnostic grid; no original final/protected evaluation used'}
 (output/'raw-paired.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
 a=parser.parse_args();run(a.root,a.output)
