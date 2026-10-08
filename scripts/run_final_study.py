"""Immutable final study. Same frozen transformer/head; separate supervision and cost."""
import argparse,copy,json,time,subprocess
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from mindscape.core.config import load_config
from mindscape.core.schema import State
from mindscape.core.serialization import decode
from mindscape.data.generation import load_dataset
from mindscape.data.backends import stable_hash
from mindscape.models.final_backend import FrozenFeatures,FinalBackend
from mindscape.models.final_policy import FinalPolicy
from mindscape.training.claims import supervised_arrays,collect_experience,CONDITIONS
from mindscape.models.study_encoding import policy_features,claim_labels
from mindscape.memory.episodic import EpisodicMemory
from mindscape.evaluation.runner import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',required=True);p.add_argument('--cache',default='work/final_features');p.add_argument('--config',default='configs/experiments/final_study.toml');p.add_argument('--output',required=True);args=p.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);config=load_config(args.config);splits,manifest=load_dataset(args.dataset)
 encoder=FrozenFeatures(args.model,args.cache)
 protocol={'config':config,'dataset_hash':stable_hash(manifest),'manifest':manifest,'backbone_revision':encoder.revision,'backbone_parameters':encoder.parameter_count,'device':encoder.device,'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'fixed_protocol':'docs/final_experimental_protocol.md','head_input':'frozen pooled tokens plus categorical numeric features; no arithmetic features','cache':'Shared target-free frozen features. Encoder cache misses charged separately; head calls measured for every inference. Shared setup not hidden in training efficiency.'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[];models={}
 def train(condition,budget,seed,variant='full',no_state=False):
  path=out/f'{variant}_{condition}_n{budget}_s{seed}';path.mkdir(exist_ok=False);backend=FinalBackend(encoder,seed);m=FinalPolicy(backend,condition,dream=False,no_state=no_state)
  ids=manifest['nested_subsets'][str(budget)];byid={e.example_id:e for e in splits['train']};es=[byid[i] for i in ids];started=time.perf_counter();calls=encoder.calls;tokens=encoder.tokens;experience=None
  if condition=='experiential':
   memory=EpisodicMemory(path/'episodes.sqlite');x,y,experience=collect_experience(es,m,memory,seed,config['attempts_per_problem'])
   # Use exactly the same accepted rows/update budget in the no-memory control.
   if variant!='no_episodic_memory':
    xs=[];ys=[]
    for _,ep in reversed(memory.retrieve('train',limit=len(es))):
     for r in ep['trajectory']:
      if r['reward']==1:xs.append(policy_features(decode(State,r['state'])));ys.append(claim_labels(r['action']['value']))
    x,y=np.asarray(xs),np.asarray(ys,dtype=int)
  else:x,y=supervised_arrays(es,condition,no_state=no_state)
  stats=backend.fit(x,y,config['steps'],seed,2 if condition in ('trajectory','experiential') else 9) if len(x) else {'gradient_steps':0,'minibatch_rows_processed':0}
  stats.update(training_rows=len(x),training_problem_count=len(es),transformer_calls=encoder.calls-calls,transformer_tokens=encoder.tokens-tokens,feedback_attempts=(experience or {}).get('attempts',0),examples_seen=config['steps']*64,training_time=time.perf_counter()-started)
  cfg=dict(condition=condition,budget=budget,seed=seed,variant=variant,steps=config['steps'])
  regime={'answer_only':'answer_only','structured':'structured','trajectory':'trajectory_supervised','experiential':'experiential'}[condition]
  meta=dict(config=cfg,dataset_hash=stable_hash(manifest),regime=regime,training_time=stats['training_time'],training_rows=len(x),training_ids=ids,experience=experience,resources=stats,checkpoint=str((path/'checkpoint').resolve()),parameter_count=backend.parameter_count,trainable_parameters=backend.trainable_parameters)
  m.training_metadata=meta;backend.save(path/'checkpoint');(path/'metadata.json').write_text(json.dumps(meta,indent=2));(path/'config.yaml').write_text(json.dumps(cfg,indent=2));(path/'metrics.json').write_text(json.dumps(stats,indent=2))
  # Actually reload each checkpoint before evaluation.
  reloaded=FinalBackend.load(path/'checkpoint',encoder)
  if len(x) and not np.array_equal(backend.logits(x[:1]),reloaded.logits(x[:1])):raise ValueError('Reload mismatch')
  m.backend=reloaded;return m,meta
 def evaluate(m,meta,variant='full'):
  for split in ['test','ood_test']:
   tokens=encoder.tokens;calls=encoder.calls
   folder,metrics=run(m,args.dataset,out/'evaluations',split,meta['regime'],meta['config']['budget'],notes='Frozen final study; 100 numerical candidates; no inference correctness feedback.')
   rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()];byid={e.example_id:e for e in splits[split]};grouped=defaultdict(list)
   for r in rows:
    e=byid[r['example_id']];grouped[(e.metadata['structural_category'],e.metadata['difficulty']['carry_count'])].append(r)
   rec=dict(condition=m.condition,variant=variant,seed=meta['config']['seed'],budget=meta['config']['budget'],split=split,directory=str(folder),metrics=metrics,training_time=meta['training_time'],training_rows=meta['training_rows'],parameter_count=m.parameter_count,trainable_parameters=m.backend.trainable_parameters,resources=meta['resources'],inference_transformer_tokens=encoder.tokens-tokens,inference_transformer_calls=encoder.calls-calls,error_counts=dict(Counter(r['error_type'] for r in rows)),difficulty_errors=[dict(structure=k[0],carry_count=k[1],count=len(v),accuracy=sum(r['correct'] for r in v)/len(v),errors=dict(Counter(r['error_type'] for r in v))) for k,v in sorted(grouped.items())])
   (folder/'plots').mkdir();records.append(rec)
   with (out/'runs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  print(f'{variant} {m.condition} n{meta["config"]["budget"]} s{meta["config"]["seed"]}',flush=True)
 for budget in config['budgets']:
  for seed in config['seeds']:
   for condition in CONDITIONS:
    m,meta=train(condition,budget,seed);evaluate(m,meta)
    if budget==50:models[(condition,seed)]=(m,meta)
 for seed in config['seeds']:
  base,meta=models[('trajectory',seed)]
  for variant,options in [('no_dream',{}),('dream',{'dream':True}),('no_environment_interaction',{'no_interaction':True}),('no_goal_evaluator',{'no_goal':True}),('matched_information_control',{'use_memory':False})]:
   m=copy.copy(base)
   for k,v in options.items():setattr(m,k,v)
   evaluate(m,meta,variant)
  m,meta=train('trajectory',50,seed,'no_explicit_state',True);evaluate(m,meta,'no_explicit_state')
  m,meta=train('experiential',50,seed,'no_episodic_memory');evaluate(m,meta,'no_episodic_memory')
 (out/'resource_totals.json').write_text(json.dumps({'unique_transformer_batches':encoder.calls,'unique_transformer_input_tokens':encoder.tokens,'feature_requests':encoder.requests,'backbone_parameters':encoder.parameter_count},indent=2))
 print(out,flush=True)
if __name__=='__main__':main()
